"""Load official price reports into the catalog and price tables.

A load reads every row, sends the ones it cannot trust to quarantine with a reason, and
stores the rest as validated official prices. Loading the same file twice is a no-op,
and prices already stored from another export of the same report are skipped.
"""

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from hashlib import sha256

from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from agrodash.accounts.models import User
from agrodash.catalog.models import BaseUnit, MarketKind, Unit
from agrodash.catalog.resolver import CatalogResolver
from agrodash.ingestion.models import (
    IngestionRun,
    IngestionStatus,
    IngestionTrigger,
    QuarantinedRow,
    RejectionReason,
    SourceLayout,
)
from agrodash.ingestion.normalize import (
    InvalidValue,
    clean_name,
    fold,
    name_key,
    normalize_category,
    parse_amount,
    parse_week_date,
)
from agrodash.ingestion.readers import UnreadableFile, detect_layout, is_blank, read_rows
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType

MAX_FILE_BYTES = 50 * 1024 * 1024
# The wholesale report has no market column; it is collected at Mercado Nuevo.
WHOLESALE_MARKET = "Mercado Nuevo"
WHOLESALE_COLUMNS = 8
RETAIL_COLUMNS = 14
RETAIL_MARKET_COLUMNS = range(4, 11)
SUPERMARKET_NAMES = {"supermercado"}
INSERT_BATCH = 5000
AMOUNT_PLACES = Decimal("0.01")
PER_BASE_UNIT_PLACES = Decimal("0.0001")


class IngestionError(Exception):
    """The file could not be loaded; its run is recorded as failed with this message."""


@dataclass
class _Observation:
    row_number: int
    row: list[object]
    raw_name: str
    category: str
    unit: Unit
    market_name: str
    market_kind: MarketKind
    price_type: PriceType
    observed_on: date
    amount: Decimal


def ingest_file(
    session: Session,
    content: bytes,
    file_name: str,
    trigger: IngestionTrigger,
    started_by: User | None = None,
) -> IngestionRun:
    """Load one report file and return its run (the earlier run if it was already loaded)."""
    digest = sha256(content).hexdigest()
    loaded = _succeeded_run(session, digest)
    if loaded is not None:
        return loaded

    run = IngestionRun(
        file_name=file_name[:255],
        file_sha256=digest,
        trigger=trigger,
        started_by_id=started_by.id if started_by else None,
    )
    try:
        rows = _read(content, file_name)
        run.layout = detect_layout(rows)
        session.add(run)
        session.flush()
        _Load(session, run).load(rows)
        run.status = IngestionStatus.SUCCEEDED
        run.finished_at = datetime.now(UTC)
        session.commit()
    except Exception as error:
        session.rollback()
        message = _failure_message(session, error, digest)
        session.add(
            IngestionRun(
                file_name=file_name[:255],
                file_sha256=digest,
                layout=run.layout,
                trigger=trigger,
                started_by_id=run.started_by_id,
                status=IngestionStatus.FAILED,
                error=message[:500],
                finished_at=datetime.now(UTC),
            )
        )
        session.commit()
        raise IngestionError(message) from error
    return run


def _succeeded_run(session: Session, digest: str) -> IngestionRun | None:
    return session.scalar(
        select(IngestionRun).where(
            IngestionRun.file_sha256 == digest, IngestionRun.status == IngestionStatus.SUCCEEDED
        )
    )


def _read(content: bytes, file_name: str) -> list[list[object]]:
    if len(content) > MAX_FILE_BYTES:
        raise IngestionError(f"{file_name} is larger than {MAX_FILE_BYTES // 2**20} MB")
    try:
        rows = read_rows(content, file_name)
    except UnreadableFile as error:
        raise IngestionError(str(error)) from error
    if all(is_blank(row) for row in rows):
        raise IngestionError(f"{file_name} has no rows")
    return rows


def _failure_message(session: Session, error: Exception, digest: str) -> str:
    if isinstance(error, IngestionError):
        return str(error)
    if isinstance(error, IntegrityError) and _succeeded_run(session, digest):
        return "Another run loaded this file at the same time"
    return f"Unexpected error: {error.__class__.__name__}: {error}"


class _Load:
    def __init__(self, session: Session, run: IngestionRun) -> None:
        self.session = session
        self.run = run
        self.resolver = CatalogResolver(session)
        self.rejections: list[dict] = []
        self.observations: list[_Observation] = []
        # Observations that passed every check, whether new or already stored.
        self.accepted = 0

    def load(self, rows: list[list[object]]) -> None:
        if self.run.layout is SourceLayout.RETAIL:
            if len(rows[0]) < RETAIL_COLUMNS:
                raise IngestionError("The retail header needs 7 market columns before Semana")
            markets = [clean_name(rows[0][column]) for column in RETAIL_MARKET_COLUMNS]
            self._read_rows(rows[1:], 2, lambda n, row: self._retail_row(n, row, markets))
        else:
            self._read_rows(rows, 1, self._wholesale_row)
        self._store()
        if self.rejections and not self.accepted:
            # Most likely not a price report, or one in a layout we misread. Failing keeps
            # the file loadable again instead of marking it done with nothing stored.
            raise IngestionError("No row could be loaded; check that the file is a price report")

    def _read_rows(self, rows, first_line, parse_row) -> None:
        for row_number, row in enumerate(rows, start=first_line):
            if is_blank(row):
                continue
            self.run.rows_read += 1
            parse_row(row_number, list(row))

    def _wholesale_row(self, row_number: int, row: list[object]) -> None:
        if len(row) < WHOLESALE_COLUMNS or _letters(row[2]) != "mayorista":
            self._reject(row_number, row, RejectionReason.UNEXPECTED_LAYOUT, "expected 'Mayorista'")
            return
        name, rubro, _, unit_text, amount_cell, week, month, year = row[:WHOLESALE_COLUMNS]
        amounts = self._amounts(row_number, row, [(WHOLESALE_MARKET, amount_cell)])
        self._observe(
            row_number,
            row,
            amounts,
            (name, rubro, unit_text, week, month, year),
            PriceType.WHOLESALE,
        )

    def _retail_row(self, row_number: int, row: list[object], markets: list[str]) -> None:
        if len(row) < RETAIL_COLUMNS or _letters(row[2]) != "minorista":
            self._reject(row_number, row, RejectionReason.UNEXPECTED_LAYOUT, "expected 'Minorista'")
            return
        cells = [
            (market, row[column])
            for market, column in zip(markets, RETAIL_MARKET_COLUMNS, strict=True)
        ]
        amounts = self._amounts(row_number, row, cells)
        name, rubro, _, unit_text = row[:4]
        week, month, year = row[11:14]
        self._observe(
            row_number, row, amounts, (name, rubro, unit_text, week, month, year), PriceType.RETAIL
        )

    def _amounts(self, row_number, row, cells) -> list[tuple[str, Decimal]]:
        amounts = []
        for market, cell in cells:
            try:
                amount = parse_amount(cell)
            except InvalidValue as error:
                self._reject(row_number, row, RejectionReason.INVALID_PRICE, f"{market}: {error}")
                continue
            if amount is None:
                self.run.values_missing += 1
            else:
                amounts.append((market, amount))
        return amounts

    def _observe(self, row_number, row, amounts, fields, price_type: PriceType) -> None:
        if not amounts:
            return
        name, rubro, unit_text, week, month, year = fields
        try:
            observed_on = parse_week_date(week, month, year)
        except InvalidValue as error:
            self._reject(row_number, row, RejectionReason.INVALID_DATE, str(error))
            return
        unit = self.resolver.unit(str(unit_text or ""))
        if unit is None:
            detail = f"unit {clean_name(unit_text or '')!r} has no known size"
            self._reject(row_number, row, RejectionReason.UNKNOWN_UNIT, detail)
            return
        for market, amount in amounts:
            self.observations.append(
                _Observation(
                    row_number=row_number,
                    row=row,
                    raw_name=str(name),
                    category=normalize_category(rubro),
                    unit=unit,
                    market_name=market,
                    market_kind=_market_kind(market, price_type),
                    price_type=price_type,
                    observed_on=observed_on,
                    amount=amount,
                )
            )

    def _store(self) -> None:
        default_units = _majority(
            (name_key(observation.raw_name), observation.unit.base_unit)
            for observation in self.observations
        )
        resolved = [
            (
                observation,
                self.resolver.product(
                    observation.raw_name,
                    observation.category,
                    default_units[name_key(observation.raw_name)],
                ),
                self.resolver.market(observation.market_name, observation.market_kind),
            )
            for observation in self.observations
        ]
        self.session.flush()
        product_ids = {product.id for _, product, _ in resolved}
        series_units = self._series_base_units(resolved, product_ids)
        stored = self._stored_keys(product_ids)
        seen = set()
        prices = []
        for observation, product, market in resolved:
            unit = observation.unit
            series_unit = series_units[(product.id, observation.price_type)]
            if unit.base_unit != series_unit:
                detail = (
                    f"{unit.label} cannot join the {observation.price_type} series of "
                    f"{product.name}, priced per {series_unit}"
                )
                self._reject_observation(observation, RejectionReason.UNIT_MISMATCH, detail)
                continue
            key = (product.id, market.id, observation.observed_on, observation.price_type)
            if key in seen:
                detail = f"{product.name} already has a price at {market.name} that week"
                self._reject_observation(observation, RejectionReason.DUPLICATE_PRICE, detail)
                continue
            amount = observation.amount.quantize(AMOUNT_PLACES)
            per_base_unit = (observation.amount / unit.factor).quantize(PER_BASE_UNIT_PLACES)
            if amount <= 0 or per_base_unit <= 0:
                detail = f"{observation.amount} is too small to store"
                self._reject_observation(observation, RejectionReason.INVALID_PRICE, detail)
                continue
            seen.add(key)
            self.accepted += 1
            if key in stored:
                continue
            prices.append(
                {
                    "product_id": product.id,
                    "market_id": market.id,
                    "unit_id": unit.id,
                    "ingestion_run_id": self.run.id,
                    "observed_on": observation.observed_on,
                    "price_type": observation.price_type,
                    "source": PriceSource.OFFICIAL,
                    "status": PriceStatus.VALIDATED,
                    "amount": amount,
                    "price_per_base_unit": per_base_unit,
                }
            )
        self._insert(Price, prices)
        self._insert(QuarantinedRow, self.rejections)
        self.run.prices_loaded = len(prices)
        # A retail row can have several rejected cells; count each row once.
        self.run.rows_rejected = len({rejection["row_number"] for rejection in self.rejections})

    def _series_base_units(self, resolved, product_ids: set[int]) -> dict[tuple, BaseUnit]:
        """Base unit of each (product, price type) series: what stored prices use, else
        what most rows of this file use. Wholesale and retail may differ (melons are sold
        by the pound wholesale and by the piece retail), but one series never mixes them.
        """
        counts = self.session.execute(
            select(Price.product_id, Price.price_type, Unit.base_unit, func.count())
            .join(Unit, Price.unit_id == Unit.id)
            .where(Price.source == PriceSource.OFFICIAL, Price.product_id.in_(product_ids))
            .group_by(Price.product_id, Price.price_type, Unit.base_unit)
        )
        stored: dict[tuple, tuple[int, BaseUnit]] = {}
        for product_id, price_type, base_unit, count in counts:
            key = (product_id, price_type)
            if key not in stored or count > stored[key][0]:
                stored[key] = (count, base_unit)
        incoming = _majority(
            ((product.id, observation.price_type), observation.unit.base_unit)
            for observation, product, _ in resolved
        )
        return incoming | {key: base_unit for key, (_, base_unit) in stored.items()}

    def _stored_keys(self, product_ids: set[int]) -> set[tuple]:
        price_type = (
            PriceType.RETAIL if self.run.layout is SourceLayout.RETAIL else PriceType.WHOLESALE
        )
        rows = self.session.execute(
            select(Price.product_id, Price.market_id, Price.observed_on, Price.price_type).where(
                Price.source == PriceSource.OFFICIAL,
                Price.price_type == price_type,
                Price.product_id.in_(product_ids),
            )
        )
        return {tuple(row) for row in rows}

    def _insert(self, model, values: list[dict]) -> None:
        for start in range(0, len(values), INSERT_BATCH):
            self.session.execute(insert(model), values[start : start + INSERT_BATCH])

    def _reject(self, row_number, row, reason: RejectionReason, detail: str) -> None:
        self.rejections.append(
            {
                "run_id": self.run.id,
                "row_number": row_number,
                "reason": reason,
                "detail": detail[:200],
                "raw_values": json.dumps(row, ensure_ascii=False, default=str),
            }
        )

    def _reject_observation(self, observation: _Observation, reason, detail: str) -> None:
        self._reject(observation.row_number, observation.row, reason, detail)


def _majority(pairs) -> dict:
    """Most frequent value per key; on ties, the value seen first."""
    counts: dict[object, Counter] = {}
    for key, value in pairs:
        counts.setdefault(key, Counter())[value] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in counts.items()}


def _market_kind(market: str, price_type: PriceType) -> MarketKind:
    if fold(market) in SUPERMARKET_NAMES:
        return MarketKind.SUPERMARKET
    return MarketKind.WHOLESALE if price_type is PriceType.WHOLESALE else MarketKind.RETAIL


def _letters(cell: object) -> str:
    return "".join(character for character in fold(cell) if character.isalpha())
