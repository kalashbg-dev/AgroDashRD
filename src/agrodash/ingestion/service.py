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
    repair_name,
)
from agrodash.ingestion.readers import UnreadableFile, detect_layout, is_blank, read_rows
from agrodash.prices.models import (
    Price,
    PriceSource,
    PriceStatus,
    PriceType,
    per_base_unit,
    round_amount,
)

MAX_FILE_BYTES = 50 * 1024 * 1024
# The wholesale report has no market column; it is collected at Mercado Nuevo.
WHOLESALE_MARKET = "Mercado Nuevo"
WHOLESALE_COLUMNS = 8
WHOLESALE_DATE_COLUMNS = (5, 6, 7)
RETAIL_DATE_COLUMNS = (11, 12, 13)
# How close a repeated name must be to count as a second product rather than a duplicate.
ADJACENT_REPEAT_ROWS = 3
# A base unit used this many weeks is a real way of selling the product, not a typo.
MIN_UNIT_WEEKS = 4
RETAIL_COLUMNS = 14
RETAIL_MARKET_COLUMNS = range(4, 11)
SUPERMARKET_NAMES = {"supermercado"}
INSERT_BATCH = 5000


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
    loaded = loaded_run(session, content)
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


def loaded_run(session: Session, content: bytes) -> IngestionRun | None:
    """The run that already loaded this exact file, if any."""
    return _succeeded_run(session, sha256(content).hexdigest())


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
        # Cells exactly as read, by row number, so quarantine shows the source.
        self.originals: dict[int, list[object]] = {}

    def load(self, rows: list[list[object]]) -> None:
        if self.run.layout is SourceLayout.RETAIL:
            if len(rows[0]) < RETAIL_COLUMNS:
                raise IngestionError("The retail header needs 7 market columns before Semana")
            markets = [clean_name(rows[0][column]) for column in RETAIL_MARKET_COLUMNS]
            self._read_rows(
                rows[1:], 2, RETAIL_DATE_COLUMNS, lambda n, row: self._retail_row(n, row, markets)
            )
        else:
            self._read_rows(rows, 1, WHOLESALE_DATE_COLUMNS, self._wholesale_row)
        self._store()
        if self.rejections and not self.accepted:
            # Most likely not a price report, or one in a layout we misread. Failing keeps
            # the file loadable again instead of marking it done with nothing stored.
            raise IngestionError("No row could be loaded; check that the file is a price report")

    def _read_rows(self, rows, first_line, date_columns, parse_row) -> None:
        numbered = [
            (row_number, list(row))
            for row_number, row in enumerate(rows, start=first_line)
            if not is_blank(row)
        ]
        for row_number, row in self._prepare(numbered, date_columns):
            self.run.rows_read += 1
            parse_row(row_number, row)

    def _prepare(self, numbered, date_columns):
        """Fix what the reports get wrong row by row, keeping each original for quarantine.

        - A row whose date cannot be read ("3333", "|") takes the date of the two rows
          around it when those agree: the reports are written in week blocks. A readable
          date is never changed, and a row with no date at all is left alone: its cells
          were shifted.
        - The same name listed again right below itself (within ADJACENT_REPEAT_ROWS) in
          one week block is two products the report does not tell apart (Berenjena
          Pompadur is 30-43% dearer in its first row); the repeat becomes
          "<name> (fila 2)" so both series are kept. A repeat further away is a block
          that holds two weeks under one label, and stays a duplicate.
        """
        labels = [_date_label(row, date_columns) for _, row in numbered]
        block, listed, last_seen = None, Counter(), {}
        for index, (row_number, row) in enumerate(numbered):
            self.originals[row_number] = list(row)
            label = labels[index]
            before = labels[index - 1] if index else None
            after = labels[index + 1] if index + 1 < len(labels) else None
            if (
                any(label)
                and len(row) > max(date_columns)
                and before is not None
                and before == after
                and label != before
                and not _readable_date(row, date_columns)
            ):
                source = numbered[index - 1][1]
                for column in date_columns:
                    row[column] = source[column]
                label = before
                self.run.rows_corrected += 1
            if label != block:
                block, listed, last_seen = label, Counter(), {}
            name = repair_name(row[0] if row[0] is not None else "")
            key = name_key(name)
            if key in last_seen and index - last_seen[key] <= ADJACENT_REPEAT_ROWS:
                listed[key] += 1
                name = f"{clean_name(name)} (fila {listed[key] + 1})"
            else:
                listed[key] = 0
            last_seen[key] = index
            row[0] = name
            yield row_number, row

    def _wholesale_row(self, row_number: int, row: list[object]) -> None:
        # The type column is often damaged (#NAME?, a rubro, blank) on sound rows; only a
        # row that says it belongs to the other report is out of place.
        if len(row) < WHOLESALE_COLUMNS or _letters(row[2]) == "minorista":
            self._reject(row_number, row, RejectionReason.UNEXPECTED_LAYOUT, "not a wholesale row")
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
        if len(row) < RETAIL_COLUMNS or _letters(row[2]) == "mayorista":
            self._reject(row_number, row, RejectionReason.UNEXPECTED_LAYOUT, "not a retail row")
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
        if not str(name).strip():
            detail = "prices without a product name"
            self._reject(row_number, row, RejectionReason.UNEXPECTED_LAYOUT, detail)
            return
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
        # New products take what most of their rows say, not whatever the first row says.
        default_units = _majority(
            (name_key(observation.raw_name), observation.unit.base_unit)
            for observation in self.observations
        )
        categories = _majority(
            (name_key(observation.raw_name), observation.category)
            for observation in self.observations
        )
        resolved = [
            (
                observation,
                self.resolver.product(
                    observation.raw_name,
                    categories[name_key(observation.raw_name)],
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
            series_bases = series_units[(product.id, observation.price_type)]
            if unit.base_unit not in series_bases:
                priced_per = " or ".join(sorted(series_bases))
                detail = (
                    f"{unit.label} cannot join the {observation.price_type} series of "
                    f"{product.name}, priced per {priced_per}"
                )
                self._reject_observation(observation, RejectionReason.UNIT_MISMATCH, detail)
                continue
            key = (product.id, market.id, observation.observed_on, observation.price_type)
            if key in seen:
                detail = f"{product.name} already has a price at {market.name} that week"
                self._reject_observation(observation, RejectionReason.DUPLICATE_PRICE, detail)
                continue
            amount = round_amount(observation.amount)
            converted = per_base_unit(observation.amount, unit.factor)
            if amount <= 0 or converted <= 0:
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
                    "price_per_base_unit": converted,
                }
            )
        self._insert(Price, prices)
        self._insert(QuarantinedRow, self.rejections)
        self.run.prices_loaded = len(prices)
        # A retail row can have several rejected cells; count each row once.
        self.run.rows_rejected = len({rejection["row_number"] for rejection in self.rejections})

    def _series_base_units(self, resolved, product_ids: set[int]) -> dict[tuple, set[BaseUnit]]:
        """Base units each (product, price type) may be priced in.

        A unit used for at least MIN_UNIT_WEEKS weeks (stored and incoming together) is a
        real way of selling the product and gets its own series: pepino retail moved from
        the pound to the piece in 2024. Wholesale and retail may differ too (melons are
        sold by the pound wholesale and by the piece retail). Otherwise the most used unit
        wins and a stray label is rejected.
        """
        weeks: dict[tuple, Counter] = {}
        stored = self.session.execute(
            select(
                Price.product_id,
                Price.price_type,
                Unit.base_unit,
                func.count(func.distinct(Price.observed_on)),
            )
            .join(Unit, Price.unit_id == Unit.id)
            .where(Price.source == PriceSource.OFFICIAL, Price.product_id.in_(product_ids))
            .group_by(Price.product_id, Price.price_type, Unit.base_unit)
        )
        for product_id, price_type, base_unit, count in stored:
            weeks.setdefault((product_id, price_type), Counter())[base_unit] += count
        incoming: dict[tuple, set] = {}
        for observation, product, _ in resolved:
            key = (product.id, observation.price_type, observation.unit.base_unit)
            incoming.setdefault(key, set()).add(observation.observed_on)
        for (product_id, price_type, base_unit), dates in incoming.items():
            weeks.setdefault((product_id, price_type), Counter())[base_unit] += len(dates)
        return {
            key: {base for base, count in counter.items() if count >= MIN_UNIT_WEEKS}
            or {counter.most_common(1)[0][0]}
            for key, counter in weeks.items()
        }

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
                "raw_values": json.dumps(
                    self.originals.get(row_number, row), ensure_ascii=False, default=str
                ),
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


def _readable_date(row: list[object], date_columns) -> bool:
    try:
        parse_week_date(*(row[column] if column < len(row) else None for column in date_columns))
    except InvalidValue:
        return False
    return True


def _date_label(row: list[object], date_columns) -> tuple[str, ...]:
    return tuple(fold(row[column]) if column < len(row) else "" for column in date_columns)


def _letters(cell: object) -> str:
    return "".join(character for character in fold(cell) if character.isalpha())
