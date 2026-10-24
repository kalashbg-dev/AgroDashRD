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
    letters,
    name_key,
    normalize_category,
    parse_amount,
    parse_week_date,
)
from agrodash.ingestion.readers import UnreadableFile, detect_layout, is_blank, read_rows
from agrodash.ingestion.repair import (
    UnitLedger,
    duplicate_block_rows,
    nullify_copied_market_columns,
    repair_rows,
    repeated_week_rows,
    retail_average_rows,
    retail_weeks_with_wholesale_units,
    shifted_market_cells,
    shifted_price_blocks,
    unit_in_effect,
)
from agrodash.prices.models import (
    Price,
    PriceSource,
    PriceStatus,
    PriceType,
    per_base_unit,
    round_amount,
)

MAX_FILE_BYTES = 50 * 1024 * 1024
# The wholesale report has no market column: its one price stands for Mercadom and Mercado
# Nuevo, the two markets that trade wholesale. Mercado Nuevo keeps its own retail prices.
WHOLESALE_MARKET = "Referencia mayorista (Mercadom y Mercado Nuevo)"
WHOLESALE_COLUMNS = 8
WHOLESALE_DATE_COLUMNS = (5, 6, 7)
WHOLESALE_UNIT_COLUMN = 3
RETAIL_AVERAGES = "a week of retail averages (per lb, unit, dozen or liter) in the wholesale report"
RETAIL_DATE_COLUMNS = (11, 12, 13)
# A base unit used this many weeks is a real way of selling the product, not a typo.
MIN_UNIT_WEEKS = 4
# Audit M2: a conversion deviating this far from the product's usual level at that
# market is a wrongly labelled selling unit, not a price (Zapote priced 10x overnight).
LEVEL_FACTOR = Decimal("4")
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
        # Retail weeks whose unit column was pasted from the wholesale report (G5, Caso 1).
        self._wholesale_unit_rows: set[int] = set()
        # Last retail unit text seen per product, used to recover units in corrupted weeks.
        self._product_units: dict[str, str] = {}
        # Last priced amount per market and product (audit M1/M2): drift and recovery
        # compare consecutive rows, the level band in _store uses committed history.
        self._ledger = UnitLedger()

    def load(self, rows: list[list[object]]) -> None:
        if self.run.layout is SourceLayout.RETAIL:
            if len(rows[0]) < RETAIL_COLUMNS:
                raise IngestionError("The retail header needs 7 market columns before Semana")
            markets = [clean_name(rows[0][column]) for column in RETAIL_MARKET_COLUMNS]
            self._read_rows(
                rows[1:], 2, RETAIL_DATE_COLUMNS, lambda n, row: self._retail_row(n, row, markets)
            )
        else:
            self._read_rows(
                rows, 1, WHOLESALE_DATE_COLUMNS, self._wholesale_row, WHOLESALE_UNIT_COLUMN
            )
        self._store()
        if self.rejections and not self.accepted:
            # Most likely not a price report, or one in a layout we misread. Failing keeps
            # the file loadable again instead of marking it done with nothing stored.
            raise IngestionError("No row could be loaded; check that the file is a price report")

    def _read_rows(self, rows, first_line, date_columns, parse_row, unit_column=None) -> None:
        repaired = repair_rows(rows, first_line, date_columns)
        self.originals = repaired.originals
        self.run.rows_corrected = repaired.dates_corrected
        averages = (
            set()
            if unit_column is None
            else retail_average_rows(repaired.rows, date_columns, unit_column)
        )
        if unit_column is None:
            self._wholesale_unit_rows = retail_weeks_with_wholesale_units(
                repaired.rows, date_columns, 3
            )
        duplicates = duplicate_block_rows(repaired.rows, date_columns)
        repeated = repeated_week_rows(repaired.rows, date_columns)
        # Copy detection compares whole content, so it runs before any cell is voided.
        copied = (
            set()
            if unit_column is not None
            else nullify_copied_market_columns(
                repaired.rows, date_columns, RETAIL_MARKET_COLUMNS
            )
        )
        slid = (
            set()
            if unit_column is None
            else shifted_price_blocks(repaired.rows, date_columns)
        )
        slid_cells = (
            {}
            if unit_column is not None
            else shifted_market_cells(repaired.rows, date_columns, RETAIL_MARKET_COLUMNS)
        )
        if slid_cells:
            rows_by_number = dict(repaired.rows)
            for row_number, columns in slid_cells.items():
                row = rows_by_number[row_number]
                for column in columns:
                    if column < len(row) and row[column] is not None:
                        row[column] = None
        for row_number, row in repaired.rows:
            self.run.rows_read += 1
            if row_number in averages:
                self._reject(row_number, row, RejectionReason.UNEXPECTED_LAYOUT, RETAIL_AVERAGES)
                continue
            if row_number in duplicates:
                self._reject(
                    row_number, row, RejectionReason.UNEXPECTED_LAYOUT,
                    "duplicate block (identical to an earlier period)"
                )
                continue
            if row_number in repeated:
                self._reject(
                    row_number, row, RejectionReason.UNEXPECTED_LAYOUT,
                    "repeated price (identical to previous week)"
                )
                continue
            if row_number in slid:
                self._reject(
                    row_number, row, RejectionReason.UNEXPECTED_LAYOUT,
                    "price column shifted one row"
                )
                continue
            parse_row(row_number, row)

    def _wholesale_row(self, row_number: int, row: list[object]) -> None:
        # The type column is often damaged (#NAME?, a rubro, blank) on sound rows; only a
        # row that says it belongs to the other report is out of place.
        if len(row) < WHOLESALE_COLUMNS or letters(row[2]) == "minorista":
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
        if len(row) < RETAIL_COLUMNS or letters(row[2]) == "mayorista":
            self._reject(row_number, row, RejectionReason.UNEXPECTED_LAYOUT, "not a retail row")
            return
        cells = [
            (market, row[column])
            for market, column in zip(markets, RETAIL_MARKET_COLUMNS, strict=True)
        ]
        amounts = self._amounts(row_number, row, cells)
        name, rubro, _, unit_text = row[:4]
        week, month, year = row[11:14]
        # G5, Caso 1: when the unit column was pasted from the wholesale report, recover
        # the product's retail unit from the previous week it appeared in.
        raw_unit_text = unit_text
        if row_number in self._wholesale_unit_rows:
            recovered = self._product_units.get(name_key(str(name)))
            if recovered is not None:
                unit_text = recovered
            else:
                self._reject(
                    row_number,
                    row,
                    RejectionReason.UNKNOWN_UNIT,
                    "wholesale unit in a retail week, no previous week to recover from",
                )
                return
        self._observe(
            row_number, row, amounts, (name, rubro, unit_text, week, month, year), PriceType.RETAIL
        )
        # Track the last retail unit per product for future recovery.
        if row_number not in self._wholesale_unit_rows and str(raw_unit_text or "").strip():
            folded = fold(raw_unit_text)
            if not any(word in folded for word in ("saco", "quintal", "ciento", "caja", "bulto")) and "n/d" not in folded:
                self._product_units[name_key(str(name))] = str(raw_unit_text)

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
        # Audit M1: a stable price outranks a changed label. The ledger carries the
        # product's previous row so unreadable labels and nominal relabels keep the
        # unit the previous row priced in; real changes with the price moving keep
        # the row's own label.
        name = str(name)
        unit = unit_in_effect(
            self.resolver.unit(str(unit_text or "")),
            amounts,
            self._ledger.last(price_type, name_key(name)),
        )
        if unit is None:
            detail = f"unit {clean_name(unit_text or '')!r} has no known size"
            self._reject(row_number, row, RejectionReason.UNKNOWN_UNIT, detail)
            return
        for market, amount in amounts:
            self.observations.append(
                _Observation(
                    row_number=row_number,
                    row=row,
                    raw_name=name,
                    category=normalize_category(rubro),
                    unit=unit,
                    market_name=market,
                    market_kind=_market_kind(market, price_type),
                    price_type=price_type,
                    observed_on=observed_on,
                    amount=amount,
                )
            )
            self._ledger.record(
                price_type, name_key(name), market, amount, unit.base_unit, unit
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
        levels = self._level_context(product_ids, resolved)
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
            # Audit M2: a case the product's usual level decides against is a wrongly
            # labelled unit. Committed prices carry the reference (this run's values
            # fill in products selling here for the first time); the row never
            # includes itself, in-band spikes pass, and a lone price is no evidence.
            usual = self._usual_level(
                levels[0],
                levels[1],
                (product.id, market.id, observation.price_type, unit.base_unit),
                converted,
            )
            if usual is not None and (
                converted > usual * LEVEL_FACTOR or converted < usual / LEVEL_FACTOR
            ):
                detail = (
                    f"{unit.label} converts to {converted} per {unit.base_unit}, but "
                    f"{product.name} usually prices at {usual} at {market.name}"
                )
                self._reject_observation(observation, RejectionReason.UNIT_MISMATCH, detail)
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

    def _level_context(self, product_ids: set[int], resolved) -> tuple[dict, dict]:
        """Level evidence for audit M2, split so one odd row cannot move the reference.

        The committed average per (product, market, price type, base unit) stands for
        the product's usual level anywhere it has sold before; this run's converted
        values only fill in products with no committed level yet. Rejected rows never
        land in either, and the row under check is excluded from its own reference,
        so an outlier does not pull the level the legal rows face.
        """
        committed = {}
        rows = self.session.execute(
            select(
                Price.product_id,
                Price.market_id,
                Price.price_type,
                Unit.base_unit,
                func.avg(Price.price_per_base_unit),
            )
            .join(Unit, Price.unit_id == Unit.id)
            .where(Price.source == PriceSource.OFFICIAL, Price.product_id.in_(product_ids))
            .group_by(Price.product_id, Price.market_id, Price.price_type, Unit.base_unit)
        )
        for product_id, market_id, price_type, base_unit, average in rows:
            committed[(product_id, market_id, price_type, base_unit)] = Decimal(str(average))
        incoming: dict[tuple, list[Decimal]] = {}
        for observation, product, market in resolved:
            value = per_base_unit(observation.amount, observation.unit.factor)
            if value is None or value <= 0:
                continue
            key = (product.id, market.id, observation.price_type, observation.unit.base_unit)
            incoming.setdefault(key, []).append(value)
        return committed, incoming

    @staticmethod
    def _usual_level(committed, incoming, key, own: Decimal) -> Decimal | None:
        """The price level a row's conversion faces, or None without other evidence."""
        committed_level = committed.get(key)
        if committed_level is not None:
            return committed_level
        others = sorted(incoming.get(key, ()))
        try:
            others.remove(own)
        except ValueError:
            pass
        if not others:
            return None
        return others[len(others) // 2]

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
