"""Data-quality report: what was rejected, which names still need review, and which
price series are long, dense and current enough to forecast."""

from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import extract, func, select
from sqlalchemy.orm import Session

from agrodash.catalog.models import Market, Product, ProductAlias, Unit
from agrodash.ingestion.models import IngestionRun, IngestionStatus, QuarantinedRow
from agrodash.ingestion.normalize import name_key
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType

# A series is ready to forecast with two years of weekly prices, three of every four
# weeks covered, and a price in the last eight weeks of the data.
MIN_WEEKS = 104
MIN_COVERAGE = 0.75
MAX_STALENESS = timedelta(weeks=8)
# A product whose prices stop while a similar name in its category starts within three
# weeks was most likely renamed in the reports ("Cilantro/ grande/ primera" became
# "Cilantro, primera" in February 2018). Only a person can confirm it.
RENAME_MAX_GAP = timedelta(weeks=3)
RENAME_MIN_SIMILARITY = 0.5
# Week 5 only exists in some months, so coverage is measured on weeks 1-4 (days 1-22).
LAST_COUNTED_DAY = 22


@dataclass(frozen=True)
class SeriesCoverage:
    product_id: int
    market_id: int
    product: str
    market: str
    price_type: str
    base_unit: str
    first_on: date
    last_on: date
    weeks: int
    coverage: float
    ready: bool
    # Why it is not ready: "inactive", "short", "sparse", "stale"; empty when ready.
    not_ready: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityReport:
    official_prices: dict[PriceType, int]
    pending_reports: int
    rejections: dict[str, int]
    products: int
    aliases: int
    unreviewed_aliases: list[tuple[str, str]]
    units: int
    latest_observation: date | None
    series: list[SeriesCoverage]
    rename_candidates: list[tuple[str, str, int]]

    @property
    def ready_series(self) -> list[SeriesCoverage]:
        return [series for series in self.series if series.ready]


def build_report(session: Session) -> QualityReport:
    official = Price.source == PriceSource.OFFICIAL
    latest = session.scalar(select(func.max(Price.observed_on)).where(official))
    return QualityReport(
        official_prices={
            price_type: count
            for price_type, count in session.execute(
                select(Price.price_type, func.count()).where(official).group_by(Price.price_type)
            )
        },
        pending_reports=session.scalar(
            select(func.count()).where(
                Price.source == PriceSource.FIELD_REPORT, Price.status == PriceStatus.PENDING
            )
        ),
        rejections=_rejections(session),
        products=session.scalar(select(func.count(Product.id))),
        aliases=session.scalar(select(func.count(ProductAlias.id))),
        unreviewed_aliases=[
            (raw_name, name)
            for raw_name, name in session.execute(
                select(ProductAlias.raw_name, Product.name)
                .join(Product)
                .where(ProductAlias.reviewed.is_(False))
                .order_by(Product.name, ProductAlias.raw_name)
            )
        ],
        units=session.scalar(select(func.count(Unit.id))),
        latest_observation=latest,
        series=series_coverage(session),
        rename_candidates=rename_candidates(session),
    )


def _rejections(session: Session) -> dict[str, int]:
    """Rejections of the loads that added prices. Another export of an already loaded
    report adds none and would quarantine the same rows again, so it is left out;
    weekly files add new prices, so every one of them counts."""
    counted_runs = select(IngestionRun.id).where(
        IngestionRun.status == IngestionStatus.SUCCEEDED, IngestionRun.prices_loaded > 0
    )
    rows = session.execute(
        select(QuarantinedRow.reason, func.count())
        .where(QuarantinedRow.run_id.in_(counted_runs))
        .group_by(QuarantinedRow.reason)
        .order_by(QuarantinedRow.reason)
    )
    return {str(reason): count for reason, count in rows}


def rename_candidates(session: Session) -> list[tuple[str, str, int]]:
    """(old name, new name, days between) for products that look renamed."""
    spans = session.execute(
        select(
            Product.name,
            Product.category,
            func.min(Price.observed_on),
            func.max(Price.observed_on),
        )
        .join(Price, Price.product_id == Product.id)
        .where(Price.source == PriceSource.OFFICIAL)
        .group_by(Product.id, Product.name, Product.category)
    ).all()
    candidates = []
    for old, category, _, old_last in spans:
        for new, new_category, new_first, _ in spans:
            gap = new_first - old_last
            if (
                new_category == category
                and timedelta(0) < gap <= RENAME_MAX_GAP
                and _similarity(old, new) >= RENAME_MIN_SIMILARITY
            ):
                candidates.append((old, new, gap.days))
    return sorted(candidates)


def _similarity(left: str, right: str) -> float:
    left_words, right_words = set(name_key(left).split()), set(name_key(right).split())
    if not left_words | right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def series_coverage(session: Session) -> list[SeriesCoverage]:
    """Every official series (product, market, price type, base unit) and its coverage."""
    # Wholesale and retail reports may end on different dates; compare within each type.
    latest_by_type = dict(
        session.execute(
            select(Price.price_type, func.max(Price.observed_on))
            .where(Price.source == PriceSource.OFFICIAL)
            .group_by(Price.price_type)
        ).all()
    )
    counted = extract("day", Price.observed_on) <= LAST_COUNTED_DAY
    rows = session.execute(
        select(
            Product.id,
            Market.id,
            Product.name,
            Market.name,
            Price.price_type,
            Unit.base_unit,
            Product.is_active,
            Market.is_active,
            func.min(Price.observed_on),
            func.max(Price.observed_on),
            func.count(func.distinct(Price.observed_on)),
            func.count(func.distinct(Price.observed_on)).filter(counted),
        )
        .join(Product, Price.product_id == Product.id)
        .join(Market, Price.market_id == Market.id)
        .join(Unit, Price.unit_id == Unit.id)
        .where(Price.source == PriceSource.OFFICIAL)
        # A product sold by the pound and later by the piece has one series per unit.
        .group_by(
            Product.id,
            Market.id,
            Product.name,
            Market.name,
            Price.price_type,
            Unit.base_unit,
            Product.is_active,
            Market.is_active,
        )
        .order_by(Product.name, Market.name, Price.price_type, Unit.base_unit)
    )
    series = []
    for (
        product_id,
        market_id,
        product,
        market,
        price_type,
        base_unit,
        product_active,
        market_active,
        first_on,
        last_on,
        weeks,
        counted_weeks,
    ) in rows:
        months = (last_on.year - first_on.year) * 12 + last_on.month - first_on.month + 1
        coverage = counted_weeks / (4 * months)
        reasons = tuple(
            reason
            for reason, applies in (
                ("inactive", not (product_active and market_active)),
                ("short", weeks < MIN_WEEKS),
                ("sparse", coverage < MIN_COVERAGE),
                ("stale", latest_by_type[price_type] - last_on > MAX_STALENESS),
            )
            if applies
        )
        series.append(
            SeriesCoverage(
                product_id=product_id,
                market_id=market_id,
                product=product,
                market=market,
                price_type=str(price_type),
                base_unit=str(base_unit),
                first_on=first_on,
                last_on=last_on,
                weeks=weeks,
                coverage=coverage,
                ready=not reasons,
                not_ready=reasons,
            )
        )
    return series


@dataclass(frozen=True)
class QuarantineRow:
    file_name: str
    row_number: int
    reason: str
    detail: str | None
    raw_values: str


def quarantine(
    session: Session, reason: str | None = None, limit: int = 200
) -> list[QuarantineRow]:
    """Rejected rows of the loads the quality report counts, newest load first."""
    query = (
        select(
            IngestionRun.file_name,
            QuarantinedRow.row_number,
            QuarantinedRow.reason,
            QuarantinedRow.detail,
            QuarantinedRow.raw_values,
        )
        .join(IngestionRun)
        .where(IngestionRun.status == IngestionStatus.SUCCEEDED, IngestionRun.prices_loaded > 0)
        .order_by(IngestionRun.id.desc(), QuarantinedRow.row_number)
        .limit(limit)
    )
    if reason:
        query = query.where(QuarantinedRow.reason == reason)
    return [
        QuarantineRow(file_name, row_number, str(row_reason), detail, raw_values)
        for file_name, row_number, row_reason, detail, raw_values in session.execute(query)
    ]


def recent_runs(session: Session, limit: int = 10) -> list[IngestionRun]:
    return list(session.scalars(select(IngestionRun).order_by(IngestionRun.id.desc()).limit(limit)))
