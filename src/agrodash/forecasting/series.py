"""Price series on the axis forecasting models need: report periods, not calendar days.

The reports publish weeks named Primera..Quinta, and a fifth week appears in some months
with no calendar rule behind it. Counting days between stored dates would make the gaps
look irregular (3 to 10 days around a month's end), so every series is indexed by the
ordered set of periods the reports actually published.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import false, func, select
from sqlalchemy.orm import Session

from agrodash.catalog.models import BaseUnit, Unit
from agrodash.ingestion.normalize import week_of_month, week_slot
from agrodash.ingestion.quality import SeriesCoverage
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType

# (year, month, week of month 1-5)
Period = tuple[int, int, int]
# About 54 periods are published per year (a few months have a fifth week).
SEASON_LENGTH = 54


@dataclass(frozen=True)
class SeriesKey:
    product_id: int
    market_id: int
    price_type: PriceType
    base_unit: BaseUnit

    @property
    def uid(self) -> str:
        return f"{self.product_id}-{self.market_id}-{self.price_type}-{self.base_unit}"

    @classmethod
    def from_coverage(cls, series: SeriesCoverage) -> "SeriesKey":
        return cls(
            series.product_id,
            series.market_id,
            PriceType(series.price_type),
            BaseUnit(series.base_unit),
        )


@dataclass(frozen=True)
class TrainingData:
    # unique_id, ds (report period index), y (price per base unit)
    frame: pd.DataFrame
    keys: dict[str, SeriesKey]
    calendar: list[Period]
    # Periods filled by linear interpolation, per series.
    imputed: dict[str, int]
    # Index in `calendar` of the latest period any official price was published for;
    # the calendar's padded weeks after it have not happened yet.
    latest: int | None


def period_of(day: date) -> Period:
    return (day.year, day.month, week_of_month(day))


def period_date(period: Period) -> date:
    return week_slot(*period)


def next_periods(last: Period, count: int) -> list[Period]:
    """The periods after `last`, assuming weeks 1-4 per month.

    Fifth weeks cannot be predicted, so forecasts never target one.
    """
    year, month, week = last
    periods = []
    for _ in range(count):
        if week >= 4:
            year, month, week = (year + 1, 1, 1) if month == 12 else (year, month + 1, 1)
        else:
            week += 1
        periods.append((year, month, week))
    return periods


def months_after(period: Period, months: int) -> tuple[int, int]:
    """(year, month) `months` months after the month of `period`."""
    index = period[0] * 12 + period[1] - 1 + months
    return index // 12, index % 12 + 1


def report_calendar(session: Session) -> list[Period]:
    """Every report period from the first official price to the last.

    Weeks 1-4 of every month are always periods, even when a month was not fully
    published (only the first week of August 2019 and July 2023 exists): those are
    missing values, not shorter months. A fifth week is a period only when published.
    """
    published = {
        period_of(day)
        for day in session.scalars(
            select(Price.observed_on).where(Price.source == PriceSource.OFFICIAL).distinct()
        )
    }
    if not published:
        return []
    (first_year, first_month, _), (last_year, last_month, _) = min(published), max(published)
    periods = set(published)
    year, month = first_year, first_month
    while (year, month) <= (last_year, last_month):
        periods.update((year, month, week) for week in range(1, 5))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return sorted(periods)


def load_training_data(session: Session, keys: Iterable[SeriesKey]) -> TrainingData:
    """Validated official prices of `keys`, one value per report period.

    Periods a series misses between its first and last price are filled by linear
    interpolation (and counted), because the models need an unbroken sequence.
    """
    keys = {key.uid: key for key in keys}
    calendar = report_calendar(session)
    position = {period: index for index, period in enumerate(calendar)}
    rows = session.execute(
        select(
            Price.product_id,
            Price.market_id,
            Price.price_type,
            Unit.base_unit,
            Price.observed_on,
            Price.price_per_base_unit,
        )
        .join(Unit, Price.unit_id == Unit.id)
        .where(
            Price.source == PriceSource.OFFICIAL,
            Price.status == PriceStatus.VALIDATED,
            Price.repeated == false(),
            Price.product_id.in_({key.product_id for key in keys.values()}),
        )
    )
    observed = pd.DataFrame(
        [
            (
                SeriesKey(product_id, market_id, price_type, base_unit).uid,
                position[period_of(day)],
                float(value),
            )
            for product_id, market_id, price_type, base_unit, day, value in rows
        ],
        columns=["unique_id", "ds", "y"],
    )
    observed = observed[observed["unique_id"].isin(keys)]
    frames, imputed = [], {}
    for uid, group in observed.groupby("unique_id", sort=True):
        values = group.groupby("ds")["y"].mean().sort_index()
        complete = values.reindex(range(values.index.min(), values.index.max() + 1))
        imputed[uid] = int(complete.isna().sum())
        frames.append(
            pd.DataFrame(
                {"unique_id": uid, "ds": complete.index, "y": complete.interpolate().values}
            )
        )
    frame = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["unique_id", "ds", "y"])
    )
    published = session.scalar(
        select(func.max(Price.observed_on)).where(Price.source == PriceSource.OFFICIAL)
    )
    latest = position[period_of(published)] if published else None
    return TrainingData(frame=frame, keys=keys, calendar=calendar, imputed=imputed, latest=latest)
