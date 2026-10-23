"""Run the forecasting job: forecast every ready series and store the run."""

from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version

import pandas as pd
from sqlalchemy import update
from sqlalchemy.orm import Session

from agrodash.forecasting.engine import (
    HORIZON,
    LONG_MIN_PERIODS,
    SELECTED_MODEL,
    benchmark,
    forecast_all,
    month_steps,
)
from agrodash.forecasting.models import Forecast, ForecastEvaluation, ForecastRun, ForecastStatus
from agrodash.forecasting.series import (
    SeriesKey,
    load_training_data,
    months_after,
    next_periods,
    period_date,
)
from agrodash.ingestion.quality import series_coverage

MODEL_VERSION = (
    f"naive-range/1 statsforecast {version('statsforecast')} "
    f"mlforecast {version('mlforecast')} scikit-learn {version('scikit-learn')}"
)
VALUE_PLACES = Decimal("0.0001")


class ForecastError(Exception):
    """The run failed; it is recorded as failed with this message."""


def run_forecasts(session: Session, run_benchmark: bool = True) -> ForecastRun:
    """Forecast every series ready to forecast and store the run.

    Series that are not ready (too short, too sparse, stale or retired) get no forecast:
    the app shows them as "sin pronóstico".
    """
    # A job killed mid-run never records its end; the weekly job runs one at a time, so
    # any run still marked running when a new one starts was interrupted.
    session.execute(
        update(ForecastRun)
        .where(ForecastRun.status == ForecastStatus.RUNNING)
        .values(
            status=ForecastStatus.FAILED,
            error="Interrupted before it finished",
            finished_at=datetime.now(UTC),
        )
    )
    run = ForecastRun(model_version=MODEL_VERSION, horizon=HORIZON)
    session.add(run)
    session.commit()
    try:
        keys = [
            SeriesKey.from_coverage(series) for series in series_coverage(session) if series.ready
        ]
        if keys:
            _forecast(session, run, keys, run_benchmark)
        run.series_count = len(keys)
        run.status = ForecastStatus.SUCCEEDED
        run.finished_at = datetime.now(UTC)
        session.commit()
    except Exception as error:
        session.rollback()
        # Only the first line: database errors append the SQL and its parameters.
        lines = str(error).splitlines()
        message = f"{error.__class__.__name__}: {lines[0] if lines else ''}"
        run.status = ForecastStatus.FAILED
        run.error = message[:500]
        run.finished_at = datetime.now(UTC)
        session.commit()
        raise ForecastError(message) from error
    return run


def _forecast(
    session: Session, run: ForecastRun, keys: list[SeriesKey], run_benchmark: bool
) -> None:
    data = load_training_data(session, keys)
    # Every forecast targets the periods after the latest published report; a series whose
    # last price is older forecasts across that gap.
    last = data.calendar[data.latest]
    targets = dict(enumerate(next_periods(last, HORIZON), start=1))
    gaps = (data.latest - data.frame.groupby("unique_id")["ds"].max()).to_dict()
    forecasts = [forecast_all(data.frame, gaps=gaps)]
    # Series with years of history also get one point per month for a year: week 4 of
    # month m ahead, as many report periods away as months usually hold. The four weekly
    # steps already cover the first month.
    steps = month_steps()[1:]
    for months, step in enumerate(steps, start=2):
        targets[step] = (*months_after(last, months), 4)
    sizes = data.frame.groupby("unique_id")["ds"].size()
    long = data.frame[data.frame["unique_id"].isin(sizes[sizes >= LONG_MIN_PERIODS].index)]
    if not long.empty:
        ahead = forecast_all(long, horizon=steps[-1], gaps=gaps, history=None)
        forecasts.append(ahead[ahead["step"].isin(steps)])
    for row in pd.concat(forecasts).itertuples():
        session.add(
            Forecast(
                run=run,
                **_series(data.keys[row.unique_id]),
                model=SELECTED_MODEL,
                step=row.step,
                target_on=period_date(targets[row.step]),
                value=_decimal(row.value),
                lo_80=_decimal(row.lo_80),
                hi_80=_decimal(row.hi_80),
                lo_95=_decimal(row.lo_95),
                hi_95=_decimal(row.hi_95),
            )
        )
    if not run_benchmark:
        return
    try:
        scores = benchmark(data.frame)
    except Exception as error:
        # The benchmark only monitors the alternatives; it must not cost the forecasts.
        lines = str(error).splitlines()
        run.error = f"Benchmark failed: {error.__class__.__name__}: {lines[0] if lines else ''}"[
            :500
        ]
        return
    for row in scores.itertuples():
        session.add(
            ForecastEvaluation(
                run=run,
                **_series(data.keys[row.unique_id]),
                model=row.model,
                mase=float(row.mase),
                selected=row.model == SELECTED_MODEL,
            )
        )


def _series(key: SeriesKey) -> dict:
    return {
        "product_id": key.product_id,
        "market_id": key.market_id,
        "price_type": key.price_type,
        "base_unit": key.base_unit,
    }


def _decimal(value) -> Decimal | None:
    if value is None or value != value:  # None or NaN: no range for this step
        return None
    return Decimal(str(value)).quantize(VALUE_PLACES)
