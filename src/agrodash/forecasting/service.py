"""Run the forecasting job: forecast every ready series and store the run."""

from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version

from sqlalchemy.orm import Session

from agrodash.forecasting.engine import HORIZON, SELECTED_MODEL, benchmark, forecast_all
from agrodash.forecasting.models import Forecast, ForecastEvaluation, ForecastRun, ForecastStatus
from agrodash.forecasting.series import SeriesKey, load_training_data, next_periods, period_date
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
        message = f"{error.__class__.__name__}: {error}"
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
    targets = next_periods(data.calendar[data.latest], HORIZON)
    gaps = (data.latest - data.frame.groupby("unique_id")["ds"].max()).to_dict()
    for row in forecast_all(data.frame, gaps=gaps).itertuples():
        session.add(
            Forecast(
                run=run,
                **_series(data.keys[row.unique_id]),
                model=SELECTED_MODEL,
                step=row.step,
                target_on=period_date(targets[row.step - 1]),
                value=_decimal(row.value),
                lo_80=_decimal(row.lo_80),
                hi_80=_decimal(row.hi_80),
                lo_95=_decimal(row.lo_95),
                hi_95=_decimal(row.hi_95),
            )
        )
    if not run_benchmark:
        return
    for row in benchmark(data.frame).itertuples():
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
