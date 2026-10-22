from datetime import date
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from agrodash import cli
from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, Unit
from agrodash.forecasting import service
from agrodash.forecasting.models import Forecast, ForecastEvaluation, ForecastRun, ForecastStatus
from agrodash.forecasting.service import ForecastError, run_forecasts
from agrodash.ingestion import quality
from agrodash.ingestion.normalize import week_slot
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType


def add_weekly_prices(session, name, months, price=10.0):
    product = Product(name=name, category="Frutas", base_unit=BaseUnit.POUND)
    market = session.query(Market).first() or Market(name="Mercado Nuevo", kind=MarketKind.RETAIL)
    unit = session.query(Unit).first() or Unit(
        code="1-lb", label="1 lb", base_unit=BaseUnit.POUND, factor=Decimal(1)
    )
    for year, month in months:
        for week in range(1, 5):
            price += 1
            session.add(
                Price(
                    product=product,
                    market=market,
                    unit=unit,
                    observed_on=week_slot(year, month, week),
                    price_type=PriceType.RETAIL,
                    source=PriceSource.OFFICIAL,
                    status=PriceStatus.VALIDATED,
                    amount=Decimal(str(price)),
                    price_per_base_unit=Decimal(str(price)),
                )
            )
    session.commit()
    return product


@pytest.fixture
def ready_series(session, monkeypatch):
    monkeypatch.setattr(quality, "MIN_WEEKS", 8)
    months = [(2026, month) for month in (7, 8, 9)]
    ready = add_weekly_prices(session, "Lista", months)
    add_weekly_prices(session, "Corta", [(2026, 9)])
    return ready


def fake_benchmark(frame, **_):
    return pd.DataFrame(
        [
            (uid, model, mase)
            for uid in frame["unique_id"].unique()
            for model, mase in (("Naive", 1.0), ("GBM", 1.4))
        ],
        columns=["unique_id", "model", "mase"],
    )


def test_ready_series_get_the_last_price_for_the_next_four_periods(
    session, ready_series, monkeypatch
):
    monkeypatch.setattr(service, "benchmark", fake_benchmark)

    run = run_forecasts(session)

    assert (run.status, run.series_count, run.horizon) == (ForecastStatus.SUCCEEDED, 1, 4)
    assert run.finished_at is not None
    forecasts = session.query(Forecast).order_by(Forecast.step).all()
    assert {forecast.product_id for forecast in forecasts} == {ready_series.id}
    # Last price was week 4 of September 2026 (22.0); next periods roll into October.
    assert [(f.step, f.target_on, f.value, f.model) for f in forecasts] == [
        (1, date(2026, 10, 1), Decimal("22.0000"), "Naive"),
        (2, date(2026, 10, 8), Decimal("22.0000"), "Naive"),
        (3, date(2026, 10, 15), Decimal("22.0000"), "Naive"),
        (4, date(2026, 10, 22), Decimal("22.0000"), "Naive"),
    ]
    assert all(f.lo_95 <= f.lo_80 <= f.value <= f.hi_80 <= f.hi_95 for f in forecasts)


def test_the_benchmark_is_stored_with_the_last_price_selected(session, ready_series, monkeypatch):
    monkeypatch.setattr(service, "benchmark", fake_benchmark)

    run_forecasts(session)

    evaluations = {(e.model, e.mase, e.selected) for e in session.query(ForecastEvaluation)}
    assert evaluations == {("Naive", 1.0, True), ("GBM", 1.4, False)}


def test_skipping_the_benchmark_stores_forecasts_only(session, ready_series, monkeypatch):
    monkeypatch.setattr(service, "benchmark", lambda *_, **__: pytest.fail("benchmark ran"))

    run_forecasts(session, run_benchmark=False)

    assert session.query(ForecastEvaluation).count() == 0
    assert session.query(Forecast).count() == 4


def test_a_failed_run_is_recorded_and_stores_no_forecasts(session, ready_series, monkeypatch):
    def broken(*_, **__):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(service, "forecast_all", broken)

    with pytest.raises(ForecastError, match="RuntimeError: out of memory"):
        run_forecasts(session)

    run = session.query(ForecastRun).one()
    assert (run.status, run.series_count) == (ForecastStatus.FAILED, 0)
    assert "out of memory" in run.error
    assert session.query(Forecast).count() == 0


def test_forecast_command_reports_what_it_did(cli_database, monkeypatch, capsys):
    monkeypatch.setattr(quality, "MIN_WEEKS", 8)
    monkeypatch.setattr(service, "benchmark", fake_benchmark)
    with Session(cli_database) as session:
        add_weekly_prices(session, "Lista", [(2026, month) for month in (7, 8, 9)])
        add_weekly_prices(session, "Corta", [(2026, 9)])

    assert cli.main(["forecast"]) == 0

    out = capsys.readouterr().out
    assert (
        "Forecast 1 series for 4 report periods (run 1); 1 series not ready: sin pronóstico" in out
    )
    assert (
        "Benchmark median MASE (below 1 beats the series' one-step naive): Naive 1.00, GBM 1.40"
        in out
    )


def test_series_that_stopped_earlier_are_forecast_after_the_latest_report(session, monkeypatch):
    monkeypatch.setattr(quality, "MIN_WEEKS", 8)
    monkeypatch.setattr(service, "benchmark", fake_benchmark)
    add_weekly_prices(session, "Al dia", [(2026, month) for month in (7, 8, 9)])
    # Ready too (stale by less than 8 weeks), but its last price is from August.
    lagging = add_weekly_prices(session, "Atrasada", [(2026, month) for month in (6, 7, 8)])

    run_forecasts(session)

    targets = [
        f.target_on
        for f in session.query(Forecast).filter_by(product_id=lagging.id).order_by(Forecast.step)
    ]
    assert targets == [date(2026, 10, 1), date(2026, 10, 8), date(2026, 10, 15), date(2026, 10, 22)]


def test_a_run_without_ready_series_succeeds_with_nothing_to_forecast(session):
    run = run_forecasts(session)

    assert (run.status, run.series_count) == (ForecastStatus.SUCCEEDED, 0)
    assert session.query(Forecast).count() == 0


def test_a_new_run_closes_runs_left_running_by_a_killed_job(session):
    stale = ForecastRun(model_version="old", horizon=4, status=ForecastStatus.RUNNING)
    session.add(stale)
    session.commit()

    run_forecasts(session)

    session.refresh(stale)
    assert (stale.status, stale.error) == (ForecastStatus.FAILED, "Interrupted before it finished")
    assert stale.finished_at is not None


def test_failure_messages_keep_only_the_first_line(session, ready_series, monkeypatch):
    def broken(*_, **__):
        raise RuntimeError(
            "database refused the insert\n[SQL: INSERT INTO forecasts ...]\n[parameters: ...]"
        )

    monkeypatch.setattr(service, "forecast_all", broken)

    with pytest.raises(ForecastError) as raised:
        run_forecasts(session)

    assert str(raised.value) == "RuntimeError: database refused the insert"
    assert session.query(ForecastRun).one().error == "RuntimeError: database refused the insert"


def test_a_failing_benchmark_keeps_the_forecasts_and_notes_the_error(
    session, ready_series, monkeypatch
):
    def broken(*_, **__):
        raise RuntimeError("numba crashed")

    monkeypatch.setattr(service, "benchmark", broken)

    run = run_forecasts(session)

    assert run.status is ForecastStatus.SUCCEEDED
    assert run.error == "Benchmark failed: RuntimeError: numba crashed"
    assert session.query(Forecast).count() == 4
    assert session.query(ForecastEvaluation).count() == 0
