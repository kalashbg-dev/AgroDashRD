import numpy as np
import pandas as pd
import pytest

from agrodash.forecasting.engine import (
    HORIZON,
    INTERVAL_HISTORY,
    benchmark,
    forecast_all,
    month_steps,
)


def frame(values_by_series: dict[str, list[float]]) -> pd.DataFrame:
    return pd.concat(
        pd.DataFrame({"unique_id": uid, "ds": range(len(values)), "y": values})
        for uid, values in values_by_series.items()
    )


def test_the_forecast_is_the_last_price_for_every_step():
    forecasts = forecast_all(frame({"a": [10.0, 11.0, 12.0], "b": [5.0, 5.0, 4.0]}))

    assert forecasts.groupby("unique_id")["value"].unique().to_dict() == {"a": [12.0], "b": [4.0]}
    assert forecasts.groupby("unique_id")["step"].apply(list).to_dict() == {
        "a": [1, 2, 3, 4],
        "b": [1, 2, 3, 4],
    }


def test_ranges_follow_how_the_series_moved_in_the_last_year():
    # A price that alternates 100 / 110: one step ahead it moves -9% or +10%.
    values = [100.0 if index % 2 == 0 else 110.0 for index in range(200)]

    first = forecast_all(frame({"a": values})).set_index("step").loc[1]

    assert first.value == 110.0
    assert first.lo_95 == pytest.approx(110 * 100 / 110)
    assert first.hi_95 == pytest.approx(110 * 1.1)
    assert first.lo_95 <= first.lo_80 <= first.value <= first.hi_80 <= first.hi_95


def test_only_the_last_year_shapes_the_range():
    calm = [100.0] * (INTERVAL_HISTORY + HORIZON)
    wild_long_ago = [50.0, 150.0] * 100

    first = forecast_all(frame({"a": wild_long_ago + calm})).set_index("step").loc[1]

    assert (first.lo_95, first.hi_95) == (100.0, 100.0)


def test_a_series_too_short_for_a_range_still_gets_its_last_price():
    forecasts = forecast_all(frame({"a": [7.0]}))

    assert forecasts["value"].tolist() == [7.0] * HORIZON
    assert forecasts[["lo_80", "hi_80", "lo_95", "hi_95"]].isna().all().all()


def test_benchmark_scores_every_model_against_the_last_price():
    rng = np.random.default_rng(1)
    walks = {f"s{i}": list(100 + np.cumsum(rng.normal(0, 1, 160))) for i in range(3)}

    scores = benchmark(frame(walks), models=("AutoETS", "GBM"), windows=2)

    assert set(scores["model"]) == {"Naive", "AutoETS", "GBM"}
    assert set(scores["unique_id"]) == set(walks)
    assert scores["mase"].gt(0).all()


def test_flat_series_have_no_meaningful_score_and_are_left_out():
    flat = {"flat": [10.0] * 120, "moving": list(np.linspace(1, 50, 120))}

    scores = benchmark(frame(flat), models=("GBM",), windows=2)

    assert set(scores["unique_id"]) == {"moving"}


def test_the_range_always_contains_the_forecast():
    rising = [100.0 + index for index in range(80)]

    forecasts = forecast_all(frame({"a": rising}))

    assert (forecasts["lo_95"] <= forecasts["value"]).all()
    assert (forecasts["hi_95"] > forecasts["value"]).all()


def test_a_gap_before_the_target_widens_the_range_like_a_longer_horizon():
    values = [100.0 if index % 3 else 130.0 for index in range(120)]
    series = frame({"a": values})

    near = forecast_all(series).set_index("step")
    later = forecast_all(series, gaps={"a": 3}).set_index("step")

    # Step 1 after a 3-period gap is 4 periods past the last price.
    assert (later.loc[1, "lo_95"], later.loc[1, "hi_95"]) == (
        near.loc[4, "lo_95"],
        near.loc[4, "hi_95"],
    )


def test_unknown_benchmark_models_are_refused():
    with pytest.raises(ValueError, match="Unknown benchmark model 'Prophet'"):
        benchmark(frame({"a": list(np.linspace(1, 50, 120))}), models=("Prophet",))


def test_long_horizons_take_their_range_from_the_whole_history():
    calm = [100.0] * (INTERVAL_HISTORY + 60)
    wild_long_ago = [50.0, 150.0] * 100

    last = (
        forecast_all(frame({"a": wild_long_ago + calm}), horizon=54, history=None)
        .set_index("step")
        .loc[54]
    )

    assert last.value == 100.0
    assert last.lo_95 < 100.0 < last.hi_95


def test_month_steps_follow_the_report_calendar():
    # About 4.5 report periods per month (some months publish a fifth week).
    assert month_steps(3) == [5, 9, 14]
    assert month_steps()[-1] == 54
