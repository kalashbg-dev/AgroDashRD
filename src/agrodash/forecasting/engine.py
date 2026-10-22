"""Price forecasts: the last price, with a range from how the series moved last year.

The backtest on the official history (L22-L23 in the task log) showed that one month
ahead the last price is the best forecast. Picking AutoETS, Theta or a gradient-boosting
model per series looked better in-sample but was worse on periods the selection had not
seen (MASE 1.086-1.125 against 1.063). So the forecast is the last price, and every run
benchmarks those models against it to notice if that ever changes.
"""

import warnings

import numpy as np
import pandas as pd

from agrodash.forecasting.series import SEASON_LENGTH

HORIZON = 4
# One year of report periods. Over six backtest windows the 80% range held 75.9-87.7% of
# later prices (about 82.5% on average) and the 95% range 90.6-96.2% (about 93.4%).
# Shorter histories cover less; longer ones are wider without covering much more.
INTERVAL_HISTORY = SEASON_LENGTH
LEVELS = {80: (0.10, 0.90), 95: (0.025, 0.975)}
SELECTED_MODEL = "Naive"
BENCHMARK_MODELS = ("AutoETS", "Theta", "GBM")
BENCHMARK_WINDOWS = 4
GBM_LAGS = [1, 2, 3, 4, 8, 13, 26]


def forecast_all(
    frame: pd.DataFrame, horizon: int = HORIZON, gaps: dict[str, int] | None = None
) -> pd.DataFrame:
    """One row per series and step: the last price and its 80% and 95% range.

    The range for step k is the last price times the quantiles of how prices changed
    over k periods (price k periods later / price) during the last year, widened if
    needed so it contains the last price. `gaps` holds, per series, how many periods
    separate its last price from the latest report: step 1 is then gap + 1 periods away.
    """
    rows = []
    for uid, group in frame.sort_values("ds").groupby("unique_id", sort=True):
        values = group["y"].reset_index(drop=True)
        last = float(values.iloc[-1])
        gap = (gaps or {}).get(uid, 0)
        recent = values.iloc[-(INTERVAL_HISTORY + gap + horizon) :]
        for step in range(1, horizon + 1):
            ratios = (recent.shift(-(gap + step)) / recent).dropna()
            row = {"unique_id": uid, "step": step, "value": last}
            for level, (low, high) in LEVELS.items():
                if ratios.empty:
                    row[f"lo_{level}"] = row[f"hi_{level}"] = None
                else:
                    # The range always contains the forecast itself: a series that only
                    # rose last year would otherwise show a range above its own forecast.
                    row[f"lo_{level}"] = min(last, last * float(ratios.quantile(low)))
                    row[f"hi_{level}"] = max(last, last * float(ratios.quantile(high)))
            rows.append(row)
    return pd.DataFrame(rows)


def benchmark(
    frame: pd.DataFrame,
    models: tuple[str, ...] = BENCHMARK_MODELS,
    windows: int = BENCHMARK_WINDOWS,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Backtest MASE of the last price and `models` on the latest `windows` x `horizon`
    periods of every series.

    MASE is the mean absolute error divided by the series' own one-step naive error, so
    series of any price level compare. Flat series (no change at all) have no scale and
    are left out.
    """
    from mlforecast import MLForecast
    from mlforecast.target_transforms import Differences
    from sklearn.ensemble import HistGradientBoostingRegressor
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, AutoTheta, Naive

    unknown = set(models) - {"AutoETS", "Theta", "GBM"}
    if unknown:
        raise ValueError(f"Unknown benchmark model {sorted(unknown)[0]!r}")
    scale = frame.groupby("unique_id")["y"].apply(lambda y: y.diff().abs().mean())
    frame = frame[frame["unique_id"].isin(scale[scale > 0].index)]
    if frame.empty:
        return pd.DataFrame(columns=["unique_id", "model", "mase"])
    statistical = {
        "AutoETS": AutoETS(season_length=1),
        "Theta": AutoTheta(season_length=1, alias="Theta"),
    }
    chosen = [Naive()] + [statistical[name] for name in models if name in statistical]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # n_jobs=1: on macOS worker processes recompile numba and run far slower.
        results = StatsForecast(models=chosen, freq=1, n_jobs=1).cross_validation(
            df=frame, h=horizon, n_windows=windows, step_size=horizon
        )
        if "GBM" in models:
            gbm = MLForecast(
                models={"GBM": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)},
                freq=1,
                lags=GBM_LAGS,
                target_transforms=[Differences([1])],
            ).cross_validation(df=frame, h=horizon, n_windows=windows, step_size=horizon)
            # Left join: a series the GBM step drops would otherwise lose its other scores.
            results = results.merge(
                gbm[["unique_id", "ds", "cutoff", "GBM"]],
                on=["unique_id", "ds", "cutoff"],
                how="left",
            )
    names = [SELECTED_MODEL, *models]
    errors = {
        name: (results[name] - results["y"]).abs().groupby(results["unique_id"]).mean() / scale
        for name in names
    }
    scores = pd.DataFrame(errors).reset_index(names="unique_id")
    scores = scores.melt(id_vars="unique_id", var_name="model", value_name="mase")
    return scores[np.isfinite(scores["mase"])].reset_index(drop=True)
