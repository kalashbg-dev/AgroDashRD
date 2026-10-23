from datetime import date

import altair as alt
import pandas as pd

from agrodash.web.charts import history_chart, season_heatmap, spread_chart


def spec(chart) -> dict:
    return chart.to_dict()


def test_time_axes_show_months_as_numbers_not_english_names():
    history = pd.DataFrame(
        {"market": ["Nuevo"], "observed_on": [date(2026, 9, 1)], "value": [10.0]}
    )

    chart = spec(history_chart(history, alt.Scale(), "light", "libra"))

    assert chart["layer"][0]["encoding"]["x"]["axis"]["format"] == "%m/%Y"


def test_the_spread_chart_draws_one_line_per_price_type():
    spread = pd.DataFrame(
        {
            "month": [date(2026, 8, 1), date(2026, 9, 1)],
            "wholesale": [20.0, 22.0],
            "retail": [30.0, 31.0],
            "spread": [0.5, 0.41],
        }
    )

    chart = spec(spread_chart(spread, "dark", "libra"))

    assert chart["encoding"]["color"]["scale"]["domain"] == ["Mayorista", "Minorista"]
    values = next(iter(chart["datasets"].values()))
    assert sorted({row["type"] for row in values}) == ["Mayorista", "Minorista"]


def test_heatmaps_leave_room_for_the_legend_and_axis_inside_their_height():
    index = pd.DataFrame(
        {"year": ["2024", "2025"], "label": ["Ene", "Ene"], "index": [95.0, 105.0]}
    )

    chart = spec(season_heatmap(index, "year", "light", "Año"))

    assert chart["height"] == 26 * 2 + 90
