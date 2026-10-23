"""Altair charts for the public pages, following the dataviz method.

Colors are the reference categorical palette, validated in light and dark mode
(lines use the adjacent pairlist). A market keeps its color whatever else is
selected: color follows the entity, never the selection order.
"""

import altair as alt
import pandas as pd

PALETTE = {
    "light": [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ],
    "dark": [
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ],
}
TEXT_MUTED = {"light": "#52514e", "dark": "#c3c2b7"}
HEIGHT = 260
MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
# Diverging: cheap (cool) and dear (warm) poles from the palette, a gray midpoint at 100.
SEASON = {
    "light": {"Bajo": "#2a78d6", "Normal": "#c9c8c1", "Alto": "#eb6834"},
    "dark": {"Bajo": "#3987e5", "Normal": "#5a5953", "Alto": "#d95926"},
}


def market_colors(market_ids_by_name: dict[str, int], theme: str) -> alt.Scale:
    """A fixed color per market, chosen from its id."""
    palette = PALETTE[theme]
    names = sorted(market_ids_by_name, key=market_ids_by_name.get)
    return alt.Scale(
        domain=names, range=[palette[market_ids_by_name[name] % len(palette)] for name in names]
    )


def _axis(theme: str, title: str | None = None, **extra) -> alt.Axis:
    return alt.Axis(
        title=title,
        labelColor=TEXT_MUTED[theme],
        titleColor=TEXT_MUTED[theme],
        gridOpacity=0.25,
        domain=False,
        tickColor=TEXT_MUTED[theme],
        **extra,
    )


def history_chart(history: pd.DataFrame, colors: alt.Scale, theme: str, unit: str) -> alt.Chart:
    """Price per base unit over time, one 2px line per market, with a hover crosshair."""
    hover = alt.selection_point(fields=["observed_on"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(history).encode(
        x=alt.X("observed_on:T", axis=_axis(theme, format="%m/%Y")),
        y=alt.Y(
            "value:Q",
            axis=_axis(theme, f"RD$ por {unit}", format=",.2~f"),
            scale=alt.Scale(zero=False),
        ),
        color=alt.Color(
            "market:N",
            scale=colors,
            legend=alt.Legend(title=None, orient="top", labelColor=TEXT_MUTED[theme])
            if history["market"].nunique() > 1
            else None,
        ),
    )
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(size=80, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    )
    rule = (
        alt.Chart(history)
        .mark_rule(color=TEXT_MUTED[theme], strokeWidth=1)
        .encode(
            x="observed_on:T",
            opacity=alt.condition(hover, alt.value(0.4), alt.value(0)),
            tooltip=[
                alt.Tooltip("market:N", title="Mercado"),
                alt.Tooltip("observed_on:T", title="Semana", format="%d %b %Y"),
                alt.Tooltip("value:Q", title="RD$", format=",.2f"),
            ],
        )
        .add_params(hover)
    )
    return (lines + points + rule).properties(height=HEIGHT)


def forecast_chart(
    history: pd.DataFrame, forecast: pd.DataFrame, color: str, theme: str, unit: str
) -> alt.Chart:
    """The last year of prices, then the forecast with its 80% and 95% ranges as bands."""
    # The forecast line starts at the last observed price so the two connect.
    last = history.iloc[[-1]].assign(
        lo_80=history["value"].iloc[-1], hi_80=history["value"].iloc[-1]
    )
    last = last.assign(lo_95=last["lo_80"], hi_95=last["hi_80"])
    ahead = pd.concat([last, forecast], ignore_index=True)
    x = alt.X("observed_on:T", axis=_axis(theme, format="%m/%Y"))
    y_axis = _axis(theme, f"RD$ por {unit}", format=",.2~f")
    band95 = (
        alt.Chart(ahead)
        .mark_area(color=color, opacity=0.15)
        .encode(x=x, y=alt.Y("lo_95:Q", axis=y_axis, scale=alt.Scale(zero=False)), y2="hi_95:Q")
    )
    band80 = (
        alt.Chart(ahead).mark_area(color=color, opacity=0.3).encode(x=x, y="lo_80:Q", y2="hi_80:Q")
    )
    past = alt.Chart(history).mark_line(color=color, strokeWidth=2).encode(x=x, y="value:Q")
    future = (
        alt.Chart(ahead)
        .mark_line(color=color, strokeWidth=2, strokeDash=[4, 4])
        .encode(
            x=x,
            y="value:Q",
            tooltip=[
                alt.Tooltip("observed_on:T", title="Semana", format="%d %b %Y"),
                alt.Tooltip("value:Q", title="Pronóstico RD$", format=",.2f"),
                alt.Tooltip("lo_80:Q", title="80% desde", format=",.2f"),
                alt.Tooltip("hi_80:Q", title="80% hasta", format=",.2f"),
            ],
        )
    )
    dots = (
        alt.Chart(forecast).mark_point(color=color, size=80, filled=True).encode(x=x, y="value:Q")
    )
    return (band95 + band80 + past + future + dots).properties(height=HEIGHT)


def season_chart(profile: pd.DataFrame, theme: str) -> alt.Chart:
    """The usual index of each month as bars around 100, colored Alto, Normal or Bajo."""
    colors = SEASON[theme]
    x = alt.X("label:N", sort=MONTHS, axis=_axis(theme, labelAngle=0))
    bars = (
        alt.Chart(profile)
        .mark_bar(cornerRadiusEnd=4, size=14)
        .encode(
            x=x,
            y=alt.Y(
                "index:Q",
                axis=_axis(theme, "Índice (100 = promedio del año)"),
                # Centered on 100 so small swings stay visible; at least 80 to 120.
                scale=alt.Scale(
                    domain=[
                        min(80, profile["index"].min() - 5),
                        max(120, profile["index"].max() + 5),
                    ]
                ),
            ),
            y2=alt.datum(100),
            color=alt.Color(
                "level:N",
                scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                legend=alt.Legend(title=None, orient="top", labelColor=TEXT_MUTED[theme]),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Mes"),
                alt.Tooltip("index:Q", title="Índice", format=".0f"),
                alt.Tooltip("level:N", title="Precio"),
                alt.Tooltip("years:Q", title="Años"),
            ],
        )
    )
    baseline = (
        alt.Chart().mark_rule(color=TEXT_MUTED[theme], strokeWidth=1).encode(y=alt.datum(100))
    )
    return (bars + baseline).properties(height=HEIGHT)


def season_heatmap(index: pd.DataFrame, rows: str, theme: str, title: str) -> alt.Chart:
    """Index per month and `rows` (year or product) as cells, cheap to dear."""
    colors = SEASON[theme]
    count = index[rows].nunique()
    return (
        alt.Chart(index)
        .mark_rect(cornerRadius=2, stroke=None)
        .encode(
            x=alt.X("label:N", sort=MONTHS, axis=_axis(theme, labelAngle=0, orient="top")),
            y=alt.Y(
                f"{rows}:O",
                axis=_axis(theme, labelLimit=150),
                sort="ascending",
            ),
            color=alt.Color(
                "index:Q",
                scale=alt.Scale(
                    domain=[70, 100, 130],
                    range=[colors["Bajo"], colors["Normal"], colors["Alto"]],
                    # The default hcl interpolation tints the blue-to-gray half green.
                    interpolate="rgb",
                    clamp=True,
                ),
                legend=alt.Legend(
                    title="Índice",
                    orient="bottom",
                    labelColor=TEXT_MUTED[theme],
                    titleColor=TEXT_MUTED[theme],
                ),
            ),
            tooltip=[
                alt.Tooltip(f"{rows}:O", title=title),
                alt.Tooltip("label:N", title="Mes"),
                alt.Tooltip("index:Q", title="Índice", format=".0f"),
            ],
        )
        # Streamlit fits the legend and the top axis inside the height: leave them room.
        .properties(height=26 * int(count) + 90)
    )


def ranking_chart(ranking: pd.DataFrame, color: str, theme: str, unit: str) -> alt.Chart:
    """Latest price per market as horizontal bars, cheapest on top, with the average."""
    bars = (
        alt.Chart(ranking)
        .mark_bar(color=color, cornerRadiusEnd=4, size=14)
        .encode(
            x=alt.X("value:Q", axis=_axis(theme, f"RD$ por {unit}", format=",.2~f")),
            y=alt.Y("market:N", sort=None, axis=_axis(theme, labelLimit=150)),
            tooltip=[
                alt.Tooltip("market:N", title="Mercado"),
                alt.Tooltip("value:Q", title="RD$", format=",.2f"),
                alt.Tooltip("difference:Q", title="Contra el promedio", format="+.0%"),
            ],
        )
    )
    average = (
        alt.Chart(ranking)
        .mark_rule(color=TEXT_MUTED[theme], strokeDash=[4, 4])
        .encode(x="mean(value):Q")
    )
    # Streamlit fits the x axis inside the height: leave it room.
    return (bars + average).properties(height=40 * len(ranking) + 70)


def spread_chart(spread: pd.DataFrame, theme: str, unit: str) -> alt.Chart:
    """Monthly wholesale and retail prices of one product, two 2px lines."""
    names = {"wholesale": "Mayorista", "retail": "Minorista"}
    long = spread.melt("month", ["wholesale", "retail"], var_name="type", value_name="value")
    long["type"] = long["type"].map(names)
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30, filled=True))
        .encode(
            x=alt.X("month:T", axis=_axis(theme, format="%m/%Y")),
            y=alt.Y(
                "value:Q",
                axis=_axis(theme, f"RD$ por {unit}", format=",.2~f"),
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(domain=list(names.values()), range=PALETTE[theme][:2]),
                legend=alt.Legend(title=None, orient="top", labelColor=TEXT_MUTED[theme]),
            ),
            tooltip=[
                alt.Tooltip("type:N", title="Precio"),
                alt.Tooltip("month:T", title="Mes", format="%m/%Y"),
                alt.Tooltip("value:Q", title="RD$", format=",.2f"),
            ],
        )
        .properties(height=HEIGHT)
    )
