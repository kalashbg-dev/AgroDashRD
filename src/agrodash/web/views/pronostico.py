from datetime import timedelta

import pandas as pd
import streamlit as st

from agrodash.forecasting.engine import HORIZON, month_steps
from agrodash.prices.seasons import season_profile
from agrodash.web.charts import PALETTE, forecast_chart
from agrodash.web.state import require
from agrodash.web.ui import (
    NOT_READY,
    UNIT_NAMES,
    choose_product,
    coverage,
    forecasts,
    history,
    money,
    money_md,
    season_index,
    theme,
)

HORIZONS = {"4 semanas": HORIZON, "3 meses": month_steps(3)[-1], "12 meses": month_steps()[-1]}
CAPTIONS = {
    "4 semanas": (
        "El pronóstico es el último precio publicado: en las pruebas con la historia oficial, "
        "ningún modelo lo superó a un mes vista. Los rangos salen de cómo se movió el precio "
        "en el último año y, en esas pruebas, contuvieron cerca del 82 % (80 %) y del 93 % "
        "(95 %) de los precios reales."
    ),
    "3 meses": (
        "A tres meses tampoco hubo un método mejor que el último precio, ni ajustándolo por "
        "temporada. El rango sale de toda la historia del precio y, en las pruebas, contuvo "
        "el 84 % (80 %) y el 96 % (95 %) de los precios reales. La columna Temporada dice si "
        "ese mes suele tener precio alto o bajo."
    ),
    "12 meses": (
        "A un año el último precio es solo un punto de partida: el rango es lo importante. "
        "Sale de toda la historia del precio y, en las pruebas, contuvo el 82 % (80 %) y el "
        "92 % (95 %) de los precios reales. La columna Temporada dice si ese mes suele tener "
        "precio alto o bajo."
    ),
}

require("pronostico")
st.title("Pronóstico")
product, price_type = choose_product("pronostico")

points = forecasts(product.id, price_type)
if not points:
    st.subheader("Sin pronóstico")
    reasons = sorted(
        {
            NOT_READY[reason]
            for series in coverage()
            if series.product_id == product.id and series.price_type == price_type
            for reason in series.not_ready
        }
    )
    if reasons:
        st.write("Este producto no tiene pronóstico porque " + "; ".join(reasons) + ".")
    else:
        st.write("Todavía no se calculó ningún pronóstico.")
    st.stop()

by_market = {}
for point in points:
    by_market.setdefault(point.market, []).append(point)
market = st.selectbox("Mercado", list(by_market))
steps = by_market[market]
unit = UNIT_NAMES[steps[0].base_unit]

horizon = (
    st.segmented_control("Plazo", list(HORIZONS), default="4 semanas", key="pronostico_horizon")
    or "4 semanas"
)
shown = [step for step in steps if step.step <= HORIZONS[horizon]]
if horizon != "4 semanas" and all(step.step <= HORIZON for step in shown):
    st.info(
        f"No hay proyección a {horizon} para este mercado: hacen falta tres años de precios. "
        "Se muestran las próximas 4 semanas."
    )
    horizon = "4 semanas"
target = shown[-1] if horizon != "4 semanas" else shown[0]
label = "Próxima semana" if horizon == "4 semanas" else f"En {horizon}"
when = f"{target.target_on:%d/%m}" if horizon == "4 semanas" else f"{target.target_on:%m/%Y}"
st.metric(f"{label} ({when})", f"{money(target.value)} / {unit}")
if target.lo_80 is not None:
    st.caption(
        f"Con 80 % de probabilidad entre {money_md(target.lo_80)} y "
        f"{money_md(target.hi_80)}; con 95 %, entre {money_md(target.lo_95)} y "
        f"{money_md(target.hi_95)}."
    )

# Half a year keeps four weeks readable on a phone; longer plazos show two years.
weeks = 26 if horizon == "4 semanas" else 104
since = shown[0].target_on - timedelta(weeks=weeks)
past = history(product.id, price_type, steps[0].base_unit, (steps[0].market_id,), since)
past_frame = pd.DataFrame([vars(p) for p in past])
future = pd.DataFrame(
    {
        "observed_on": [step.target_on for step in shown],
        "value": [float(step.value) for step in shown],
        **{
            column: [float(getattr(step, column) or step.value) for step in shown]
            for column in ("lo_80", "hi_80", "lo_95", "hi_95")
        },
    }
)
if not past_frame.empty:
    color = PALETTE[theme()][steps[0].market_id % len(PALETTE[theme()])]
    st.altair_chart(forecast_chart(past_frame, future, color, theme(), unit), width="stretch")

seasons = season_index((product.id,), price_type)
levels = {m.month: m.level for m in season_profile(seasons)} if not seasons.empty else {}
if horizon == "4 semanas":
    table = {
        "Semana": [step.target_on.strftime("%d/%m/%Y") for step in shown],
        "Pronóstico": [money(step.value) for step in shown],
        "Rango 80 %": [
            f"{money(step.lo_80)} – {money(step.hi_80)}" if step.lo_80 is not None else "—"
            for step in shown
        ],
    }
else:
    # The value is the same every month (shown above); the range is what changes. Plain
    # numbers keep the three columns within a phone screen.
    table = {
        "Fecha": [
            step.target_on.strftime("%d/%m" if step.step <= HORIZON else "%m/%Y") for step in shown
        ],
        "Rango 80 % (RD$)": [
            f"{float(step.lo_80):,.2f} – {float(step.hi_80):,.2f}"
            if step.lo_80 is not None
            else "—"
            for step in shown
        ],
    }
    if levels:
        table["Temporada"] = [levels.get(step.target_on.month, "—") for step in shown]
st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
st.caption(CAPTIONS[horizon])
