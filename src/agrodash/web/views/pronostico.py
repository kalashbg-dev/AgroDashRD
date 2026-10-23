from datetime import timedelta

import pandas as pd
import streamlit as st

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
    theme,
)

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

next_week = steps[0]
st.metric(f"Próxima semana ({next_week.target_on:%d/%m})", f"{money(next_week.value)} / {unit}")
if next_week.lo_80 is not None:
    st.caption(
        f"Con 80 % de probabilidad entre {money_md(next_week.lo_80)} y "
        f"{money_md(next_week.hi_80)}; con 95 %, entre {money_md(next_week.lo_95)} y "
        f"{money_md(next_week.hi_95)}."
    )

# Half a year keeps the four forecast weeks readable on a phone.
since = next_week.target_on - timedelta(weeks=26)
past = history(product.id, price_type, steps[0].base_unit, (steps[0].market_id,), since)
past_frame = pd.DataFrame([vars(p) for p in past])
future = pd.DataFrame(
    {
        "observed_on": [step.target_on for step in steps],
        "value": [float(step.value) for step in steps],
        **{
            column: [float(getattr(step, column) or step.value) for step in steps]
            for column in ("lo_80", "hi_80", "lo_95", "hi_95")
        },
    }
)
if not past_frame.empty:
    color = PALETTE[theme()][steps[0].market_id % len(PALETTE[theme()])]
    st.altair_chart(forecast_chart(past_frame, future, color, theme(), unit), width="stretch")

st.dataframe(
    pd.DataFrame(
        {
            "Semana": [step.target_on.strftime("%d/%m/%Y") for step in steps],
            "Pronóstico": [money(step.value) for step in steps],
            "Rango 80 %": [
                f"{money(step.lo_80)} – {money(step.hi_80)}" if step.lo_80 is not None else "—"
                for step in steps
            ],
        }
    ),
    hide_index=True,
    width="stretch",
)
st.caption(
    "El pronóstico es el último precio publicado: en las pruebas con la historia oficial, "
    "ningún modelo lo superó a un mes vista. Los rangos salen de cómo se movió el precio "
    "en el último año y, en esas pruebas, contuvieron cerca del 82 % (80 %) y del 93 % (95 %) "
    "de los precios reales."
)
