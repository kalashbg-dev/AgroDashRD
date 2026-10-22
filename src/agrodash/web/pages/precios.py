from datetime import timedelta

import pandas as pd
import streamlit as st

from agrodash.web.charts import history_chart, market_colors
from agrodash.web.state import require
from agrodash.web.ui import UNIT_NAMES, choose_product, history, latest_prices, market_name, theme

require("precios")
st.title("Precios")
product, price_type = choose_product("precios")

latest = latest_prices(product.id, price_type)
if not latest:
    st.info("No hay precios de este tipo para el producto elegido.")
    st.stop()

st.subheader("Último precio por mercado")
units = {row.base_unit for row in latest}
# One unit (the usual case) goes in the header so the table fits a phone screen.
price_column = f"RD$ / {UNIT_NAMES[next(iter(units))]}" if len(units) == 1 else "RD$"
st.dataframe(
    pd.DataFrame(
        {
            "Mercado": [market_name(row.market) for row in latest],
            price_column: [
                f"{float(row.value):,.2f}"
                + ("" if len(units) == 1 else f" / {UNIT_NAMES[row.base_unit]}")
                for row in latest
            ],
            "Semana": [row.observed_on.strftime("%d/%m/%y") for row in latest],
            "4 sem.": [f"{row.change:+.0%}" if row.change is not None else "—" for row in latest],
        }
    ),
    hide_index=True,
    use_container_width=True,
)

st.subheader("Historia")
markets = {row.market: row for row in latest}
chosen = st.multiselect(
    "Mercados (hasta 3)", list(markets), default=list(markets)[:1], max_selections=3
)
if not chosen:
    st.caption("Elige al menos un mercado.")
    st.stop()
unit = markets[chosen[0]].base_unit
same_unit = [name for name in chosen if markets[name].base_unit == unit]
if len(same_unit) < len(chosen):
    st.caption(f"Solo se comparan mercados con precios por {UNIT_NAMES[unit]}.")
since = max(row.observed_on for row in latest) - timedelta(days=730)
points = history(
    product.id, price_type, unit, tuple(markets[name].market_id for name in same_unit), since
)
frame = pd.DataFrame([vars(point) for point in points])
colors = market_colors({row.market: row.market_id for row in latest}, theme())
st.altair_chart(history_chart(frame, colors, theme(), UNIT_NAMES[unit]), use_container_width=True)
st.caption("Precios oficiales validados de los dos últimos años, por semana de reporte.")
