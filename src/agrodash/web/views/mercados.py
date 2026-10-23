from datetime import timedelta

import pandas as pd
import streamlit as st

from agrodash.prices.compare import HARVEST_UNITS, harvest_values, market_ranking
from agrodash.web.charts import PALETTE, ranking_chart, spread_chart
from agrodash.web.state import require
from agrodash.web.ui import (
    UNIT_NAMES,
    choose_product,
    latest_prices,
    market_name,
    money,
    spread,
    theme,
)

require("mercados")
st.title("Mercados")
product, price_type = choose_product("mercados")
kind = "mayorista" if price_type == "wholesale" else "minorista"

latest = latest_prices(product.id, price_type)
if not latest:
    st.info("No hay precios de este tipo para el producto elegido.")
    st.stop()
unit = latest[0].base_unit
ranking = market_ranking(latest, unit)
frame = pd.DataFrame([vars(row) for row in ranking])
frame["market"] = frame["market"].map(market_name)

st.subheader("Dónde está más barato")
if len(ranking) < 2:
    st.info(f"Solo un mercado publica hoy el precio {kind} de este producto.")
else:
    cheapest, dearest = ranking[0], ranking[-1]
    st.markdown(
        f"**{market_name(cheapest.market)}** es el más barato y **{market_name(dearest.market)}** "
        f"el más caro: {dearest.value / cheapest.value - 1:.0%} de diferencia."
    )
    st.altair_chart(
        ranking_chart(frame, PALETTE[theme()][0], theme(), UNIT_NAMES[unit]), width="stretch"
    )
    st.caption("La línea punteada es el promedio de los mercados con precio reciente.")

st.subheader("Mayorista y minorista")
months = spread(product.id)
if months.empty:
    st.info("Este producto no tiene precios mayoristas y minoristas en los mismos meses.")
else:
    months = months[months["month"] >= months["month"].max() - timedelta(days=3 * 365)]
    last = months.iloc[-1]
    st.markdown(
        f"En {last['month']:%m/%Y} el precio minorista fue **{last['spread']:+.0%}** "
        "frente al mayorista."
    )
    st.altair_chart(spread_chart(months, theme(), UNIT_NAMES[unit]), width="stretch")
    st.caption("Promedio mensual de los mercados de cada tipo, últimos tres años.")

st.subheader("¿Cuánto vale mi cosecha?")
quantity_column, unit_column = st.columns(2)
quantity = quantity_column.number_input("Cantidad", min_value=0.0, value=1.0, step=1.0)
units = HARVEST_UNITS[unit]
harvest_unit = unit_column.selectbox("Unidad", list(units))
values = sorted(harvest_values(ranking, quantity, units[harvest_unit]), key=lambda v: -v[1])
st.dataframe(
    pd.DataFrame(
        {
            "Mercado": [market_name(market) for market, _ in values],
            "Valor": [money(value) for _, value in values],
        }
    ),
    hide_index=True,
    width="stretch",
)
st.caption(
    f"Al último precio {kind} de cada mercado. Lo que recibe quien produce suele ser menor: "
    "el intermediario y el transporte se llevan una parte."
)
