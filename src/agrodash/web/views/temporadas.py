import pandas as pd
import streamlit as st

from agrodash.prices.seasons import HIGH, LOW, MIN_YEARS, season_profile
from agrodash.web.charts import MONTHS, season_chart, season_heatmap
from agrodash.web.state import require
from agrodash.web.ui import category_name, choose_product, products, season_index, theme

MONTH_NAMES = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]

require("temporadas")
st.title("Temporadas")
product, price_type = choose_product("temporadas")

index = season_index((product.id,), price_type)
if index.empty:
    st.info(
        "Este producto todavía no tiene un año completo de precios: hace falta para ver su "
        "temporada."
    )
    st.stop()

profile = season_profile(index)
years = index["year"].nunique()


def months(level: str) -> str:
    chosen = [MONTH_NAMES[m.month - 1] for m in profile if m.level == level]
    return ", ".join(chosen) if chosen else "ninguno"


st.markdown(f"**Precio alto:** {months('Alto')}  \n**Precio bajo:** {months('Bajo')}")
st.caption(
    f"100 es el precio promedio de cada año. Alto: {HIGH - 100} % o más por encima; bajo: "
    f"{100 - LOW} % o más por debajo. Calculado mercado por mercado con {years} "
    f"año{'s' if years != 1 else ''} completo{'s' if years != 1 else ''} de precios."
)
if years < MIN_YEARS:
    st.warning(f"Con menos de {MIN_YEARS} años el patrón puede ser casualidad.")

st.subheader("Mes a mes")
frame = pd.DataFrame([vars(m) for m in profile])
frame["label"] = [MONTHS[m - 1] for m in frame["month"]]
st.altair_chart(season_chart(frame, theme()), width="stretch")

st.subheader("Año por año")
# Years as text: as numbers the chart drew them as a single bogus row.
by_year = index.assign(
    year=index["year"].astype(str), label=[MONTHS[m - 1] for m in index["month"]]
)
st.altair_chart(season_heatmap(by_year, "year", theme(), "Año"), width="stretch")

st.subheader(f"Todo el rubro: {category_name(product.category)}")
peers = {p.id: p.name for p in products(price_type) if p.category == product.category}
rubro = season_index(tuple(sorted(peers)), price_type)
rubro = rubro.groupby(["product_id", "month"], as_index=False)["index"].median()
rubro = rubro.assign(
    product=rubro["product_id"].map(peers), label=[MONTHS[m - 1] for m in rubro["month"]]
)
st.altair_chart(season_heatmap(rubro, "product", theme(), "Producto"), width="stretch")
st.caption("Cada fila es un producto con al menos un año completo de precios.")
