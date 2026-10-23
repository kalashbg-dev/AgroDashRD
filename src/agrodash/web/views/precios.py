from datetime import date, timedelta

import pandas as pd
import streamlit as st

from agrodash.web.charts import history_chart, market_colors
from agrodash.web.state import require
from agrodash.web.ui import (
    UNIT_NAMES,
    category_name,
    choose_product,
    csv,
    history,
    latest_prices,
    market_name,
    price_changes,
    products,
    products_history,
    theme,
)

PERIODS = {"6 meses": 26, "1 año": 52, "2 años": 104, "5 años": 260, "Todo": None}
TOP = 10


def period_start(latest_on, key: str):
    """The first day of the chosen period, ending at `latest_on`."""
    label = st.segmented_control("Periodo", list(PERIODS), default="2 años", key=key) or "2 años"
    weeks = PERIODS[label]
    return latest_on - timedelta(weeks=weeks) if weeks else date.min


require("precios")
st.title("Precios")
product, price_type = choose_product("precios")
product_tab, compare_tab, movers_tab = st.tabs(["Producto", "Comparar", "Mayores cambios"])

latest = latest_prices(product.id, price_type)
with product_tab:
    if not latest:
        st.info("No hay precios de este tipo para el producto elegido.")
    else:
        st.subheader("Último precio por mercado")
        units = {row.base_unit for row in latest}
        # One unit (the usual case) goes in the header so the table fits a phone screen.
        price_column = f"RD$ / {UNIT_NAMES[next(iter(units))]}" if len(units) == 1 else "RD$"
        table = pd.DataFrame(
            {
                "Mercado": [market_name(row.market) for row in latest],
                price_column: [
                    f"{float(row.value):,.2f}"
                    + ("" if len(units) == 1 else f" / {UNIT_NAMES[row.base_unit]}")
                    for row in latest
                ],
                "Semana": [row.observed_on.strftime("%d/%m/%y") for row in latest],
                "4 sem.": [
                    f"{row.change:+.0%}" if row.change is not None else "—" for row in latest
                ],
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")

        st.subheader("Historia")
        markets = {row.market: row for row in latest}
        chosen = st.multiselect(
            "Mercados (hasta 3)",
            list(markets),
            default=list(markets)[:1],
            max_selections=3,
            placeholder="Elige mercados",
        )
        if not chosen:
            st.caption("Elige al menos un mercado.")
        else:
            unit = markets[chosen[0]].base_unit
            same_unit = [name for name in chosen if markets[name].base_unit == unit]
            if len(same_unit) < len(chosen):
                st.caption(f"Solo se comparan mercados con precios por {UNIT_NAMES[unit]}.")
            newest = max(row.observed_on for row in latest)
            since = period_start(newest, "precios_period")
            points = history(
                product.id,
                price_type,
                unit,
                tuple(markets[name].market_id for name in same_unit),
                since,
            )
            frame = pd.DataFrame([vars(point) for point in points])
            colors = market_colors({row.market: row.market_id for row in latest}, theme())
            st.altair_chart(
                history_chart(frame, colors, theme(), UNIT_NAMES[unit]), width="stretch"
            )
            st.caption("Precios oficiales validados, por semana de reporte.")
            st.download_button(
                "Descargar historia (CSV)",
                csv(
                    frame.rename(
                        columns={"market": "mercado", "observed_on": "semana", "value": "precio"}
                    )
                ),
                file_name=f"precios-{product.id}.csv",
                mime="text/csv",
                key="download_history",
            )

with compare_tab:
    if not latest:
        st.info("No hay precios de este tipo para el producto elegido.")
    else:
        unit = latest[0].base_unit
        others = {p.id: p for p in products(price_type) if p.id != product.id}
        picked = st.multiselect(
            "Comparar con (hasta 3)",
            list(others),
            format_func=lambda pid: others[pid].name,
            max_selections=3,
            placeholder="Elige productos",
            key="compare_products",
        )
        newest = max(row.observed_on for row in latest)
        since = period_start(newest, "compare_period")
        points = products_history(tuple([product.id, *picked]), price_type, unit, since)
        frame = pd.DataFrame([vars(point) for point in points])
        names = sorted(frame["market"].unique()) if not frame.empty else []
        missing = [others[pid].name for pid in picked if others[pid].name not in names]
        if missing:
            st.caption(f"Sin precios por {UNIT_NAMES[unit]} en este periodo: {', '.join(missing)}.")
        if not frame.empty:
            # A product keeps its color whatever else is picked.
            ids = {product.name: product.id, **{others[pid].name: pid for pid in picked}}
            colors = market_colors({name: ids[name] for name in names}, theme())
            st.altair_chart(
                history_chart(frame, colors, theme(), UNIT_NAMES[unit]), width="stretch"
            )
        st.caption("Cada línea es el promedio de los mercados que publican el producto.")

with movers_tab:
    changes = price_changes(price_type)
    categories = sorted({change.category for change in changes}, key=category_name)
    rubro = st.selectbox(
        "Rubro",
        [None, *categories],
        format_func=lambda c: "Todos los rubros" if c is None else category_name(c),
        key="movers_rubro",
    )
    shown = [c for c in changes if rubro is None or c.category == rubro]
    if not shown:
        st.info("No hay productos con precios en las últimas cuatro semanas.")
    else:

        def movers(rows):
            return pd.DataFrame(
                {
                    # The change first after the name: it is what this table is about.
                    "Producto": [row.product for row in rows],
                    "4 sem.": [f"{row.change:+.0%}" for row in rows],
                    "RD$": [f"{row.value:,.2f} / {UNIT_NAMES[row.base_unit]}" for row in rows],
                }
            )

        st.subheader("Suben más")
        st.dataframe(
            movers([c for c in shown if c.change > 0][:TOP]), hide_index=True, width="stretch"
        )
        st.subheader("Bajan más")
        st.dataframe(
            movers([c for c in reversed(shown) if c.change < 0][:TOP]),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Último precio de cada producto (promedio de sus mercados) contra el de cuatro "
            "semanas antes; el cambio es la mediana de los mercados."
        )
        st.download_button(
            "Descargar todos (CSV)",
            csv(
                pd.DataFrame(
                    {
                        "producto": [c.product for c in shown],
                        "rubro": [category_name(c.category) for c in shown],
                        "unidad": [UNIT_NAMES[c.base_unit] for c in shown],
                        "precio": [round(c.value, 2) for c in shown],
                        "cambio_4_semanas": [round(c.change, 4) for c in shown],
                    }
                )
            ),
            file_name="mayores-cambios.csv",
            mime="text/csv",
            key="download_movers",
        )
