from decimal import Decimal

import pandas as pd
import streamlit as st

from agrodash.accounts.models import User
from agrodash.prices.queries import active_markets, reports_by, units_for_report
from agrodash.prices.reports import ReportError, local_today, report_price
from agrodash.web.messages import spanish
from agrodash.web.state import current_user, database, require
from agrodash.web.ui import choose_product

STATUS_NAMES = {"pending": "Pendiente", "validated": "Validado", "rejected": "Rechazado"}

require("reportar")
user = current_user()
st.title("Reportar precio")
st.caption("Tu reporte queda pendiente hasta que un administrador lo valide.")
product, price_type = choose_product("reportar")

with database() as session:
    markets = active_markets(session)
    units = units_for_report(session, product.id, price_type)

market = st.selectbox("Mercado", markets, format_func=lambda m: m.label, key="report_market")
unit = st.selectbox("Unidad", units, format_func=lambda u: u.label, key="report_unit")
amount = st.number_input(
    "Precio (RD$)", min_value=0.01, value=None, step=1.0, format="%.2f", key="report_amount"
)
observed_on = st.date_input(
    "Fecha observada", value=local_today(), max_value=local_today(), key="report_date"
)

if st.button("Enviar reporte", type="primary", key="report_submit"):
    if amount is None:
        st.error("Escribe el precio.")
    else:
        with database() as session:
            try:
                reporter = session.get_one(User, user["id"])
                price = report_price(
                    session,
                    reporter,
                    product_id=product.id,
                    market_id=market.id,
                    unit_id=unit.id,
                    amount=Decimal(str(amount)),
                    observed_on=observed_on,
                    price_type=price_type,
                )
            except ReportError as error:
                st.error(spanish(str(error)))
            else:
                if price.is_outlier:
                    st.warning(
                        "Reporte guardado, pero está lejos del precio oficial reciente: "
                        "un administrador lo revisará con atención."
                    )
                else:
                    st.success("Reporte guardado. Queda pendiente de validación.")

with database() as session:
    mine = reports_by(session, user["id"])
if mine:
    st.subheader("Tus últimos reportes")
    st.dataframe(
        pd.DataFrame(
            {
                "Producto": [row.product for row in mine],
                "Mercado": [row.market.removeprefix("Mercado ") for row in mine],
                "Precio": [f"{float(row.amount):,.2f} / {row.unit}" for row in mine],
                "Estado": [STATUS_NAMES[row.status] for row in mine],
            }
        ),
        hide_index=True,
        width="stretch",
    )
