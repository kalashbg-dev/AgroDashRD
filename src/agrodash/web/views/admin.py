import pandas as pd
import streamlit as st

from agrodash.accounts.models import User
from agrodash.accounts.service import (
    AccountError,
    create_account,
    list_users,
    manageable_roles,
    set_active,
    unlock,
)
from agrodash.infra.settings import get_settings
from agrodash.ingestion.drive import DriveError, download
from agrodash.ingestion.models import IngestionTrigger
from agrodash.ingestion.quality import quarantine, recent_runs
from agrodash.prices.reports import ReportError, pending_reports, review_report
from agrodash.web.admin import REASONS, load_report
from agrodash.web.messages import spanish
from agrodash.web.state import current_user, database, require

require("admin")
actor_id = current_user()["id"]
st.title("Administración")
load_tab, reports_tab, quarantine_tab, users_tab = st.tabs(
    ["Cargar datos", "Reportes", "Cuarentena", "Usuarios"]
)


def show(kind: str, message: str) -> None:
    getattr(st, kind)(message)


with load_tab:
    uploaded = st.file_uploader(
        "Reporte de precios (.csv, .xlsx o .xls)", type=["csv", "xlsx", "xls"], key="upload_file"
    )
    if uploaded is not None and st.button("Cargar archivo", type="primary", key="upload_submit"):
        with st.spinner("Cargando…"), database() as session:
            show(
                *load_report(
                    session, uploaded.name, uploaded.getvalue(), IngestionTrigger.UPLOAD, actor_id
                )
            )
        st.cache_data.clear()
    if st.button("Cargar desde Google Drive", key="drive_submit"):
        for file_id in get_settings().drive_file_id_list:
            with st.spinner("Descargando de Google Drive…"):
                try:
                    file_name, content = download(file_id)
                except DriveError as error:
                    st.error(f"No se pudo descargar {file_id}: {error}")
                    continue
            with st.spinner(f"Cargando {file_name}…"), database() as session:
                show(*load_report(session, file_name, content, IngestionTrigger.DRIVE, actor_id))
        st.cache_data.clear()
    with database() as session:
        runs = recent_runs(session)
    if runs:
        st.caption("Últimas cargas")
        st.dataframe(
            pd.DataFrame(
                {
                    "Archivo": [run.file_name for run in runs],
                    "Estado": [
                        {"succeeded": "Cargado", "failed": "Falló", "running": "En curso"}[
                            str(run.status)
                        ]
                        for run in runs
                    ],
                    "Precios": [run.prices_loaded for run in runs],
                    "Cuarentena": [run.rows_rejected for run in runs],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

with reports_tab:
    with database() as session:
        pending = [
            (
                report.id,
                report.product.name,
                report.market.name,
                report.observed_on,
                report.amount,
                report.unit.label,
                report.is_outlier,
                report.reporter.email if report.reporter else "—",
            )
            for report in pending_reports(session)
        ]
    if not pending:
        st.info("No hay reportes pendientes.")
    for report_id, product, market, observed_on, amount, unit, outlier, reporter in pending:
        with st.container(border=True):
            st.markdown(f"**{product}** · {market.removeprefix('Mercado ')}")
            st.caption(f"{observed_on:%d/%m/%Y} · {float(amount):,.2f} / {unit} · {reporter}")
            if outlier:
                st.warning("Lejos del precio oficial reciente.")
            approve, reject = st.columns(2)
            decision = None
            if approve.button("Validar", key=f"approve_{report_id}", use_container_width=True):
                decision = True
            if reject.button("Rechazar", key=f"reject_{report_id}", use_container_width=True):
                decision = False
            if decision is not None:
                with database() as session:
                    try:
                        review_report(
                            session, session.get(User, actor_id), report_id, approve=decision
                        )
                    except ReportError as error:
                        st.error(spanish(str(error)))
                    else:
                        st.cache_data.clear()
                        st.rerun()

with quarantine_tab:
    reason = st.selectbox(
        "Motivo", [None, *REASONS], format_func=lambda r: "Todos" if r is None else REASONS[r]
    )
    with database() as session:
        rows = quarantine(session, reason)
    st.dataframe(
        pd.DataFrame(
            {
                "Archivo": [row.file_name for row in rows],
                "Fila": [row.row_number for row in rows],
                "Motivo": [REASONS.get(row.reason, row.reason) for row in rows],
                "Detalle": [row.detail for row in rows],
            }
        ),
        hide_index=True,
        use_container_width=True,
        key="quarantine_table",
    )
    st.caption(
        "Las 200 filas más recientes. El detalle completo sale con `agrodash quality-report`."
    )

with users_tab:
    with database() as session:
        actor = session.get(User, actor_id)
        roles = manageable_roles(actor.role)
        people = [
            (
                user.id,
                user.email,
                user.full_name,
                user.role,
                user.is_active,
                user.locked_until is not None,
            )
            for user in list_users(session)
        ]
    st.dataframe(
        pd.DataFrame(
            {
                "Correo": [p[1] for p in people],
                "Nombre": [p[2] or "" for p in people],
                "Rol": [str(p[3]).lower() for p in people],
                "Activa": ["sí" if p[4] else "no" for p in people],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.subheader("Nueva cuenta")
    email = st.text_input("Correo", key="new_email")
    name = st.text_input("Nombre (opcional)", key="new_name")
    role = st.selectbox("Rol", roles, format_func=lambda r: str(r).lower(), key="new_role")
    password = st.text_input("Contraseña inicial", type="password", key="new_password")
    if st.button("Crear cuenta", type="primary", key="new_submit"):
        with database() as session:
            try:
                created = create_account(
                    session, session.get(User, actor_id), email, password, role, name or None
                )
            except AccountError as error:
                st.error(spanish(str(error)))
            else:
                st.success(f"Cuenta creada para {created.email}.")
    others = [p for p in people if p[3] in roles and p[0] != actor_id]
    if others:
        st.subheader("Gestionar cuenta")
        chosen = st.selectbox("Cuenta", others, format_func=lambda p: p[1], key="manage_user")
        toggle, unlock_button = st.columns(2)
        with database() as session:
            try:
                if toggle.button(
                    "Desactivar" if chosen[4] else "Activar",
                    key="manage_toggle",
                    use_container_width=True,
                ):
                    set_active(
                        session, session.get(User, actor_id), chosen[0], active=not chosen[4]
                    )
                    st.rerun()
                if unlock_button.button(
                    "Desbloquear",
                    key="manage_unlock",
                    disabled=not chosen[5],
                    use_container_width=True,
                ):
                    unlock(session, session.get(User, actor_id), chosen[0])
                    st.rerun()
            except AccountError as error:
                st.error(spanish(str(error)))
