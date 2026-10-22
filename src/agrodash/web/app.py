"""AgroDashRD web app: `agrodash web` or `streamlit run src/agrodash/web/app.py`."""

from pathlib import Path

import streamlit as st

from agrodash.web.access import visible_pages
from agrodash.web.state import current_role, current_user, sign_out

PAGES = Path(__file__).parent / "pages"
TITLES = {
    "precios": ("Precios", ":material/sell:"),
    "pronostico": ("Pronóstico", ":material/trending_up:"),
    "ingresar": ("Ingresar", ":material/login:"),
    "reportar": ("Reportar precio", ":material/edit_note:"),
    "admin": ("Administración", ":material/admin_panel_settings:"),
    "catalogo": ("Catálogo", ":material/inventory_2:"),
}

st.set_page_config(page_title="AgroDashRD", page_icon=":material/agriculture:", layout="centered")

pages = [
    st.Page(PAGES / f"{name}.py", title=TITLES[name][0], icon=TITLES[name][1], url_path=name)
    for name in visible_pages(current_role())
]
user = current_user()
if user:
    with st.sidebar:
        st.caption(f"{user['name'] or user['email']} · {user['role'].lower()}")
        st.button("Cerrar sesión", on_click=sign_out, use_container_width=True)
st.navigation(pages).run()
