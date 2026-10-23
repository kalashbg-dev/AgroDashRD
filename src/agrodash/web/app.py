"""AgroDashRD web app: `agrodash web` or `streamlit run src/agrodash/web/app.py`."""

from pathlib import Path

import streamlit as st

from agrodash.web.access import landing_page, visible_pages
from agrodash.web.state import current_role, current_user, sign_out

# Not "pages/": Streamlit would turn that folder into an automatic multipage app whose
# URLs run each file directly, skipping this navigation (and its role-based menu).
PAGES = Path(__file__).parent / "views"
ASSETS = Path(__file__).parent / "assets"
TITLES = {
    "precios": ("Precios", ":material/sell:"),
    "pronostico": ("Pronóstico", ":material/trending_up:"),
    "temporadas": ("Temporadas", ":material/calendar_month:"),
    "mercados": ("Mercados", ":material/storefront:"),
    "ingresar": ("Mi cuenta", ":material/account_circle:")
    if current_user()
    else ("Ingresar", ":material/login:"),
    "reportar": ("Reportar precio", ":material/edit_note:"),
    "admin": ("Administración", ":material/admin_panel_settings:"),
    "catalogo": ("Catálogo", ":material/inventory_2:"),
}

st.set_page_config(page_title="AgroDashRD", page_icon=str(ASSETS / "icon.svg"), layout="centered")
# One logo for both themes: the theme is not always known on a session's first run, so
# its colors read on the light and the dark sidebar (3.5:1 or more on each).
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")

role = current_role()
pages = {
    name: st.Page(
        PAGES / f"{name}.py",
        title=TITLES[name][0],
        icon=TITLES[name][1],
        url_path=name,
        default=name == landing_page(role),
    )
    for name in visible_pages(role)
}
user = current_user()
if user:
    with st.sidebar:
        # Plain text: Markdown would turn the email address into a mailto link.
        st.text(f"{user['name'] or user['email']} · {user['role'].lower()}")
        st.button("Cerrar sesión", on_click=sign_out, width="stretch")
navigation = st.navigation(list(pages.values()))
# Right after signing in, go to the role's own page instead of staying on the account page.
if st.session_state.pop("just_signed_in", False):
    st.switch_page(pages[landing_page(role)])
navigation.run()
