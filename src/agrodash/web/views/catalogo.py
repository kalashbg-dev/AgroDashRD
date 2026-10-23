import streamlit as st

from agrodash.accounts.models import User
from agrodash.catalog.admin import (
    CatalogError,
    confirm_alias,
    merge_family,
    merge_products,
    rename_product,
    set_market_active,
    set_product_active,
)
from agrodash.catalog.families import FamilyMember, product_families
from agrodash.catalog.models import Market, Product, ProductAlias
from agrodash.ingestion.quality import rename_candidates
from agrodash.web.messages import spanish
from agrodash.web.state import current_user, database, require

require("catalogo")
actor_id = current_user()["id"]
st.title("Catálogo")
families_tab, renames_tab, aliases_tab, products_tab, markets_tab = st.tabs(
    ["Repetidos", "Renombres", "Nombres", "Productos", "Mercados"]
)


def merge(families) -> None:
    """Merge each family into its target and list the names that could not be merged."""
    with database() as session:
        actor = session.get(User, actor_id)
        problems = [problem for family in families for problem in run_merge(session, actor, family)]
    st.cache_data.clear()
    st.session_state["family_problems"] = problems
    st.rerun()


def run_merge(session, actor, family):
    try:
        return merge_family(session, actor, family.target.id, [m.id for m in family.members])
    except CatalogError as error:
        return [(family.target.name, str(error))]


def plural(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def span(member: FamilyMember) -> str:
    return f"{member.first:%m/%Y} – {member.last:%m/%Y}, {member.prices} precios"


def run(action, *args, **kwargs) -> bool:
    """Run a catalog change as the signed-in admin; show the error in Spanish if it fails."""
    with database() as session:
        try:
            action(session, session.get(User, actor_id), *args, **kwargs)
        except CatalogError as error:
            st.error(spanish(str(error)))
            return False
    st.cache_data.clear()
    return True


with database() as session:
    products = list(session.query(Product).order_by(Product.name))
    ids = {product.name: product.id for product in products}
    candidates = [
        (old, new, days)
        for old, new, days in rename_candidates(session)
        if old in ids and new in ids and session.get(Product, ids[old]).is_active
    ]
    unreviewed = [
        (alias.id, alias.raw_name, alias.product.name)
        for alias in session.query(ProductAlias)
        .filter_by(reviewed=False)
        .order_by(ProductAlias.raw_name)
    ]
    markets = [(m.id, m.name, m.is_active) for m in session.query(Market).order_by(Market.name)]
    product_rows = [(p.id, p.name, p.is_active) for p in products]
    families = product_families(session)

with families_tab:
    safe = [family for family in families if family.safe]
    suggested = [family for family in families if not family.safe]
    st.markdown(
        f"**{plural(len(safe), 'grupo seguro', 'grupos seguros')}** y "
        f"{plural(len(suggested), 'sugerido', 'sugeridos')}: el mismo producto con nombres "
        "distintos según la época. Al unirlos, toda su historia queda en el nombre más reciente "
        "y los demás se retiran."
    )
    for name, reason in st.session_state.pop("family_problems", []):
        st.warning(f"{name}: {spanish(reason)}")
    if safe and st.button(
        f"Unir {plural(len(safe), 'grupo seguro', 'grupos seguros')}",
        key="merge_safe_families",
        type="primary",
    ):
        merge(safe)
    if suggested:
        st.caption(
            'Sugeridos: difieren en un "primera" que los reportes viejos omitían o en una '
            "letra. Revisa que sean el mismo producto antes de unirlos."
        )
    for family in suggested:
        with st.container(border=True):
            st.markdown(f"**{family.target.name}** · {span(family.target)}")
            for member in family.members[1:]:
                st.caption(f"{member.name} · {span(member)}")
            if st.button("Unir", key=f"merge_family_{family.target.id}"):
                merge([family])

with renames_tab:
    st.caption(
        "Productos cuyos precios terminan cuando empieza otro de nombre parecido. Si son el "
        "mismo producto, fusiónalos: sus precios pasan al nombre nuevo y el viejo se retira."
    )
    if not candidates:
        st.info("No hay renombres sugeridos.")
    for old, new, days in candidates:
        with st.container(border=True):
            st.markdown(f"**{old}** → **{new}**")
            st.caption(f"El nuevo empieza {days} días después de que termina el viejo.")
            if st.button("Fusionar", key=f"merge_{ids[old]}_{ids[new]}") and run(
                merge_products, ids[old], ids[new]
            ):
                st.rerun()

with aliases_tab:
    st.caption("Nombres de los reportes y el producto al que se asignaron. Confirma los correctos.")
    search = st.text_input("Buscar", key="alias_search").strip().lower()
    shown = [row for row in unreviewed if search in row[1] or search in row[2].lower()][:30]
    st.caption(f"{len(unreviewed)} sin revisar; se muestran {len(shown)}.")
    for alias_id, raw_name, product in shown:
        name_column, button_column = st.columns([4, 1])
        name_column.markdown(f"`{raw_name}` → {product}")
        if button_column.button("Confirmar", key=f"confirm_{alias_id}") and run(
            confirm_alias, alias_id
        ):
            st.rerun()

with products_tab:
    rows_by_id = {row[0]: row for row in product_rows}
    chosen = rows_by_id[
        st.selectbox(
            "Producto",
            list(rows_by_id),
            format_func=lambda pid: (
                rows_by_id[pid][1] if rows_by_id[pid][2] else f"{rows_by_id[pid][1]} (retirado)"
            ),
            key="rename_product",
        )
    ]
    new_name = st.text_input("Nuevo nombre", key="rename_name")
    rename_column, active_column = st.columns(2)
    renamed = rename_column.button("Renombrar", key="rename_submit", width="stretch")
    if renamed and run(rename_product, chosen[0], new_name):
        st.success("Producto renombrado.")
    toggled = active_column.button(
        "Retirar" if chosen[2] else "Reactivar", key="product_toggle", width="stretch"
    )
    if toggled and run(set_product_active, chosen[0], active=not chosen[2]):
        st.rerun()

with markets_tab:
    st.caption("Un mercado retirado deja de pronosticarse y de ofrecerse para reportes.")
    for market_id, name, active in markets:
        name_column, button_column = st.columns([3, 1])
        name_column.markdown(name if active else f"{name} (retirado)")
        toggled = button_column.button(
            "Retirar" if active else "Reactivar", key=f"market_{market_id}"
        )
        if toggled and run(set_market_active, market_id, active=not active):
            st.rerun()
