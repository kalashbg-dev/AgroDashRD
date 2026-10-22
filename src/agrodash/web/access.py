"""Which pages each role can open. Pure, so access rules are tested without Streamlit."""

from agrodash.accounts.models import Role

PUBLIC = "public"
SIGNED_OUT_ONLY = "signed_out_only"
REPORTERS = frozenset({Role.TECNICO, Role.ADMIN, Role.SUPERADMIN})
MANAGERS = frozenset({Role.ADMIN, Role.SUPERADMIN})

# Navigation order. Reading prices and forecasts needs no account.
PAGE_ACCESS = {
    "precios": PUBLIC,
    "pronostico": PUBLIC,
    "ingresar": SIGNED_OUT_ONLY,
    "reportar": REPORTERS,
    "admin": MANAGERS,
    "catalogo": MANAGERS,
}


def can_open(role: Role | None, page: str) -> bool:
    access = PAGE_ACCESS.get(page)
    if access is None:
        return False
    if access == PUBLIC:
        return True
    if access == SIGNED_OUT_ONLY:
        return role is None
    return role in access


def visible_pages(role: Role | None) -> list[str]:
    return [page for page in PAGE_ACCESS if can_open(role, page)]
