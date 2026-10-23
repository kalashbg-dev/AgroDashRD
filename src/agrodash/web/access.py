"""Which pages each role can open. Pure, so access rules are tested without Streamlit."""

from agrodash.accounts.models import Role

PUBLIC = "public"
REPORTERS = frozenset({Role.TECNICO, Role.ADMIN, Role.SUPERADMIN})
MANAGERS = frozenset({Role.ADMIN, Role.SUPERADMIN})

# Navigation order. Reading prices and forecasts needs no account.
PAGE_ACCESS = {
    "precios": PUBLIC,
    "pronostico": PUBLIC,
    "temporadas": PUBLIC,
    "mercados": PUBLIC,
    # "Ingresar" for visitors, "Mi cuenta" (and signing out) once signed in.
    "ingresar": PUBLIC,
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
    return role in access


def visible_pages(role: Role | None) -> list[str]:
    return [page for page in PAGE_ACCESS if can_open(role, page)]


def landing_page(role: Role | None) -> str:
    """Where a role starts: its own work, or prices for visitors."""
    if role in MANAGERS:
        return "admin"
    if role in REPORTERS:
        return "reportar"
    return "precios"
