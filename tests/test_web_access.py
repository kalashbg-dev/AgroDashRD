import pytest

from agrodash.accounts.models import Role
from agrodash.web.access import can_open, landing_page, visible_pages


@pytest.mark.parametrize(
    ("role", "pages"),
    [
        (None, ["precios", "pronostico", "temporadas", "mercados", "ingresar"]),
        (Role.TECNICO, ["precios", "pronostico", "temporadas", "mercados", "ingresar", "reportar"]),
        (
            Role.ADMIN,
            [
                "precios",
                "pronostico",
                "temporadas",
                "mercados",
                "ingresar",
                "reportar",
                "admin",
                "catalogo",
            ],
        ),
        (
            Role.SUPERADMIN,
            [
                "precios",
                "pronostico",
                "temporadas",
                "mercados",
                "ingresar",
                "reportar",
                "admin",
                "catalogo",
            ],
        ),
    ],
)
def test_each_role_sees_its_pages(role, pages):
    assert visible_pages(role) == pages


@pytest.mark.parametrize(
    ("role", "page", "allowed"),
    [
        (None, "precios", True),
        (None, "reportar", False),
        (None, "admin", False),
        (Role.TECNICO, "reportar", True),
        (Role.TECNICO, "admin", False),
        (Role.TECNICO, "catalogo", False),
        (Role.ADMIN, "admin", True),
    ],
)
def test_pages_check_the_role_themselves(role, page, allowed):
    assert can_open(role, page) is allowed


def test_unknown_pages_are_never_open():
    assert can_open(Role.SUPERADMIN, "debug") is False


@pytest.mark.parametrize(
    ("role", "page"),
    [
        (None, "precios"),
        (Role.TECNICO, "reportar"),
        (Role.ADMIN, "admin"),
        (Role.SUPERADMIN, "admin"),
    ],
)
def test_each_role_starts_on_its_own_page(role, page):
    assert landing_page(role) == page
