import pytest

from agrodash.web.messages import spanish


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Price must be greater than zero", "El precio debe ser mayor que cero."),
        (
            "Password must have at least 12 characters",
            "La contraseña debe tener al menos 12 caracteres.",
        ),
        ("You cannot create ADMIN accounts", "No puedes crear cuentas ADMIN."),
        ("ana@example.com is already registered", "ana@example.com ya está registrado."),
        (
            "Arroz Selecto retail prices are per lb; choose a unit in lb",
            "Esta serie se mide por libra: elige una unidad en libra.",
        ),
        ("Something new happened", "Something new happened"),
    ],
)
def test_domain_messages_are_shown_in_spanish(message, expected):
    assert spanish(message) == expected


def test_merging_into_a_retired_product_is_explained_in_spanish():
    assert spanish("Cannot merge into a retired product") == (
        "No se puede unir con un producto retirado. Reactívalo primero."
    )
