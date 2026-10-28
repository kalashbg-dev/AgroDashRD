import pytest

from agrodash.catalog.names import NameParts, name_parts


@pytest.mark.parametrize(
    ("name", "parts"),
    [
        (
            "Berenjena/ (Pompadur)/ morada oscura/ grande",
            NameParts("Berenjena", "Pompadur", "morado oscuro", "grande", None),
        ),
        ("Ají (Gustoso), verde, segunda", NameParts("Ají", "Gustoso", "verde", None, "segunda")),
        (
            "Cebolla roja (Importada) primera, pequeña",
            NameParts("Cebolla", "Importada", "rojo", "pequeño", "primera"),
        ),
        (
            "Habichuela roja/ José Beta (corta)/ segunda",
            NameParts("Habichuela", "José Beta", "rojo", "corto", "segunda"),
        ),
        ("Mango Keitt/ grande/ primera", NameParts("Mango", "Keitt", None, "grande", "primera")),
        (
            "Maíz amarillo (Francés Largo), segunda",
            NameParts("Maíz", "Francés Largo", "amarillo", None, "segunda"),
        ),
        ("Naranja (Agria), 2da", NameParts("Naranja", "Agria", None, None, "segunda")),
        (
            "Aguacate (Criollo), primera, medinao",
            NameParts("Aguacate", "Criollo", None, "mediano", "primera"),
        ),
        (
            "Plátano Macho x Hembra/ grande (fila 2)",
            NameParts("Plátano", "Macho x Hembra", None, "grande", None),
        ),
        ("Plátano (Maduro) Grande", NameParts("Plátano", None, "maduro", "grande", None)),
        ("Leche (Líquida)", NameParts("Leche", "Líquida", None, None, None)),
        ("Molondrón", NameParts("Molondrón")),
    ],
)
def test_a_product_name_splits_into_the_properties_it_states(name, parts):
    assert name_parts(name) == parts
