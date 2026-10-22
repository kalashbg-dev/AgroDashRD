from decimal import Decimal

from agrodash.catalog.models import (
    BaseUnit,
    Market,
    MarketKind,
    Product,
    ProductAlias,
    Unit,
    UnitAlias,
)
from agrodash.catalog.resolver import CatalogResolver


def test_spelling_variants_share_one_product_with_unreviewed_aliases(session):
    resolver = CatalogResolver(session)

    first = resolver.product("Arroz Selecto/ primera", "Cereales", BaseUnit.POUND)
    second = resolver.product(" arroz selecto, PRIMERA ", "Cereales", BaseUnit.POUND)
    session.commit()

    assert first is second
    assert first.name == "Arroz Selecto/ primera"
    assert (first.category, first.base_unit) == ("Cereales", BaseUnit.POUND)
    aliases = session.query(ProductAlias).all()
    assert [(alias.raw_name, alias.reviewed) for alias in aliases] == [
        ("arroz selecto primera", False)
    ]


def test_typos_join_the_existing_product(session):
    resolver = CatalogResolver(session)

    pepper = resolver.product("Ají cubanela", "Legumbres-Hortalizas", BaseUnit.POUND)
    typo = resolver.product("Aji cubanella", "Legumbres-Hortalizas", BaseUnit.UNIT)
    session.commit()

    assert typo is pepper
    assert pepper.base_unit is BaseUnit.POUND
    assert session.query(ProductAlias).count() == 2


def test_other_grades_are_new_products(session):
    resolver = CatalogResolver(session)

    first = resolver.product("Arroz Selecto/ primera", "Cereales", BaseUnit.POUND)
    second = resolver.product("Arroz Selecto/ segunda", "Cereales", BaseUnit.POUND)
    session.commit()

    assert first is not second
    assert session.query(Product).count() == 2


def test_a_new_resolver_reuses_the_stored_catalog(session):
    CatalogResolver(session).product("Papa (Granola)", "Raices-Tuberculos", BaseUnit.POUND)
    session.commit()

    again = CatalogResolver(session).product("PAPA granola", "Raices-Tuberculos", BaseUnit.POUND)

    assert session.query(Product).one() is again


def test_units_are_created_once_per_size_with_one_alias_per_spelling(session):
    resolver = CatalogResolver(session)

    sack = resolver.unit("Saco/100 lb")
    crate = resolver.unit("Huacal/100 lb")
    again = resolver.unit("Saco/100 lb")
    session.commit()

    assert sack is crate is again
    assert (sack.code, sack.label, sack.base_unit, sack.factor) == (
        "100-lb",
        "100 lb",
        BaseUnit.POUND,
        Decimal("100"),
    )
    assert sorted(alias.raw_text for alias in session.query(UnitAlias)) == [
        "Huacal/100 lb",
        "Saco/100 lb",
    ]


def test_units_without_a_size_resolve_to_none_and_store_nothing(session):
    assert CatalogResolver(session).unit("Saco") is None
    assert session.query(Unit).count() == 0


def test_markets_are_created_once(session):
    resolver = CatalogResolver(session)

    first = resolver.market("Mercado Nuevo", MarketKind.WHOLESALE)
    second = resolver.market("Mercado Nuevo ", MarketKind.RETAIL)
    session.commit()

    assert first is second
    assert session.query(Market).one().kind is MarketKind.WHOLESALE
