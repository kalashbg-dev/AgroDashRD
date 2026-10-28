from datetime import date
from decimal import Decimal

import pytest

from agrodash.accounts.models import Role, User
from agrodash.catalog.admin import (
    CatalogError,
    apply_approved_merges,
    confirm_alias,
    merge_products,
    rename_product,
    set_market_active,
    set_product_active,
)
from agrodash.catalog.approved_merges import APPROVED_MERGES
from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, ProductAlias, Unit
from agrodash.ingestion.normalize import name_key
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType


@pytest.fixture
def catalog(session):
    admin = User(email="admin@example.com", password_hash="x", role=Role.ADMIN)
    technician = User(email="tecnico@example.com", password_hash="x", role=Role.TECNICO)
    old = Product(
        name="Cilantro/ grande/ primera", category="Legumbres-Hortalizas", base_unit=BaseUnit.UNIT
    )
    new = Product(
        name="Cilantro, primera", category="Legumbres-Hortalizas", base_unit=BaseUnit.UNIT
    )
    market = Market(name="Mercado Nuevo", kind=MarketKind.RETAIL)
    unit = Unit(code="1-unit", label="1 unit", base_unit=BaseUnit.UNIT, factor=Decimal(1))
    session.add_all([admin, technician, old, new, market, unit])
    session.add_all(
        [
            ProductAlias(raw_name="cilantro grande primera", product=old),
            ProductAlias(raw_name="cilantro primera", product=new),
        ]
    )
    for product, day in ((old, date(2018, 2, 8)), (new, date(2018, 2, 15))):
        session.add(price(product, market, unit, day))
    session.commit()
    return {
        "admin": admin,
        "technician": technician,
        "old": old,
        "new": new,
        "market": market,
        "unit": unit,
    }


def price(product, market, unit, day, source=PriceSource.OFFICIAL):
    return Price(
        product=product,
        market=market,
        unit=unit,
        observed_on=day,
        price_type=PriceType.RETAIL,
        source=source,
        status=PriceStatus.VALIDATED,
        amount=Decimal(25),
        price_per_base_unit=Decimal(25),
    )


def test_merging_moves_prices_and_aliases_and_retires_the_old_product(session, catalog):
    merge_products(session, catalog["admin"], catalog["old"].id, catalog["new"].id)

    assert session.query(Price).filter_by(product_id=catalog["new"].id).count() == 2
    aliases = {a.raw_name: (a.product_id, a.reviewed) for a in session.query(ProductAlias)}
    assert aliases["cilantro grande primera"] == (catalog["new"].id, True)
    assert session.get(Product, catalog["old"].id).is_active is False


def test_merging_refuses_when_both_have_an_official_price_for_the_same_week(session, catalog):
    session.add(price(catalog["old"], catalog["market"], catalog["unit"], date(2018, 2, 15)))
    session.commit()

    with pytest.raises(CatalogError, match="1 week where both products have an official price"):
        merge_products(session, catalog["admin"], catalog["old"].id, catalog["new"].id)

    assert session.query(Price).filter_by(product_id=catalog["old"].id).count() == 2


def test_a_product_cannot_be_merged_into_itself(session, catalog):
    with pytest.raises(CatalogError, match="into itself"):
        merge_products(session, catalog["admin"], catalog["new"].id, catalog["new"].id)


def test_renaming_keeps_names_unique(session, catalog):
    rename_product(
        session, catalog["admin"], catalog["old"].id, "  Cilantro  (Long Standing), grande "
    )
    assert session.get(Product, catalog["old"].id).name == "Cilantro (Long Standing), grande"

    with pytest.raises(CatalogError, match="already exists"):
        rename_product(session, catalog["admin"], catalog["old"].id, "Cilantro, primera")


def test_confirming_an_alias_marks_it_reviewed(session, catalog):
    alias = session.query(ProductAlias).filter_by(raw_name="cilantro primera").one()

    confirm_alias(session, catalog["admin"], alias.id)

    assert alias.reviewed is True


def test_products_and_markets_can_be_retired_and_restored(session, catalog):
    set_product_active(session, catalog["admin"], catalog["old"].id, active=False)
    set_market_active(session, catalog["admin"], catalog["market"].id, active=False)
    assert (catalog["old"].is_active, catalog["market"].is_active) == (False, False)

    set_market_active(session, catalog["admin"], catalog["market"].id, active=True)
    assert catalog["market"].is_active is True


@pytest.mark.parametrize(
    "action",
    [
        lambda s, c: merge_products(s, c["technician"], c["old"].id, c["new"].id),
        lambda s, c: rename_product(s, c["technician"], c["old"].id, "Otro"),
        lambda s, c: set_product_active(s, c["technician"], c["old"].id, active=False),
    ],
)
def test_only_managers_change_the_catalog(session, catalog, action):
    with pytest.raises(CatalogError, match="cannot change the catalog"):
        action(session, catalog)


def test_a_product_cannot_be_merged_into_a_retired_one(session, catalog):
    catalog["new"].is_active = False
    session.commit()

    with pytest.raises(CatalogError, match="Cannot merge into a retired product"):
        merge_products(session, catalog["admin"], catalog["old"].id, catalog["new"].id)

    assert session.query(Price).filter_by(product_id=catalog["old"].id).count() == 1


def test_an_approved_pair_already_joined_under_the_old_name_takes_the_canonical_name(session):
    joined = Product(name="Arroz Superior", category="Cereales", base_unit=BaseUnit.UNIT)
    session.add_all(
        [
            ProductAlias(raw_name=name_key("Arroz Superior"), product=joined),
            ProductAlias(raw_name=name_key("Arroz (Superior)/ primera"), product=joined),
        ]
    )
    session.commit()

    apply_approved_merges(session)

    assert joined.name == "Arroz (Superior)/ primera"


def test_no_approved_canonical_name_is_itself_merged_away():
    old_names = {name_key(old) for old, _ in APPROVED_MERGES}

    assert not {name_key(canon) for _, canon in APPROVED_MERGES} & old_names


def test_a_database_loaded_before_the_approval_catches_up_on_the_approved_merges(session, catalog):
    # Production was loaded with "Arroz Superior" apart from "Arroz (Superior)/ primera".
    old = Product(name="Arroz Superior", category="Cereales", base_unit=BaseUnit.UNIT)
    canon = Product(name="Arroz (Superior)/ primera", category="Cereales", base_unit=BaseUnit.UNIT)
    session.add_all(
        [
            ProductAlias(raw_name=name_key(old.name), product=old),
            ProductAlias(raw_name=name_key(canon.name), product=canon),
            price(old, catalog["market"], catalog["unit"], date(2018, 2, 8)),
            price(canon, catalog["market"], catalog["unit"], date(2018, 2, 15)),
        ]
    )
    session.commit()

    assert apply_approved_merges(session) == []
    assert apply_approved_merges(session) == []

    assert not old.is_active
    assert session.query(Price).filter(Price.product_id == canon.id).count() == 2
