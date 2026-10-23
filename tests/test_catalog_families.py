from datetime import date, timedelta
from decimal import Decimal

import pytest

from agrodash.accounts.models import Role, User
from agrodash.catalog.admin import CatalogError, merge_family
from agrodash.catalog.families import product_families
from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, Unit
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType


@pytest.fixture
def shop(session):
    market = Market(name="Mercado Nuevo", kind=MarketKind.RETAIL)
    unit = Unit(code="1-lb", label="1 lb", base_unit=BaseUnit.POUND, factor=Decimal(1))
    admin = User(email="admin@example.com", password_hash="x", role=Role.ADMIN)
    session.add_all([market, unit, admin])
    session.commit()

    def product(name, first, weeks=3, base_unit=BaseUnit.POUND):
        item = Product(name=name, category="Legumbres", base_unit=base_unit)
        session.add(item)
        for week in range(weeks):
            session.add(
                Price(
                    product=item,
                    market=market,
                    unit=unit,
                    observed_on=first + timedelta(weeks=week),
                    price_type=PriceType.RETAIL,
                    source=PriceSource.OFFICIAL,
                    status=PriceStatus.VALIDATED,
                    amount=Decimal(30),
                    price_per_base_unit=Decimal(30),
                )
            )
        session.commit()
        return item

    return {"product": product, "admin": admin}


def names(family):
    return [member.name for member in family.members]


def test_names_with_the_same_words_are_a_safe_family_led_by_the_latest_name(session, shop):
    shop["product"]("Habichuela pinta (Gira)/ primera", date(2017, 8, 1))
    shop["product"]("Habichuela gira (Pinta)/ primera", date(2018, 1, 1))

    (family,) = product_families(session)

    assert family.safe is True
    assert family.target.name == "Habichuela gira (Pinta)/ primera"
    assert names(family) == ["Habichuela gira (Pinta)/ primera", "Habichuela pinta (Gira)/ primera"]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("Ají/ Cachucha/ verde", "Ají (Cachucha)/ verde/ primera"),
        ("Berenjena (Morada)/ primera", "Berengena (Morada)/ primera"),
    ],
)
def test_a_missing_primera_or_a_typo_is_only_a_suggestion(session, shop, old, new):
    shop["product"](old, date(2017, 8, 1))
    shop["product"](new, date(2018, 2, 1))

    (family,) = product_families(session)

    assert family.safe is False
    assert family.target.name == new


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Ajo/ grande/ primera", "Ajo/ mediano/ primera"),
        ("Ajo/ primera", "Ajo/ segunda"),
    ],
)
def test_other_sizes_and_grades_are_never_a_family(session, shop, left, right):
    shop["product"](left, date(2017, 8, 1))
    shop["product"](right, date(2018, 2, 1))

    assert product_families(session) == []


def test_other_base_units_and_retired_products_are_never_a_family(session, shop):
    shop["product"]("Ajo/ primera", date(2017, 8, 1))
    shop["product"]("Primera/ ajo", date(2018, 2, 1), base_unit=BaseUnit.UNIT)
    retired = shop["product"]("Ajo primera", date(2019, 2, 1))
    retired.is_active = False
    session.commit()

    assert product_families(session) == []


def test_merging_a_family_moves_every_price_to_its_target(session, shop):
    old = shop["product"]("Ají/ Cachucha/ verde", date(2017, 8, 1))
    older = shop["product"]("Ají Cachucha verde primera", date(2016, 8, 1))
    new = shop["product"]("Ají (Cachucha)/ verde/ primera", date(2018, 2, 1))

    (family,) = product_families(session)
    problems = merge_family(
        session, shop["admin"], family.target.id, [m.id for m in family.members]
    )

    assert problems == []
    assert session.query(Price).filter_by(product_id=new.id).count() == 9
    assert (old.is_active, older.is_active, new.is_active) == (False, False, True)
    assert product_families(session) == []


def test_a_member_that_cannot_be_merged_is_reported_and_the_rest_still_merge(session, shop):
    shop["product"]("Ajo/ primera", date(2018, 2, 1))
    clashing = shop["product"]("Primera ajo", date(2018, 2, 1))
    later = shop["product"]("Ajo primera", date(2020, 1, 1))

    (family,) = product_families(session)
    problems = merge_family(session, shop["admin"], later.id, [m.id for m in family.members])

    assert problems == [
        (
            "Primera ajo",
            "3 weeks where both products have an official price; fix or delete one of them first",
        )
    ]
    assert clashing.is_active is True
    assert session.query(Price).filter_by(product_id=later.id).count() == 6


def test_only_managers_merge_families(session, shop):
    technician = User(email="t@example.com", password_hash="x", role=Role.TECNICO)
    session.add(technician)
    session.commit()

    with pytest.raises(CatalogError, match="cannot change the catalog"):
        merge_family(session, technician, 1, [1, 2])
