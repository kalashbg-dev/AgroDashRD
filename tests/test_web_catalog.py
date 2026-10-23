from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from agrodash.accounts.models import Role
from agrodash.accounts.service import create_user
from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, ProductAlias, Unit
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType

PAGE = Path(__file__).parent.parent / "src" / "agrodash" / "web" / "views" / "catalogo.py"


@pytest.fixture
def setup(cli_database):
    with Session(cli_database) as session:
        admin = create_user(session, "admin@example.com", "correct horse battery", Role.ADMIN)
        old = Product(
            name="Cilantro/ grande/ primera",
            category="Legumbres-Hortalizas",
            base_unit=BaseUnit.UNIT,
        )
        new = Product(
            name="Cilantro, primera", category="Legumbres-Hortalizas", base_unit=BaseUnit.UNIT
        )
        market = Market(name="Mercado Nuevo", kind=MarketKind.RETAIL)
        unit = Unit(code="1-unit", label="1 unit", base_unit=BaseUnit.UNIT, factor=Decimal(1))
        session.add_all([old, new, market, unit])
        session.add_all(
            [
                ProductAlias(raw_name="cilantro grande primera", product=old),
                ProductAlias(raw_name="cilantro primera", product=new),
            ]
        )
        for product, day in ((old, date(2018, 2, 8)), (new, date(2018, 2, 15))):
            session.add(
                Price(
                    product=product,
                    market=market,
                    unit=unit,
                    observed_on=day,
                    price_type=PriceType.RETAIL,
                    source=PriceSource.OFFICIAL,
                    status=PriceStatus.VALIDATED,
                    amount=Decimal(25),
                    price_per_base_unit=Decimal(25),
                )
            )
        session.commit()
        yield {
            "engine": cli_database,
            "user": {"id": admin.id, "email": admin.email, "name": None, "role": "ADMIN"},
            "old": old.id,
            "new": new.id,
        }


def open_page(user) -> AppTest:
    page = AppTest.from_file(str(PAGE), default_timeout=30)
    page.session_state["user"] = user
    return page.run()


def test_a_suggested_rename_is_merged_with_one_click(setup):
    page = open_page(setup["user"])

    page.button(key=f"merge_{setup['old']}_{setup['new']}").click().run()

    assert not page.exception
    with Session(setup["engine"]) as session:
        assert session.query(Price).filter_by(product_id=setup["new"]).count() == 2
        assert session.get(Product, setup["old"]).is_active is False


def test_an_alias_is_confirmed(setup):
    page = open_page(setup["user"])
    with Session(setup["engine"]) as session:
        alias_id = session.query(ProductAlias).filter_by(raw_name="cilantro primera").one().id

    page.button(key=f"confirm_{alias_id}").click().run()

    with Session(setup["engine"]) as session:
        assert session.get(ProductAlias, alias_id).reviewed is True


def test_a_product_is_renamed_and_errors_are_in_spanish(setup):
    page = open_page(setup["user"])
    # Options are sorted by name: "Cilantro, primera" then "Cilantro/ grande/ primera".
    page.selectbox(key="rename_product").select_index(1).run()
    page.text_input(key="rename_name").input("Cilantro, primera").run()
    page.button(key="rename_submit").click().run()

    assert [e.value for e in page.error] == ["Ya existe un producto llamado 'Cilantro, primera'."]

    page.text_input(key="rename_name").input("Cilantro (Long Standing), grande").run()
    page.button(key="rename_submit").click().run()

    with Session(setup["engine"]) as session:
        names = {p.name for p in session.query(Product)}
    assert "Cilantro (Long Standing), grande" in names


def test_safe_families_are_merged_together_and_suggestions_one_by_one(setup):
    with Session(setup["engine"]) as session:
        market = session.query(Market).one()
        unit = session.query(Unit).one()
        for name, day in (
            ("Habichuela pinta (Gira)", date(2017, 8, 1)),
            ("Habichuela gira (Pinta)", date(2018, 1, 1)),
        ):
            product = Product(name=name, category="Legumbres", base_unit=BaseUnit.UNIT)
            session.add(product)
            session.add(
                Price(
                    product=product,
                    market=market,
                    unit=unit,
                    observed_on=day,
                    price_type=PriceType.RETAIL,
                    source=PriceSource.OFFICIAL,
                    status=PriceStatus.VALIDATED,
                    amount=Decimal(25),
                    price_per_base_unit=Decimal(25),
                )
            )
        session.commit()
    page = open_page(setup["user"])

    assert "1 grupo seguro" in page.markdown[0].value
    page.button(key="merge_safe_families").click().run()

    assert not page.exception
    with Session(setup["engine"]) as session:
        active = [p.name for p in session.query(Product).filter_by(is_active=True)]
    assert "Habichuela gira (Pinta)" in active
    assert "Habichuela pinta (Gira)" not in active
