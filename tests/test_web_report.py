from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
import streamlit as st
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from agrodash.accounts.models import Role
from agrodash.accounts.service import create_user
from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, Unit
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType
from agrodash.prices.queries import units_for_report
from agrodash.prices.reports import local_today

PAGE = Path(__file__).parent.parent / "src" / "agrodash" / "web" / "views" / "reportar.py"


@pytest.fixture
def catalog(cli_database):
    st.cache_data.clear()
    with Session(cli_database) as session:
        user = create_user(session, "tecnico@example.com", "correct horse battery", Role.TECNICO)
        pound = Unit(code="1-lb", label="1 lb", base_unit=BaseUnit.POUND, factor=Decimal(1))
        sack = Unit(code="100-lb", label="100 lb", base_unit=BaseUnit.POUND, factor=Decimal(100))
        piece = Unit(code="1-unit", label="1 unit", base_unit=BaseUnit.UNIT, factor=Decimal(1))
        rice = Product(name="Arroz Selecto", category="Cereales", base_unit=BaseUnit.POUND)
        market = Market(name="Mercado Nuevo", kind=MarketKind.RETAIL)
        session.add_all([pound, sack, piece, rice, market])
        session.add(
            Price(
                product=rice,
                market=market,
                unit=pound,
                observed_on=date(2026, 9, 22),
                price_type=PriceType.RETAIL,
                source=PriceSource.OFFICIAL,
                status=PriceStatus.VALIDATED,
                amount=Decimal(30),
                price_per_base_unit=Decimal(30),
            )
        )
        session.commit()
        yield {
            "session_engine": cli_database,
            "user": {"id": user.id, "email": user.email, "name": None, "role": "TECNICO"},
            "rice": rice.id,
        }
    st.cache_data.clear()


def open_page(user=None) -> AppTest:
    page = AppTest.from_file(str(PAGE), default_timeout=30)
    if user:
        page.session_state["user"] = user
    return page.run()


def test_only_units_of_the_series_base_unit_are_offered(catalog):
    with Session(catalog["session_engine"]) as session:
        labels = [
            unit.label for unit in units_for_report(session, catalog["rice"], PriceType.RETAIL)
        ]

    assert labels == ["1 lb", "100 lb"]


def test_a_technician_reports_a_price_that_waits_for_review(catalog):
    page = open_page(catalog["user"])
    page.number_input(key="report_amount").set_value(3100.0)
    page.selectbox(key="report_unit").select_index(1)
    page.button(key="report_submit").click().run()

    assert not page.exception
    assert [s.value for s in page.success] == ["Reporte guardado. Queda pendiente de validación."]
    with Session(catalog["session_engine"]) as session:
        report = session.query(Price).filter_by(source=PriceSource.FIELD_REPORT).one()
        assert (report.status, report.price_per_base_unit, report.observed_on) == (
            PriceStatus.PENDING,
            Decimal("31.0000"),
            local_today(),
        )
    assert page.dataframe[0].value["Estado"].tolist() == ["Pendiente"]


def test_a_price_far_from_the_official_series_warns_the_reporter(catalog):
    page = open_page(catalog["user"])
    page.number_input(key="report_amount").set_value(90.0)
    page.button(key="report_submit").click().run()

    assert [w.value for w in page.warning] == [
        "Reporte guardado, pero está lejos del precio oficial reciente: "
        "un administrador lo revisará con atención."
    ]


def test_visitors_cannot_open_the_report_page(catalog):
    page = open_page()

    assert [e.value for e in page.error] == ["No tienes permiso para ver esta página."]
