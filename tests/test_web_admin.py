from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
import streamlit as st
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from agrodash.accounts.models import Role, User
from agrodash.accounts.service import create_user
from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, Unit
from agrodash.ingestion.models import IngestionTrigger
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType
from agrodash.prices.reports import report_price
from agrodash.web.admin import load_report

FIXTURES = Path(__file__).parent / "fixtures"
PAGE = Path(__file__).parent.parent / "src" / "agrodash" / "web" / "views" / "admin.py"
PASSWORD = "correct horse battery"


@pytest.fixture
def setup(cli_database):
    st.cache_data.clear()
    with Session(cli_database) as session:
        admin = create_user(session, "admin@example.com", PASSWORD, Role.ADMIN)
        technician = create_user(session, "tecnico@example.com", PASSWORD, Role.TECNICO)
        pound = Unit(code="1-lb", label="1 lb", base_unit=BaseUnit.POUND, factor=Decimal(1))
        rice = Product(name="Arroz Selecto", category="Cereales", base_unit=BaseUnit.POUND)
        market = Market(name="Mercado Nuevo", kind=MarketKind.RETAIL)
        session.add_all([pound, rice, market])
        session.commit()
        report = report_price(
            session,
            technician,
            product_id=rice.id,
            market_id=market.id,
            unit_id=pound.id,
            amount=Decimal("31"),
            observed_on=date(2026, 9, 1),
            price_type=PriceType.RETAIL,
        )
        yield {
            "engine": cli_database,
            "admin": {"id": admin.id, "email": admin.email, "name": None, "role": "ADMIN"},
            "technician": {
                "id": technician.id,
                "email": technician.email,
                "name": None,
                "role": "TECNICO",
            },
            "report_id": report.id,
        }
    st.cache_data.clear()


def open_page(user) -> AppTest:
    page = AppTest.from_file(str(PAGE), default_timeout=30)
    page.session_state["user"] = user
    return page.run()


def test_technicians_cannot_open_administration(setup):
    page = open_page(setup["technician"])

    assert [error.value for error in page.error] == ["No tienes permiso para ver esta página."]


def test_an_admin_validates_a_pending_report(setup):
    page = open_page(setup["admin"])

    page.button(key=f"approve_{setup['report_id']}").click().run()

    assert not page.exception
    with Session(setup["engine"]) as session:
        assert session.get(Price, setup["report_id"]).status is PriceStatus.VALIDATED


def test_an_admin_creates_a_technician_account(setup):
    page = open_page(setup["admin"])
    page.text_input(key="new_email").input("nuevo@example.com")
    page.text_input(key="new_password").input(PASSWORD)
    page.button(key="new_submit").click().run()

    assert [s.value for s in page.success] == ["Cuenta creada para nuevo@example.com."]
    with Session(setup["engine"]) as session:
        assert session.query(User).filter_by(email="nuevo@example.com").one().role is Role.TECNICO


def test_account_errors_are_shown_in_spanish(setup):
    page = open_page(setup["admin"])
    page.text_input(key="new_email").input("nuevo@example.com")
    page.text_input(key="new_password").input("corta")
    page.button(key="new_submit").click().run()

    assert [e.value for e in page.error] == ["La contraseña debe tener al menos 12 caracteres."]


def test_quarantine_tab_lists_rejected_rows(setup):
    with Session(setup["engine"]) as session:
        load_report(
            session,
            "mayorista.csv",
            (FIXTURES / "mayorista.csv").read_bytes(),
            IngestionTrigger.UPLOAD,
            setup["admin"]["id"],
        )

    page = open_page(setup["admin"])

    (table,) = [df.value for df in page.dataframe if "Motivo" in df.value.columns]
    assert len(table) == 6
    assert "Unidad sin tamaño conocido" in table["Motivo"].tolist()


def test_loading_a_report_describes_the_result(setup):
    content = (FIXTURES / "mayorista.csv").read_bytes()
    with Session(setup["engine"]) as session:
        first = load_report(
            session, "mayorista.csv", content, IngestionTrigger.UPLOAD, setup["admin"]["id"]
        )
        again = load_report(
            session, "mayorista.csv", content, IngestionTrigger.UPLOAD, setup["admin"]["id"]
        )
        broken = load_report(
            session, "precios.pdf", b"%PDF", IngestionTrigger.UPLOAD, setup["admin"]["id"]
        )

    assert first == (
        "success",
        "mayorista.csv: 5 precios cargados, 6 filas en cuarentena, 1 valores vacíos.",
    )
    assert again == ("info", "mayorista.csv ya estaba cargado.")
    assert broken[0] == "error" and broken[1].startswith("precios.pdf: no se pudo cargar")
    with Session(setup["engine"]) as session:
        assert session.query(Price).filter_by(source=PriceSource.OFFICIAL).count() == 5


def test_an_admin_deactivated_mid_session_loses_the_page_and_the_session(setup):
    with Session(setup["engine"]) as session:
        session.get(User, setup["admin"]["id"]).is_active = False
        session.commit()

    page = open_page(setup["admin"])

    assert [error.value for error in page.error] == ["No tienes permiso para ver esta página."]
    assert "user" not in page.session_state
    assert not page.button


def test_a_demoted_account_gets_the_pages_of_its_current_role(setup):
    with Session(setup["engine"]) as session:
        session.get(User, setup["admin"]["id"]).role = Role.TECNICO
        session.commit()

    page = open_page(setup["admin"])

    assert [error.value for error in page.error] == ["No tienes permiso para ver esta página."]
    assert page.session_state["user"]["role"] == "TECNICO"


def test_loading_data_refuses_a_deactivated_admin(setup):
    with Session(setup["engine"]) as session:
        session.get(User, setup["admin"]["id"]).is_active = False
        session.commit()
        result = load_report(
            session,
            "mayorista.csv",
            (FIXTURES / "mayorista.csv").read_bytes(),
            IngestionTrigger.UPLOAD,
            setup["admin"]["id"],
        )

    assert result == ("error", "Tu cuenta no puede cargar datos.")
    with Session(setup["engine"]) as session:
        assert session.query(Price).filter_by(source=PriceSource.OFFICIAL).count() == 0
