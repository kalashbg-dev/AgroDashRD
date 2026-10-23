from decimal import Decimal
from pathlib import Path

import pytest
import streamlit as st
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from agrodash.catalog.models import BaseUnit, Market, MarketKind, Product, Unit
from agrodash.forecasting.service import run_forecasts
from agrodash.ingestion import quality
from agrodash.ingestion.normalize import week_slot
from agrodash.prices.models import Price, PriceSource, PriceStatus, PriceType

PAGES = Path(__file__).parent.parent / "src" / "agrodash" / "web" / "views"


@pytest.fixture
def prices(cli_database, monkeypatch):
    """Arroz with 3 months of weekly prices in two markets (forecast), Papa with one week."""
    monkeypatch.setattr(quality, "MIN_WEEKS", 8)
    st.cache_data.clear()
    with Session(cli_database) as session:
        unit = Unit(code="1-lb", label="1 lb", base_unit=BaseUnit.POUND, factor=Decimal(1))
        rice = Product(name="Arroz Selecto", category="Cereales", base_unit=BaseUnit.POUND)
        potato = Product(name="Papa", category="Raices-Tuberculos", base_unit=BaseUnit.POUND)
        markets = [
            Market(name=name, kind=MarketKind.RETAIL) for name in ("Mercado Nuevo", "Mercadom")
        ]
        session.add_all([unit, rice, potato, *markets])
        value = 30
        for month in (7, 8, 9):
            for week in range(1, 5):
                value += 1
                for market in markets:
                    session.add(price(rice, market, unit, week_slot(2026, month, week), value))
        session.add(price(potato, markets[0], unit, week_slot(2026, 9, 4), 20))
        # Sorts before "Arroz Selecto" but has not been reported since 2020.
        old = Product(name="Arroz Antiguo", category="Cereales", base_unit=BaseUnit.POUND)
        session.add(old)
        session.add(price(old, markets[0], unit, week_slot(2020, 1, 4), 15))
        session.commit()
        run_forecasts(session, run_benchmark=False)
    yield
    st.cache_data.clear()


def price(product, market, unit, day, value):
    return Price(
        product=product,
        market=market,
        unit=unit,
        observed_on=day,
        price_type=PriceType.RETAIL,
        source=PriceSource.OFFICIAL,
        status=PriceStatus.VALIDATED,
        amount=Decimal(value),
        price_per_base_unit=Decimal(value),
    )


def open_page(name: str) -> AppTest:
    return AppTest.from_file(str(PAGES / f"{name}.py"), default_timeout=30).run()


def test_prices_page_shows_the_latest_prices_and_a_history_chart(prices):
    page = open_page("precios")

    assert not page.exception
    table = page.dataframe[0].value
    assert list(table.columns) == ["Mercado", "RD$ / libra", "Semana", "4 sem."]
    assert table["Mercado"].tolist() == ["Nuevo", "Mercadom"]
    assert table["RD$ / libra"].tolist() == ["42.00", "42.00"]
    assert table["Semana"].tolist() == ["22/09/26", "22/09/26"]
    assert table["4 sem."].tolist() == ["+11%", "+11%"]
    assert page.get("vega_lite_chart")


def test_forecast_page_shows_next_week_with_its_range(prices):
    page = open_page("pronostico")

    assert not page.exception
    assert page.metric[0].value == "RD$ 42.00 / libra"
    assert page.get("vega_lite_chart")
    assert page.metric[0].label == "Próxima semana (01/10)"
    # "$" is escaped, or Markdown would render "RD$ 42.00 y RD$" as a formula.
    assert r"entre RD\$ 42.00 y RD\$" in page.caption[0].value
    assert page.dataframe[0].value["Semana"].tolist() == [
        "01/10/2026",
        "08/10/2026",
        "15/10/2026",
        "22/10/2026",
    ]


def test_products_without_forecast_say_why(prices):
    page = open_page("pronostico")
    page.selectbox(key="pronostico_category").select("Raices-Tuberculos").run()

    assert not page.exception
    assert page.subheader[0].value == "Sin pronóstico"
    text = page.markdown[0].value
    assert "tiene menos de dos años de precios" in text
    assert "le faltan demasiadas semanas" in text


def test_pickers_only_offer_products_with_prices_of_the_chosen_type(prices):
    page = open_page("precios")
    page.button_group(key="precios_type").set_value("Mayorista").run()

    assert not page.exception
    assert [info.value for info in page.info] == ["No hay precios mayoristas cargados."]


def test_rubros_are_shown_in_proper_spanish(prices):
    page = open_page("precios")

    assert page.selectbox(key="precios_category").options == ["Cereales", "Raíces y tubérculos"]


def test_the_picker_opens_on_a_product_with_recent_prices(prices):
    page = open_page("precios")

    picker = page.selectbox(key="precios_product")
    assert picker.value.name == "Arroz Selecto"
    assert picker.options == ["Arroz Antiguo (hasta 01/2020)", "Arroz Selecto"]


def test_pages_say_so_when_there_is_no_data(cli_database):
    st.cache_data.clear()

    page = open_page("precios")

    assert [info.value for info in page.info] == ["No hay precios minoristas cargados."]


@pytest.fixture
def seasonal(cli_database):
    """Papaya over three full years: dear in March, cheap in September."""
    st.cache_data.clear()
    with Session(cli_database) as session:
        unit = Unit(code="1-unit", label="1 unit", base_unit=BaseUnit.UNIT, factor=Decimal(1))
        papaya = Product(name="Lechosa", category="Frutas", base_unit=BaseUnit.UNIT)
        market = Market(name="Mercado Nuevo", kind=MarketKind.RETAIL)
        session.add_all([unit, papaya, market])
        for year in (2023, 2024, 2025):
            for month in range(1, 13):
                value = {3: 130, 9: 70}.get(month, 100)
                session.add(price(papaya, market, unit, week_slot(year, month, 1), value))
        session.commit()
    yield
    st.cache_data.clear()


def test_seasons_page_names_the_dear_and_cheap_months_and_draws_three_charts(seasonal):
    page = open_page("temporadas")

    assert not page.exception
    assert "**Precio alto:** marzo" in page.markdown[0].value
    assert "**Precio bajo:** septiembre" in page.markdown[0].value
    assert len(page.get("vega_lite_chart")) == 3
    assert not page.warning


def test_seasons_page_needs_a_full_year_of_prices(prices):
    page = open_page("temporadas")

    assert not page.exception
    assert "año completo" in page.info[0].value


def test_markets_page_ranks_markets_values_a_harvest_and_explains_a_missing_spread(prices):
    page = open_page("mercados")

    assert not page.exception
    assert "de diferencia" in page.markdown[0].value
    page.number_input[0].set_value(2.0).run()
    page.selectbox[2].select("quintales (100 lb)").run()
    harvest = page.dataframe[0].value
    assert harvest["Valor"].tolist() == ["RD$ 8,400.00", "RD$ 8,400.00"]
    assert "mismos meses" in page.info[0].value


def test_markets_page_compares_wholesale_and_retail_by_month(prices, cli_database):
    with Session(cli_database) as session:
        rice = session.query(Product).filter_by(name="Arroz Selecto").one()
        market = Market(name="Mercado Mayorista", kind=MarketKind.WHOLESALE)
        unit = session.query(Unit).one()
        wholesale = price(rice, market, unit, week_slot(2026, 9, 1), 30)
        wholesale.price_type = PriceType.WHOLESALE
        session.add_all([market, wholesale])
        session.commit()
    st.cache_data.clear()

    page = open_page("mercados")

    assert not page.exception
    assert "09/2026" in page.markdown[1].value
    assert len(page.get("vega_lite_chart")) == 2


def test_forecast_page_offers_longer_plazos_only_with_three_years_of_prices(prices):
    page = open_page("pronostico")

    page.segmented_control(key="pronostico_horizon").set_value("12 meses").run()

    assert not page.exception
    assert "hacen falta tres años" in page.info[0].value
    assert list(page.dataframe[0].value.columns) == ["Semana", "Pronóstico", "Rango 80 %"]


@pytest.fixture
def years_of_prices(cli_database):
    """Papaya every week since 2023, dear in March and cheap in September, forecast."""
    st.cache_data.clear()
    with Session(cli_database) as session:
        unit = Unit(code="1-unit", label="1 unit", base_unit=BaseUnit.UNIT, factor=Decimal(1))
        papaya = Product(name="Lechosa", category="Frutas", base_unit=BaseUnit.UNIT)
        market = Market(name="Mercado Nuevo", kind=MarketKind.RETAIL)
        session.add_all([unit, papaya, market])
        for year in (2023, 2024, 2025, 2026):
            for month in range(1, 13 if year < 2026 else 10):
                value = {3: 130, 9: 70}.get(month, 100)
                for week in range(1, 5):
                    session.add(price(papaya, market, unit, week_slot(year, month, week), value))
        session.commit()
        run_forecasts(session, run_benchmark=False)
    yield
    st.cache_data.clear()


def test_forecast_page_shows_a_year_month_by_month_with_the_season(years_of_prices):
    page = open_page("pronostico")

    page.segmented_control(key="pronostico_horizon").set_value("12 meses").run()

    assert not page.exception
    assert page.metric[0].label == "En 12 meses (09/2027)"
    table = page.dataframe[0].value
    assert list(table.columns) == ["Fecha", "Rango 80 % (RD$)", "Temporada"]
    assert len(table) == 15
    assert table["Fecha"].tolist()[4] == "11/2026"
    assert table.set_index("Fecha").loc["03/2027", "Temporada"] == "Alto"


def test_prices_page_compares_products_lists_movers_and_downloads_csv(prices):
    page = open_page("precios")
    page.multiselect(key="compare_products").select("Arroz Antiguo").run()
    page.segmented_control(key="compare_period").set_value("Todo").run()

    assert not page.exception
    assert len(page.get("vega_lite_chart")) == 2
    movers = page.dataframe[1].value
    assert movers["Producto"].tolist() == ["Arroz Selecto"]
    assert movers["4 sem."].tolist() == ["+11%"]
    assert {button.proto.label for button in page.get("download_button")} == {
        "Descargar historia (CSV)",
        "Descargar todos (CSV)",
    }


def test_seasons_page_warns_with_fewer_than_three_years(seasonal, cli_database):
    with Session(cli_database) as session:
        session.query(Price).filter(Price.observed_on < week_slot(2024, 1, 1)).delete()
        session.commit()
    st.cache_data.clear()

    page = open_page("temporadas")

    assert not page.exception
    assert "menos de 3 años" in page.warning[0].value
