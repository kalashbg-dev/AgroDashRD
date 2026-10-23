"""Shared pieces for the Streamlit pages: Spanish labels, theme and cached queries."""

from datetime import timedelta

import streamlit as st

from agrodash.forecasting import queries as forecast_queries
from agrodash.ingestion.quality import series_coverage
from agrodash.prices import queries as price_queries
from agrodash.prices.models import PriceType
from agrodash.web.state import database

UNIT_NAMES = {"lb": "libra", "unit": "unidad", "l": "litro"}
PRICE_TYPES = {"Minorista": PriceType.RETAIL, "Mayorista": PriceType.WHOLESALE}
# The catalog stores rubros without accents (matching the reports); show them properly.
CATEGORY_NAMES = {
    "Avicolas": "Avícolas",
    "Lacteos": "Lácteos",
    "Legumbres-Hortalizas": "Legumbres y hortalizas",
    "Leguminosas Secas": "Leguminosas secas",
    "Musaceas": "Musáceas",
    "Raices-Tuberculos": "Raíces y tubérculos",
}
NOT_READY = {
    "inactive": "el producto o el mercado fue retirado",
    "short": "tiene menos de dos años de precios",
    "sparse": "le faltan demasiadas semanas",
    "stale": "no tiene precios en las últimas ocho semanas",
}
# A product without prices for this long is no longer in the reports.
STALE_AFTER = timedelta(weeks=8)
# Data changes once a week; ten minutes keeps pages fast without showing old prices long.
CACHE_SECONDS = 600
LOADING = "Cargando datos…"


def theme() -> str:
    return "dark" if st.context.theme.type == "dark" else "light"


def market_name(name: str) -> str:
    """Market names without the repeated "Mercado " prefix, to fit narrow screens."""
    return name.removeprefix("Mercado ")


def money(value) -> str:
    return f"RD$ {float(value):,.2f}"


def money_md(value) -> str:
    """money() for Markdown text: two "$" in one line would render as a formula."""
    return money(value).replace("$", r"\$")


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=LOADING)
def products(price_type: PriceType):
    with database() as session:
        return price_queries.products_with_prices(session, price_type)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=LOADING)
def latest_prices(product_id: int, price_type: PriceType):
    with database() as session:
        return price_queries.latest_prices(session, product_id, price_type)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=LOADING)
def history(product_id, price_type, base_unit, market_ids: tuple[int, ...], since):
    with database() as session:
        return price_queries.price_history(
            session, product_id, price_type, base_unit, list(market_ids), since
        )


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=LOADING)
def forecasts(product_id: int, price_type: PriceType):
    with database() as session:
        return forecast_queries.latest_forecasts(session, product_id, price_type)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=LOADING)
def coverage():
    with database() as session:
        return series_coverage(session)


def category_name(category: str) -> str:
    return CATEGORY_NAMES.get(category, category)


def choose_product(key: str):
    """Price type first, then rubro and product among those with prices of that type."""
    label = st.segmented_control(
        "Tipo de precio", list(PRICE_TYPES), default="Minorista", key=f"{key}_type"
    )
    label = label or "Minorista"
    price_type = PRICE_TYPES[label]
    choices = products(price_type)
    if not choices:
        st.info(f"No hay precios {label.lower()}s cargados.")
        st.stop()
    categories = sorted({product.category for product in choices}, key=category_name)
    category = st.selectbox("Rubro", categories, format_func=category_name, key=f"{key}_category")
    in_category = [product for product in choices if product.category == category]
    # Open on a product the reports still publish; label the ones they stopped.
    current = max(product.latest_on for product in choices) - STALE_AFTER
    recent = [index for index, p in enumerate(in_category) if p.latest_on >= current]
    product = st.selectbox(
        "Producto",
        in_category,
        index=recent[0] if recent else 0,
        format_func=lambda p: (
            p.name if p.latest_on >= current else f"{p.name} (hasta {p.latest_on:%m/%Y})"
        ),
        key=f"{key}_product",
    )
    return product, price_type
