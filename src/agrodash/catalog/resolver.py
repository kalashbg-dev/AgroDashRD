from sqlalchemy import select
from sqlalchemy.orm import Session

from agrodash.catalog.models import (
    BaseUnit,
    Market,
    MarketKind,
    Product,
    ProductAlias,
    Unit,
    UnitAlias,
)
from agrodash.ingestion.normalize import (
    clean_name,
    name_key,
    parse_unit,
    same_product_name,
    word_set_key,
)


class CatalogResolver:
    """Finds the catalog row for raw source text, drafting it when it does not exist yet.

    Drafted product aliases start unreviewed so an admin can confirm or fix the match.
    The resolver caches lookups, so use one per load and commit through its session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._products = {
            alias.raw_name: alias.product for alias in session.scalars(select(ProductAlias))
        }
        self._by_words = {word_set_key(key): product for key, product in self._products.items()}
        self._units = {alias.raw_text: alias.unit for alias in session.scalars(select(UnitAlias))}
        self._units_by_code = {unit.code: unit for unit in session.scalars(select(Unit))}
        self._markets = {market.name: market for market in session.scalars(select(Market))}

    def product(self, raw_name: str, category: str, base_unit: BaseUnit) -> Product:
        """Product for a raw name; `category` and `base_unit` only apply to new products."""
        key = name_key(raw_name)
        if key in self._products:
            return self._products[key]
        product = self._by_words.get(word_set_key(key)) or next(
            (
                known
                for known_key, known in self._products.items()
                if same_product_name(key, known_key)
            ),
            None,
        )
        if product is None:
            product = Product(
                name=clean_name(raw_name)[:120], category=category, base_unit=base_unit
            )
            self._session.add(product)
        self._session.add(ProductAlias(raw_name=key, product=product, reviewed=False))
        self._products[key] = product
        self._by_words.setdefault(word_set_key(key), product)
        return product

    def unit(self, raw_text: str) -> Unit | None:
        """Unit for a raw label, or None when its size cannot be known."""
        raw = clean_name(raw_text)[:120]
        if raw in self._units:
            return self._units[raw]
        parsed = parse_unit(raw)
        if parsed is None:
            return None
        unit = self._units_by_code.get(parsed.code)
        if unit is None:
            unit = Unit(
                code=parsed.code,
                label=parsed.label,
                base_unit=parsed.base_unit,
                factor=parsed.factor,
            )
            self._session.add(unit)
            self._units_by_code[parsed.code] = unit
        self._session.add(UnitAlias(raw_text=raw, unit=unit))
        self._units[raw] = unit
        return unit

    def market(self, name: str, kind: MarketKind) -> Market:
        name = clean_name(name)
        if name not in self._markets:
            self._markets[name] = Market(name=name, kind=kind)
            self._session.add(self._markets[name])
        return self._markets[name]
