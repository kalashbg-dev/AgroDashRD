"""Turn the messy cells of the official price reports into typed values.

Every function here is pure: the same cell always gives the same value, so ingestion can
be re-run and tested without a database. The fuzzy ones are cached because a report
repeats the same few hundred names, units and dates across tens of thousands of rows.
"""

import calendar
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from functools import lru_cache

from agrodash.catalog.models import BaseUnit


class InvalidValue(ValueError):
    """A cell that holds something, but nothing we can read."""


MONTHS = {
    name: number
    for number, name in enumerate(
        [
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        ],
        start=1,
    )
}
# Typos too close to two months to guess. "Juio" only appears in 2019, where June is
# complete and July is missing half its rows.
MONTH_TYPOS = {"juio": 7}
# Week ordinals are typed by hand ("Priemera", "Sengunda", "Cuanta"); the first letters
# are reliable, the rest is not.
WEEK_PREFIXES = {"pri": 1, "seg": 2, "sen": 2, "ter": 3, "cua": 4, "qui": 5}

CATEGORIES = [
    "Avicolas",
    "Cereales",
    "Frutas",
    "Lacteos",
    "Legumbres-Hortalizas",
    "Leguminosas Secas",
    "Musaceas",
    "Oleaginosas",
    "Pecuarios",
    "Raices-Tuberculos",
]
UNKNOWN_CATEGORY = "Otros"

MISSING_TEXT = {"", "-", "n/d", "nd", "n/a", "0"}

POUNDS_PER_KILO = Decimal("2.20462")
LITERS_PER_GALLON = Decimal("3.785411784")
# Measure words found in unit labels, with their size in base units.
MEASURES = {
    "lb": (BaseUnit.POUND, Decimal(1)),
    "lib": (BaseUnit.POUND, Decimal(1)),
    "libra": (BaseUnit.POUND, Decimal(1)),
    "libras": (BaseUnit.POUND, Decimal(1)),
    "ib": (BaseUnit.POUND, Decimal(1)),
    "quintal": (BaseUnit.POUND, Decimal(100)),
    "kilo": (BaseUnit.POUND, POUNDS_PER_KILO),
    "kg": (BaseUnit.POUND, POUNDS_PER_KILO),
    "unidad": (BaseUnit.UNIT, Decimal(1)),
    "unidades": (BaseUnit.UNIT, Decimal(1)),
    "unid": (BaseUnit.UNIT, Decimal(1)),
    "und": (BaseUnit.UNIT, Decimal(1)),
    "ud": (BaseUnit.UNIT, Decimal(1)),
    "mata": (BaseUnit.UNIT, Decimal(1)),
    "doc": (BaseUnit.UNIT, Decimal(12)),
    "docena": (BaseUnit.UNIT, Decimal(12)),
    "ciento": (BaseUnit.UNIT, Decimal(100)),
    # Packages without a stated size (cilantro bunches, canned cherries) are priced per piece.
    "paquete": (BaseUnit.UNIT, Decimal(1)),
    "paq": (BaseUnit.UNIT, Decimal(1)),
    "lata": (BaseUnit.UNIT, Decimal(1)),
    "litro": (BaseUnit.LITER, Decimal(1)),
    "lt": (BaseUnit.LITER, Decimal(1)),
    "gl": (BaseUnit.LITER, LITERS_PER_GALLON),
    "gal": (BaseUnit.LITER, LITERS_PER_GALLON),
    "galon": (BaseUnit.LITER, LITERS_PER_GALLON),
}
FACTOR_PLACES = Decimal("0.0001")

# Words that tell grades or sizes apart: names differing in them are different products.
GRADE_WORDS = {
    "primera",
    "primero",
    "segunda",
    "segundo",
    "tercera",
    "tercero",
    "grande",
    "mediano",
    "mediana",
    "pequeno",
    "pequena",
    "extra",
    "importado",
    "importada",
    "nacional",
}


@dataclass(frozen=True)
class ParsedUnit:
    base_unit: BaseUnit
    factor: Decimal

    @property
    def code(self) -> str:
        return f"{_plain(self.factor)}-{self.base_unit}"

    @property
    def label(self) -> str:
        return f"{_plain(self.factor)} {self.base_unit}"


def fold(text: object) -> str:
    """Lowercase, accent-free text with single spaces."""
    without_accents = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore")
    return " ".join(without_accents.decode().lower().split())


@lru_cache(maxsize=4096)
def name_key(name: str) -> str:
    """Key under which spelling variants of the same product name collide."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", fold(name)).split())


def clean_name(name: str) -> str:
    return " ".join(str(name).split())


def same_product_name(left_key: str, right_key: str) -> bool:
    """Whether two name keys differ only by a typo in one word.

    Numbers and grade words must match exactly, so "primera" and "segunda" or
    "100 lb" and "125 lb" stay different products.
    """
    left, right = left_key.split(), right_key.split()
    if len(left) != len(right):
        return False
    differing = [(a, b) for a, b in zip(left, right, strict=True) if a != b]
    if len(differing) != 1:
        return not differing
    a, b = differing[0]
    if not (a.isalpha() and b.isalpha()) or min(len(a), len(b)) < 4:
        return False
    if a in GRADE_WORDS or b in GRADE_WORDS:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.8


@lru_cache(maxsize=4096)
def parse_week_date(week: object, month: object, year: object) -> date:
    """Date for "week N of month": day 1 + 7*(N-1), never past the month's end."""
    week_number = WEEK_PREFIXES.get(fold(week)[:3])
    month_number = _parse_month(month)
    year_number = _parse_year(year)
    if week_number is None or month_number is None or year_number is None:
        raise InvalidValue(f"unreadable week {week!r} {month!r} {year!r}")
    last_day = calendar.monthrange(year_number, month_number)[1]
    return date(year_number, month_number, min(1 + 7 * (week_number - 1), last_day))


def _parse_month(month: object) -> int | None:
    letters = re.sub(r"[^a-z]", "", fold(month))
    if letters in MONTH_TYPOS:
        return MONTH_TYPOS[letters]
    scores = sorted((SequenceMatcher(None, letters, name).ratio(), name) for name in MONTHS)
    (second, _), (best, name) = scores[-2:]
    # A tie ("juio" is as close to junio as to julio) is a guess, not a reading.
    if best < 0.75 or best == second:
        return None
    return MONTHS[name]


def _parse_year(year: object) -> int | None:
    if isinstance(year, float) and year.is_integer():
        year = int(year)
    found = re.fullmatch(r"\D*(\d{4})\D*", str(year).strip())
    if not found or not 2000 <= int(found[1]) <= 2100:
        return None
    return int(found[1])


def parse_amount(value: object) -> Decimal | None:
    """Price in the cell, or None when the report has no price (N/D, blank, 0)."""
    if value is None:
        return None
    if isinstance(value, int | float):
        amount = Decimal(str(value))
    else:
        text = str(value).strip()
        if text.lower() in MISSING_TEXT:
            return None
        amount = _parse_decimal_text(text)
    if not amount.is_finite():
        raise InvalidValue(f"unreadable price {value!r}")
    if amount == 0:
        return None
    if amount < 0:
        raise InvalidValue(f"negative price {value!r}")
    return amount


def _parse_decimal_text(text: str) -> Decimal:
    if "," in text and "." in text or re.fullmatch(r"[1-9]\d{0,2}(,\d{3})+", text):
        text = text.replace(",", "")
    else:
        text = text.replace(",", ".")
    try:
        return Decimal(text)
    except InvalidOperation as error:
        raise InvalidValue(f"unreadable price {text!r}") from error


@lru_cache(maxsize=4096)
def parse_unit(text: object) -> ParsedUnit | None:
    """Base unit and size of a selling unit label, or None when it cannot be known.

    "Saco/100 lb" is 100 lb; "Huacal/220 Unidad" is 220 units; a bare "Paquete" is one
    piece. A bare "Saco" has no size, so prices in it cannot be compared and are rejected.
    """
    folded = fold(text).rstrip(".")
    if folded in MISSING_TEXT or "n/d" in folded:
        return None
    if folded in MEASURES:
        return _unit(*MEASURES[folded])
    quantity = re.search(r"(\d+(?:[.,]\d+)?)\s*([a-z]+)", folded)
    if quantity and quantity[2] in MEASURES:
        base_unit, size = MEASURES[quantity[2]]
        return _unit(base_unit, Decimal(quantity[1].replace(",", ".")) * size)
    last_word = re.sub(r"[^a-z]", "", folded.rsplit("/", 1)[-1])
    if "/" in folded and last_word in MEASURES:
        return _unit(*MEASURES[last_word])
    return None


def _unit(base_unit: BaseUnit, factor: Decimal) -> ParsedUnit | None:
    factor = factor.quantize(FACTOR_PLACES)
    # "Saco/0 lb" is a typo, not a size; dividing by it would be meaningless.
    return ParsedUnit(base_unit, factor) if factor > 0 else None


def _plain(number: Decimal) -> str:
    return f"{number.normalize():f}"


@lru_cache(maxsize=4096)
def normalize_category(rubro: object) -> str:
    letters = _letters(fold(rubro).replace(" y ", " "))
    best_score, best = max((_typo_similarity(letters, _letters(c)), c) for c in CATEGORIES)
    return best if best_score >= 0.75 else UNKNOWN_CATEGORY


def _letters(text: object) -> str:
    return re.sub(r"[^a-z]", "", fold(text))


def _typo_similarity(left: str, right: str) -> float:
    # Sorted letters also catch swapped letters ("Celeares" for "Cereales").
    return max(
        SequenceMatcher(None, left, right).ratio(),
        SequenceMatcher(None, sorted(left), sorted(right)).ratio(),
    )
