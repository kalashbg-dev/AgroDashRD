from datetime import date
from decimal import Decimal

import pytest

from agrodash.catalog.models import BaseUnit
from agrodash.ingestion.normalize import (
    InvalidValue,
    name_key,
    normalize_category,
    parse_amount,
    parse_unit,
    parse_week_date,
    same_product_name,
)


@pytest.mark.parametrize(
    ("week", "month", "year", "expected"),
    [
        ("Primera ", "Agosto", "2017", date(2017, 8, 1)),
        ("Priemera", "ABril", 2018, date(2018, 4, 1)),
        ("Sengunda", "`Mayo", 2019.0, date(2019, 5, 8)),
        ("Segundo", "julio", "2020", date(2020, 7, 8)),
        ("Tercero", "Juio", "2021", date(2021, 7, 15)),
        ("Cuanta", "Marzo", "2022", date(2022, 3, 22)),
        ("Quinto", "Enero", "2025`", date(2025, 1, 29)),
        ("Quinta", "Febrero", "2023", date(2023, 2, 28)),
    ],
)
def test_week_words_with_typos_map_to_a_date_inside_the_month(week, month, year, expected):
    assert parse_week_date(week, month, year) == expected


@pytest.mark.parametrize(
    ("week", "month", "year"),
    [
        ("3333", "Enero", "2020"),
        ("", "", ""),
        ("Primera", "|", "2020"),
        ("Primera", "Enero", "20252025+B38846:C38849"),
    ],
)
def test_unreadable_dates_are_invalid(week, month, year):
    with pytest.raises(InvalidValue):
        parse_week_date(week, month, year)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (2200, Decimal("2200")),
        (76.475, Decimal("76.475")),
        (" 11,500.00 ", Decimal("11500.00")),
        ("8,5", Decimal("8.5")),
        ("21", Decimal("21")),
        ("0,001", Decimal("0.001")),
        ("1,500", Decimal("1500")),
    ],
)
def test_amounts_accept_numbers_and_spanish_or_english_separators(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("raw", ["N/D", "n/d", "", "  ", "-", None, "0", 0])
def test_missing_amounts_are_none(raw):
    assert parse_amount(raw) is None


@pytest.mark.parametrize("raw", ["abc", "-5", "1.2.3", "NaN", "inf", float("nan"), float("inf")])
def test_unreadable_or_negative_amounts_are_invalid(raw):
    with pytest.raises(InvalidValue):
        parse_amount(raw)


@pytest.mark.parametrize(
    ("raw", "base", "factor"),
    [
        ("Saco/100 lb", BaseUnit.POUND, "100"),
        ("Saco/125 libra", BaseUnit.POUND, "125"),
        ("Huacal/50lb", BaseUnit.POUND, "50"),
        ("Unidad/ 07 lb", BaseUnit.POUND, "7"),
        ("Paq/1.5 Lb.", BaseUnit.POUND, "1.5"),
        ("Paq/ Lib.", BaseUnit.POUND, "1"),
        ("Quintal", BaseUnit.POUND, "100"),
        ("Kilo", BaseUnit.POUND, "2.2046"),
        ("Lata/1Kilo", BaseUnit.POUND, "2.2046"),
        ("lb", BaseUnit.POUND, "1"),
        ("Ib", BaseUnit.POUND, "1"),
        ("Ciento", BaseUnit.UNIT, "100"),
        ("Huacal/220 Unidad", BaseUnit.UNIT, "220"),
        ("Saco/180 unid/125/lib", BaseUnit.UNIT, "180"),
        ("Fardo/12 Ud", BaseUnit.UNIT, "12"),
        ("Doc", BaseUnit.UNIT, "12"),
        ("Ud.", BaseUnit.UNIT, "1"),
        ("Mata", BaseUnit.UNIT, "1"),
        ("Paquete", BaseUnit.UNIT, "1"),
        ("Paq", BaseUnit.UNIT, "1"),
        ("Jarro/Lata", BaseUnit.UNIT, "1"),
        ("Cubeta/5 gal", BaseUnit.LITER, "18.9271"),
        ("Cubeta/5 gl", BaseUnit.LITER, "18.9271"),
        ("Litro", BaseUnit.LITER, "1"),
    ],
)
def test_units_convert_to_a_base_unit_and_factor(raw, base, factor):
    unit = parse_unit(raw)

    assert (unit.base_unit, unit.factor) == (base, Decimal(factor))


def test_unit_code_and_label_are_shared_by_containers_of_the_same_size():
    sack, crate = parse_unit("Saco/100 lb"), parse_unit("Huacal/100 lb")

    assert sack.code == crate.code == "100-lb"
    assert sack.label == "100 lb"


@pytest.mark.parametrize(
    "raw",
    ["Saco", "0", "", "N/D", "N/A", "Saco/1N/DN/D lb", "2800", "Saco/0 lb", "Huacal/0.00001 lb"],
)
def test_units_without_a_known_size_are_unknown(raw):
    assert parse_unit(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Cereal ", "Cereales"),
        ("Celeales", "Cereales"),
        ("Celeares", "Cereales"),
        ("Lateo", "Lacteos"),
        ("Pescuario", "Pecuarios"),
        ("Raices - Tuberculos", "Raices-Tuberculos"),
        ("Legumbres y Hortalizas", "Legumbres-Hortalizas"),
        ("Avícolas", "Avicolas"),
        ("Oleginosas", "Oleaginosas"),
        ("musaceas", "Musaceas"),
        ("Leguminosas secas", "Leguminosas Secas"),
        ("", "Otros"),
        ("Bebidas", "Otros"),
    ],
)
def test_rubro_spellings_map_to_one_category(raw, expected):
    assert normalize_category(raw) == expected


def test_name_key_ignores_case_accents_spacing_and_punctuation():
    assert name_key(" Maíz en grano/ (Francés largo)") == name_key("maiz en grano (frances  largo)")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Ají cubanela", "Aji cubanella"),
        ("Berenjena (Morada)", "Berengena (Morada)"),
    ],
)
def test_typo_level_differences_are_the_same_product(left, right):
    assert same_product_name(name_key(left), name_key(right))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Arroz Selecto/ primera", "Arroz Selecto/ segunda"),
        ("Huevos (Consumo), primera, grande", "Huevos (Consumo), primera, mediano"),
        ("Saco 100 lb", "Saco 125 lb"),
        ("Papa", "Pera"),
        ("Arroz Selecto", "Arroz Selecto importado"),
    ],
)
def test_different_grades_sizes_or_words_are_different_products(left, right):
    assert not same_product_name(name_key(left), name_key(right))
