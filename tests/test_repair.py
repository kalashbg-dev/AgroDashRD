from decimal import Decimal

from agrodash.catalog.models import BaseUnit
from agrodash.ingestion.normalize import ParsedUnit
from agrodash.ingestion.repair import (
    UnitLedger,
    UnitSighting,
    nullify_copied_market_columns,
    peer_median,
    repair_rows,
    shifted_market_cells,
    shifted_price_blocks,
    unit_in_effect,
)

DATE_COLUMNS = (5, 6, 7)


def row(name, week, month="Agosto", year="2017"):
    return [name, "Raices", "Mayorista", "Quintal", "100", week, month, year]


def test_an_unreadable_date_between_agreeing_rows_is_repaired_and_the_original_kept():
    rows = [row("Papa", "Primera"), row("Yuca", "3333"), row("Batata", "Primera")]

    repaired = repair_rows(rows, 1, DATE_COLUMNS)

    assert repaired.rows[1] == (2, row("Yuca", "Primera"))
    assert repaired.originals[2] == row("Yuca", "3333")
    assert repaired.dates_corrected == 1


def test_readable_dates_and_rows_without_a_date_are_left_alone():
    shifted = ["Ñame", "Raices", "Mayorista", "Quintal", "100", "", "", ""]
    rows = [
        row("Papa", "Primera"),
        row("Yuca", "Segunda"),
        row("Batata", "Primera"),
        shifted,
        row("Ajo", "Primera"),
    ]

    repaired = repair_rows(rows, 1, DATE_COLUMNS)

    assert [cells for _, cells in repaired.rows] == rows
    assert repaired.dates_corrected == 0


def test_a_name_repeated_right_below_becomes_a_second_product():
    rows = [row("Berenjena", "Primera"), row("Berenjena", "Primera"), row("Berenjena", "Segunda")]

    names = [cells[0] for _, cells in repair_rows(rows, 1, DATE_COLUMNS).rows]

    assert names == ["Berenjena", "Berenjena (fila 2)", "Berenjena"]


def test_a_repeat_far_away_in_the_block_keeps_its_name():
    names = ["Papa", "Yuca", "Batata", "Ñame", "Papa"]

    repaired = repair_rows([row(name, "Tercera") for name in names], 1, DATE_COLUMNS)

    assert [cells[0] for _, cells in repaired.rows] == names


def test_enye_exported_as_semicolon_is_restored_and_blank_rows_are_skipped():
    rows = [row("Aguacate, peque;o", "Primera"), [None] * 8, row("Papa", "Primera")]

    repaired = repair_rows(rows, 10, DATE_COLUMNS)

    assert [(number, cells[0]) for number, cells in repaired.rows] == [
        (10, "Aguacate, pequeño"),
        (12, "Papa"),
    ]


RETAIL_DATE_COLUMNS = (11, 12, 13)
RETAIL_MARKET_COLUMNS = range(4, 11)


def retail_row(name, week, markets, month="Enero", year="2024"):
    return [name, "Cereales", "Minorista", "lb", *markets, week, month, year]


def number(rows):
    return list(enumerate(rows, start=2))


def test_a_market_column_equal_to_another_market_in_the_same_week_is_voided():
    rows = [retail_row(f"Arroz {i}", "Primera", [20, 21, 22, 23, 24, 100, 100]) for i in range(1, 13)]

    voided = nullify_copied_market_columns(
        number(rows), RETAIL_DATE_COLUMNS, RETAIL_MARKET_COLUMNS
    )

    assert voided == set(range(2, 14))
    assert [cells[10] for cells in rows] == [None] * 12
    assert [cells[9] for cells in rows] == [100] * 12


def test_a_market_column_equal_to_another_market_a_week_earlier_is_voided():
    rows = [
        retail_row(f"Arroz {i}", "Primera", [None] * 5 + [100, None]) for i in range(1, 13)
    ] + [
        retail_row(f"Arroz {i}", "Segunda", [None, None, None, None, 100, None, None])
        for i in range(1, 13)
    ]

    voided = nullify_copied_market_columns(
        number(rows), RETAIL_DATE_COLUMNS, RETAIL_MARKET_COLUMNS
    )

    assert voided == set(range(14, 26))
    assert [cells[8] for cells in rows[12:]] == [None] * 12
    assert [cells[9] for cells in rows[:12]] == [100] * 12


def test_a_market_column_that_differs_from_the_others_is_not_voided():
    rows = [retail_row(f"Arroz {i}", "Primera", [20, 21, 22, 23, 24, 25, 26]) for i in range(1, 13)]

    voided = nullify_copied_market_columns(
        number(rows), RETAIL_DATE_COLUMNS, RETAIL_MARKET_COLUMNS
    )

    assert voided == set()
    assert [cells[4] for cells in rows] == [20] * 12


def test_a_market_column_with_too_little_overlap_is_not_voided():
    rows = [
        retail_row(f"Arroz {i}", "Primera", [None] * 5 + [100, None]) for i in range(1, 10)
    ] + [
        retail_row(f"Arroz {i}", "Segunda", [None, None, None, None, 100, None, None])
        for i in range(1, 10)
    ]

    voided = nullify_copied_market_columns(
        number(rows), RETAIL_DATE_COLUMNS, RETAIL_MARKET_COLUMNS
    )

    assert voided == set()
    assert [cells[8] for cells in rows[9:]] == [100] * 9

def test_a_column_repeating_itself_across_weeks_is_not_voided():
    rows = [
        retail_row(f"Arroz {i}", week, [None] * 5 + [100, None])
        for week in ("Primera", "Segunda")
        for i in range(1, 13)
    ]

    voided = nullify_copied_market_columns(
        number(rows), RETAIL_DATE_COLUMNS, RETAIL_MARKET_COLUMNS
    )

    assert voided == set()
    assert [cells[9] for cells in rows] == [100] * 24


WHOLESALE_DATE_COLUMNS = (5, 6, 7)


def wholesale_block(week, month, year, prices):
    rows = []
    for index, price in enumerate(prices, start=1):
        rows.append(
            [f"Producto {index}", "Cereales", "Mayorista", "Quintal", price, week, month, year]
        )
    return rows


def test_a_wholesale_week_whose_price_column_slid_down_is_quarantined():
    rows = (
        wholesale_block("Tercera", "Junio", "2025", [100 + i for i in range(12)])
        + wholesale_block("Cuarta", "Junio", "2025", [None] + [100 + i for i in range(11)])
        + wholesale_block("Quinta", "Junio", "2025", [200 + i for i in range(12)])
    )

    fired = shifted_price_blocks(number(rows), WHOLESALE_DATE_COLUMNS)

    assert fired == set(range(14, 26))
    assert not (fired & set(range(2, 14)))


def test_a_wholesale_week_that_matches_a_sane_following_week_one_row_up_is_quarantined():
    rows = (
        wholesale_block("Primera", "Julio", "2025", [None] + [300 + i for i in range(11)])
        + wholesale_block("Segunda", "Julio", "2025", [300 + i for i in range(12)])
    )

    fired = shifted_price_blocks(number(rows), WHOLESALE_DATE_COLUMNS)

    assert fired == set(range(2, 14))


def test_a_regular_wholesale_week_is_not_quarantined():
    rows = (
        wholesale_block("Primera", "Junio", "2025", [100 + i for i in range(12)])
        + wholesale_block("Segunda", "Junio", "2025", [100 + i for i in range(12)])
    )

    assert not shifted_price_blocks(number(rows), WHOLESALE_DATE_COLUMNS)


def test_a_short_sam_name_priced_overlap_does_not_fire():
    rows = (
        wholesale_block("Primera", "Junio", "2025", [100 + i for i in range(6)])
        + wholesale_block("Segunda", "Junio", "2025", [None] + [100 + i for i in range(5)])
    )

    assert not shifted_price_blocks(number(rows), WHOLESALE_DATE_COLUMNS)


def test_a_retail_market_column_that_slid_down_is_voided():
    rows = [
        retail_row(f"Arroz {i}", "Primera", [500 + i, 510 + i, 520 + i, 530 + i, 540 + i, 550 + i, 560 + i])
        for i in range(12)
    ] + [
        retail_row(f"Arroz {i}", "Segunda", [950 + i, 960 + i, 970 + i, None, None, None, None])
        for i in range(12)
    ]
    # Primera's Los Mina column (the third market) carries Segunda's prices one row
    # up: nothing first, then the value that belongs to the row above of the same
    # product's name in Segunda.
    for index in range(1, 12):
        rows[index][6] = 970 + index - 1

    voided = shifted_market_cells(number(rows), RETAIL_DATE_COLUMNS, RETAIL_MARKET_COLUMNS)

    assert voided == {row_number: {6} for row_number in range(2, 14)}


def test_the_los_mina_column_marked_by_the_audit_is_voided_without_signal():
    rows = [
        retail_row(f"Arroz {i}", "Segunda", [20 + i, 30 + i, 40 + i, None, None, None, None], "Agosto", "2017")
        for i in range(12)
    ]

    voided = shifted_market_cells(number(rows), RETAIL_DATE_COLUMNS, RETAIL_MARKET_COLUMNS)

    assert voided == {row_number: {6} for row_number in range(2, 14)}


def test_a_nominal_relabel_whose_price_holds_keeps_the_previous_weeks_unit():
    market = "Referencia mayorista (Mercadom y Mercado Nuevo)"
    previous_unit = object()
    previous = {market: UnitSighting(Decimal(1000), BaseUnit.POUND, previous_unit)}
    relabelled = ParsedUnit(BaseUnit.POUND, Decimal(20))

    unit = unit_in_effect(relabelled, [(market, Decimal(1000))], previous)

    assert unit is previous_unit


def test_a_relabel_with_a_price_move_is_a_real_bigger_package():
    market = "Referencia mayorista (Mercadom y Mercado Nuevo)"
    previous = {market: UnitSighting(Decimal(1080), BaseUnit.POUND, object())}
    bigger = ParsedUnit(BaseUnit.POUND, Decimal(20))

    unit = unit_in_effect(bigger, [(market, Decimal(1240))], previous)

    assert unit is bigger


def test_a_relabel_in_another_base_unit_never_borrows_the_previous_unit():
    market = "Referencia mayorista (Mercadom y Mercado Nuevo)"
    previous = {market: UnitSighting(Decimal(100), BaseUnit.POUND, object())}
    per_unit = ParsedUnit(BaseUnit.UNIT, Decimal(1))

    unit = unit_in_effect(per_unit, [(market, Decimal(100))], previous)

    assert unit is per_unit


def test_an_unreadable_label_whose_price_holds_recovers_the_previous_unit():
    market = "Referencia mayorista (Mercadom y Mercado Nuevo)"
    previous_unit = ParsedUnit(BaseUnit.POUND, Decimal(100))
    previous = {market: UnitSighting(Decimal(2000), BaseUnit.POUND, previous_unit)}

    unit = unit_in_effect(None, [(market, Decimal(2000))], previous)

    assert unit is previous_unit


def test_an_unreadable_label_with_a_price_move_stays_unknown():
    market = "Referencia mayorista (Mercadom y Mercado Nuevo)"
    previous = {market: UnitSighting(Decimal(2000), BaseUnit.POUND, object())}

    unit = unit_in_effect(None, [(market, Decimal(2600))], previous)

    assert unit is None


def test_an_unreadable_label_without_a_previous_week_stays_unknown():
    unit = unit_in_effect(None, [("Referencia mayorista (Mercadom y Mercado Nuevo)", Decimal(2000))], {})

    assert unit is None


def test_a_retail_row_recovered_through_the_median_of_its_markets():
    shared = ParsedUnit(BaseUnit.POUND, Decimal(1))
    previous = {
        "Mercado Nuevo": UnitSighting(Decimal(100), BaseUnit.POUND, shared),
        "Villa Consuelo": UnitSighting(Decimal(90), BaseUnit.POUND, shared),
        "Cristo Rey": UnitSighting(Decimal(95), BaseUnit.POUND, shared),
    }
    amounts = [
        ("Mercado Nuevo", Decimal(108)),
        ("Villa Consuelo", Decimal(92)),
        ("Cristo Rey", Decimal(95)),
    ]

    unit = unit_in_effect(None, amounts, previous)

    # Ratios 1.08, 1.02 and 1.00: the median 1.02 holds within the tolerance.
    assert unit is shared


def test_a_retail_row_with_one_unknown_market_cell_still_recovers():
    shared = ParsedUnit(BaseUnit.POUND, Decimal(1))
    previous = {
        "Mercado Nuevo": UnitSighting(Decimal(100), BaseUnit.POUND, shared),
        "Villa Consuelo": UnitSighting(Decimal(90), BaseUnit.POUND, shared),
    }

    unit = unit_in_effect(None, [("Villa Consuelo", Decimal(91))], previous)

    assert unit is shared


def test_the_ledger_keeps_the_last_sighting_per_market_and_price_type():
    ledger = UnitLedger()
    ledger.record("mayorista", "ajo", "Referencia mayorista", Decimal(100), object(), BaseUnit.POUND)
    ledger.record("mayorista", "ajo", "Referencia mayorista", Decimal(120), object(), BaseUnit.POUND)
    ledger.record("minorista", "ajo", "Mercado Nuevo", Decimal(30), object(), BaseUnit.POUND)

    assert ledger.last("mayorista", "ajo")["Referencia mayorista"].amount == Decimal(120)
    assert ledger.last("minorista", "ajo")["Mercado Nuevo"].amount == Decimal(30)
    assert ledger.last("mayorista", "papa") == {}


def test_the_peer_level_needs_two_row_cells_to_judge():
    assert peer_median([Decimal(10)]) is None
    assert peer_median([]) is None


def test_the_peer_level_is_the_row_mid_price():
    assert peer_median([Decimal(10), Decimal(20)]) == Decimal(15)
    assert peer_median([Decimal(10), Decimal(20), Decimal(40)]) == Decimal(20)


def test_the_peer_level_ignores_construction_order():
    values = [Decimal(9), Decimal(40), Decimal(40), Decimal(40), Decimal(300), Decimal(40)]

    assert peer_median(values) == Decimal(40)
