from agrodash.ingestion.repair import nullify_copied_market_columns, repair_rows

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
