from agrodash.ingestion.repair import repair_rows

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
