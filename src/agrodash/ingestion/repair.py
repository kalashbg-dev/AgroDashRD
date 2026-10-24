"""Repair what the price reports get wrong row by row, before any row is parsed.

Every function here works on plain rows, so the repairs can be tested without a
database or a file.
"""

from collections import Counter
from dataclasses import dataclass

from agrodash.ingestion.normalize import (
    InvalidValue,
    clean_name,
    fold,
    name_key,
    parse_week_date,
    repair_name,
)
from agrodash.ingestion.readers import is_blank

# How close a repeated name must be to count as a second product rather than a duplicate.
ADJACENT_REPEAT_ROWS = 3
# Units a retail price comes in; a wholesale price never does, except a few by the unit.
RETAIL_UNITS = {"lb", "lib", "libra", "libras", "unidad", "und", "ud", "doc", "docena", "litro"}
# Fewer rows than this are a stray row or two, not a pasted week.
MIN_AVERAGE_BLOCK_ROWS = 5


@dataclass(frozen=True)
class RepairedRows:
    # (row number in the file, repaired cells), blank rows left out
    rows: list[tuple[int, list[object]]]
    # Cells exactly as read, by row number, so quarantine shows the source.
    originals: dict[int, list[object]]
    dates_corrected: int


def repair_rows(rows, first_line: int, date_columns: tuple[int, ...]) -> RepairedRows:
    """Repaired copies of the non-blank `rows`, numbered from `first_line`.

    - A row whose date cannot be read ("3333", "|") takes the date of the two rows
      around it when those agree: the reports are written in week blocks. A readable
      date is never changed, and a row with no date at all is left alone: its cells
      were shifted.
    - The same name listed again right below itself (within ADJACENT_REPEAT_ROWS) in
      one week block is two products the report does not tell apart (Berenjena
      Pompadur is 30-43% dearer in its first row); the repeat becomes
      "<name> (fila 2)" so both series are kept. A repeat further away is a block
      that holds two weeks under one label, and stays a duplicate.
    - An "ñ" the export wrote as ";" is restored.
    """
    numbered = [
        (row_number, list(row))
        for row_number, row in enumerate(rows, start=first_line)
        if not is_blank(row)
    ]
    originals = {row_number: list(row) for row_number, row in numbered}
    labels = [_date_label(row, date_columns) for _, row in numbered]
    repaired, corrected = [], 0
    block, listed, last_seen = None, Counter(), {}
    for index, (row_number, row) in enumerate(numbered):
        label = labels[index]
        before = labels[index - 1] if index else None
        after = labels[index + 1] if index + 1 < len(labels) else None
        if (
            any(label)
            and len(row) > max(date_columns)
            and before is not None
            and before == after
            and label != before
            and not _readable_date(row, date_columns)
        ):
            source = numbered[index - 1][1]
            for column in date_columns:
                row[column] = source[column]
            label = before
            corrected += 1
        if label != block:
            block, listed, last_seen = label, Counter(), {}
        name = repair_name(row[0] if row[0] is not None else "")
        key = name_key(name)
        if key in last_seen and index - last_seen[key] <= ADJACENT_REPEAT_ROWS:
            listed[key] += 1
            name = f"{clean_name(name)} (fila {listed[key] + 1})"
        else:
            listed[key] = 0
        last_seen[key] = index
        row[0] = name
        repaired.append((row_number, row))
    return RepairedRows(rows=repaired, originals=originals, dates_corrected=corrected)


def retail_average_rows(
    rows: list[tuple[int, list[object]]], date_columns: tuple[int, ...], unit_column: int
) -> set[int]:
    """Row numbers of the wholesale week blocks that hold retail averages instead.

    A wholesale price comes in a sack, crate, hundred or quintal; a week where most rows
    are priced per pound, unit, dozen or liter is the retail markets' average pasted into
    the wholesale report (Quinta Julio 2025). A few products sold by the unit in a sound
    week (Repollo) do not make it one.
    """
    flagged: set[int] = set()
    block: list[tuple[int, bool]] = []
    label = None
    for row_number, row in [*rows, (None, None)]:
        row_label = None if row is None else _date_label(row, date_columns)
        if row_label != label:
            retail = sum(is_retail for _, is_retail in block)
            if len(block) >= MIN_AVERAGE_BLOCK_ROWS and retail * 2 > len(block):
                flagged.update(number for number, _ in block)
            block, label = [], row_label
        if row is not None:
            unit = fold(row[unit_column] if unit_column < len(row) else "").rstrip(".")
            block.append((row_number, unit in RETAIL_UNITS))
    return flagged


def _readable_date(row: list[object], date_columns) -> bool:
    try:
        parse_week_date(*(row[column] if column < len(row) else None for column in date_columns))
    except InvalidValue:
        return False
    return True


def _date_label(row: list[object], date_columns) -> tuple[str, ...]:
    return tuple(fold(row[column]) if column < len(row) else "" for column in date_columns)


# Wholesale measure words: a retail row carrying one of these (or N/D) is a unit column
# pasted from the wholesale report (G5, Caso 1).
WHOLESALE_UNIT_WORDS = {"saco", "quintal", "ciento", "caja", "bulto"}


def retail_weeks_with_wholesale_units(
    rows: list[tuple[int, list[object]]], date_columns: tuple[int, ...], unit_column: int
) -> set[int]:
    """Row numbers of retail week blocks whose unit column holds wholesale units.

    In those weeks the unit column was pasted from the wholesale report (Saco, Quintal,
    Ciento...) while the prices are retail, so every unit is wrong. The ETL recovers each
    product's unit from the previous retail week it appeared in (G5, Caso 1).
    """
    flagged: set[int] = set()
    block: list[tuple[int, bool]] = []
    label = None
    for row_number, row in [*rows, (None, None)]:
        row_label = None if row is None else _date_label(row, date_columns)
        if row_label != label:
            if len(block) >= MIN_AVERAGE_BLOCK_ROWS and sum(flag for _, flag in block) * 2 > len(block):
                flagged.update(number for number, _ in block)
            block, label = [], row_label
        if row is not None:
            unit = fold(row[unit_column] if unit_column < len(row) else "")
            is_wholesale = any(word in unit for word in WHOLESALE_UNIT_WORDS) or "n/d" in unit
            block.append((row_number, is_wholesale))
    return flagged
