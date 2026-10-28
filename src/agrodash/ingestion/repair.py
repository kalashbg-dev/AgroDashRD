"""Repair what the price reports get wrong row by row, before any row is parsed.

Every function here works on plain rows, so the repairs can be tested without a
database or a file.
"""

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from statistics import median

from agrodash.ingestion.normalize import (
    InvalidValue,
    clean_name,
    fold,
    name_key,
    parse_amount,
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
# Market cells the copy must share with its source before the column counts as pasted.
MIN_COPIED_CELLS = 10
# Same-name priced rows must repeat the neighboring week's row ABOVE this often, and
# that week's own row this rarely, before the price column reads as slid (G1).
MIN_SLID_RATE = 0.40
MAX_ALIGNED_RATE = 0.25
# The wholesale report's price column.
WHOLESALE_PRICE_COLUMN = 4


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


def duplicate_block_rows(
    rows: list[tuple[int, list[object]]], date_columns: tuple[int, ...]
) -> set[int]:
    """Row numbers of week blocks that are a copy pasted under a wrong label (G4).

    - A block whose label already named an earlier block, with another week between
      them, is a mislabelled copy (wholesale abril 2023 is listed twice: the second
      Primera/Segunda are mayo's, the second Cuarta repeats Cuarta). Its true week
      cannot be told, so the whole block goes to quarantine.
    - A block identical to an earlier kept block under another label is a copy too
      (retail Primera noviembre 2022 = Segunda junio 2022). Quarantined blocks are not
      compared, so the real mayo 2023 Primera survives its mislabelled copy.
    Blocks under MIN_AVERAGE_BLOCK_ROWS are stray rows, never a week.
    """
    duplicates: set[int] = set()
    labels: set[tuple] = set()
    fingerprints: set[tuple] = set()
    previous = None
    for label, block in _blocks_with_numbers(rows, date_columns):
        if len(block) < MIN_AVERAGE_BLOCK_ROWS:
            continue
        fingerprint = tuple(sorted(
            tuple(str(cell) for i, cell in enumerate(row) if i not in date_columns)
            for _, row in block
        ))
        if label == previous:
            fingerprints.add(fingerprint)
            continue
        previous = label
        if label in labels or fingerprint in fingerprints:
            duplicates.update(row_number for row_number, _ in block)
            continue
        labels.add(label)
        fingerprints.add(fingerprint)
    return duplicates


def repeated_week_rows(
    rows: list[tuple[int, list[object]]], date_columns: tuple[int, ...]
) -> set[int]:
    """Row numbers of weeks that are near-copies of the previous week (G3).

    A week whose prices are ~99% identical to the previous week's is a copy
    pasted by the analyst. Flagged as "repeated price" for exclusion from
    forecasting and volatility.
    """
    # Group rows into blocks by date label.
    blocks: list[tuple[tuple, list[tuple[int, list[object]]]]] = []
    current: list[tuple[int, list[object]]] = []
    prev_label = None
    for row_number, row in rows:
        if len(row) <= max(date_columns):
            current.append((row_number, row))
            continue
        lbl = _date_label(row, date_columns)
        if prev_label is not None and lbl != prev_label:
            blocks.append((prev_label, current))
            current = []
        current.append((row_number, row))
        prev_label = lbl
    if current:
        blocks.append((prev_label, current))

    repeated: set[int] = set()
    for i in range(1, len(blocks)):
        prev_label, prev_rows = blocks[i - 1]
        curr_label, curr_rows = blocks[i]
        # Only compare blocks in the same month and year.
        if prev_label[1:] != curr_label[1:]:
            continue
        if len(curr_rows) < MIN_AVERAGE_BLOCK_ROWS:
            continue
        # Build sets of row content (excluding date columns).
        def _content(row):
            return tuple(str(cell) for j, cell in enumerate(row) if j not in date_columns)

        prev_set = {_content(row) for _, row in prev_rows if len(row) > max(date_columns)}
        curr_list = [(_rn, _content(row)) for _rn, row in curr_rows if len(row) > max(date_columns)]

        if not curr_list or not prev_set:
            continue
        matched = sum(1 for _, c in curr_list if c in prev_set)
        if matched / len(curr_list) >= 0.95:
            for row_number, _ in curr_rows:
                repeated.add(row_number)
    return repeated


def _blocks_with_numbers(
    rows: list[tuple[int, list[object]]], date_columns: tuple[int, ...]
) -> list[tuple[tuple, list[tuple[int, list[object]]]]]:
    """Consecutive rows grouped by their date label, as [(label, [(row_number, row)...])].

    Runs of the same label stay whole even when one stray row is sandwiched between
    them (a paste artifact corrupting a single date cell); the stray joins the
    surrounding period instead of breaking it. The stray keeps its original label,
    so a corrupt date still fails its own date repair after quarantine.
    """
    raw: list[tuple[tuple, list[tuple[int, list[object]]]]] = []
    rows_of_run: list[tuple[int, list[object]]] = []
    prev_label = None
    for row_number, row in rows:
        if len(row) <= max(date_columns):
            rows_of_run.append((row_number, row))
            continue
        label = _date_label(row, date_columns)
        if prev_label is not None and label != prev_label:
            raw.append((prev_label, rows_of_run))
            rows_of_run = []
        rows_of_run.append((row_number, row))
        prev_label = label
    if rows_of_run:
        raw.append((prev_label, rows_of_run))

    # A lone stray row whose label got corrupted while the rows around it share one
    # label joins that label's period instead of splitting it in two.
    blocks: list[tuple[tuple, list[tuple[int, list[object]]]]] = []
    index = 0
    while index < len(raw):
        label, group = raw[index]
        following = raw[index + 1] if index + 1 < len(raw) else None
        if (
            blocks
            and len(group) <= 1
            and following is not None
            and blocks[-1][0] == following[0]
        ):
            blocks[-1][1].extend(group)
            blocks[-1][1].extend(following[1])
            index += 2
            continue
        blocks.append((label, list(group)))
        index += 1
    return blocks


def _comparable(cell: object) -> Decimal | None:
    """The cell's price as a comparable value, or None when it holds no price."""
    try:
        return parse_amount(cell)
    except InvalidValue:
        return None


def _column_answers(
    block_rows: list[tuple[int, list[object]]], column: int
) -> list[Decimal | None]:
    return [
        _comparable(row[column]) if column < len(row) else None
        for _, row in block_rows
    ]


def _is_copy(left: list[Decimal | None], right: list[Decimal | None]) -> bool:
    """Whether a market column is a copy of another across the block's rows.

    Every cell where both columns hold a price must agree, and there must be enough
    such cells: two markets never answer the same price on a whole week (G6).
    """
    pairs = 0
    for left_cell, right_cell in zip(left, right):
        if left_cell is None or right_cell is None:
            continue
        pairs += 1
        if left_cell != right_cell:
            return False
    return pairs >= MIN_COPIED_CELLS


def nullify_copied_market_columns(
    rows: list[tuple[int, list[object]]],
    date_columns: tuple[int, ...],
    market_columns: range,
) -> set[int]:
    """Row numbers whose market column is voided because it is a copy (G6).

    A pasted column answers exactly what another market answered, in the same week
    (Supermercado = Mercadom, sep-2024 S2) or one week earlier (Cristo Rey = the
    previous Mercadom, dic-2024 S5; the previous Villa Consuelo, ene-2025 S1). Its
    cells become missing for that week so the rest of the row loads normally. In a
    same-week tie the later column is treated as the copy; a same-column repeat
    across weeks is a frozen price or a G3 copy, never a pasted column.
    """
    # Split into week blocks by date label, keeping each row's number.
    blocks = _blocks_with_numbers(rows, date_columns)

    voided: set[int] = set()
    previous_nullified: set[int] = set()
    for index, (_, block) in enumerate(blocks):
        if len(block) < MIN_AVERAGE_BLOCK_ROWS:
            previous_nullified = set()
            continue
        answers = {column: _column_answers(block, column) for column in market_columns}
        nullified: set[int] = set()
        # Same week: among identical columns only the later one is the copy.
        for position, left in enumerate(market_columns):
            if left in nullified:
                continue
            for right in market_columns[position + 1 :]:
                if right in nullified:
                    continue
                if _is_copy(answers[left], answers[right]):
                    nullified.add(right)
        # Previous week: the copy is in the current block; a same-column repeat
        # across weeks is not a pasted market column.
        for left in market_columns:
            if left in nullified or index == 0:
                continue
            if any(
                left != right and right not in previous_nullified
                and _is_copy(answers[left], previous_answers(right, blocks, index))
                for right in market_columns
            ):
                nullified.add(left)
        previous_nullified = nullified
        for column in nullified:
            for row_number, row in block:
                if column < len(row) and row[column] is not None:
                    row[column] = None
                    voided.add(row_number)
    return voided


def previous_answers(column: int, blocks, index: int) -> list[Decimal | None]:
    """Comparable values of a market column in the block one week earlier."""
    block = blocks[index - 1][1]
    return _column_answers(block, column)


def _name_positions(block: list[tuple[int, list[object]]]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for position, (_, row) in enumerate(block):
        positions.setdefault(str(row[0]).strip().lower(), position)
    return positions


def _column_slid_down(
    block: list[tuple[int, list[object]]],
    neighbor: list[tuple[int, list[object]]],
    column: int,
) -> bool:
    """Whether a block's price column carries the neighboring week's row-above prices."""
    positions = _name_positions(neighbor)
    pairs = shifted = aligned = 0
    for _, row in block:
        price = _comparable(row[column])
        if price is None:
            continue
        position = positions.get(str(row[0]).strip().lower())
        if position is None:
            continue
        pairs += 1
        if (
            position >= 1
            and _comparable(neighbor[position - 1][1][column]) == price
        ):
            shifted += 1
        if _comparable(neighbor[position][1][column]) == price:
            aligned += 1
    return (
        pairs >= MIN_COPIED_CELLS
        and shifted / pairs >= MIN_SLID_RATE
        and aligned / pairs <= MAX_ALIGNED_RATE
    )


def shifted_price_blocks(rows, date_columns) -> set[int]:
    """Row numbers of wholesale weeks whose price column slid down one row (G1).

    A week fires when the prices of its same-named products repeat a neighboring
    week's row ABOVE far more often than that week's own row: every product is
    carrying the price of the one below (Papa at 8.000, Yautía at 48 in the audit).
    The whole week goes to quarantine; realigning the column is out of scope.
    """
    blocks = _blocks_with_numbers(rows, date_columns)
    row_numbers: list[int] = []
    for index, (_, block) in enumerate(blocks):
        if len(block) < MIN_AVERAGE_BLOCK_ROWS:
            continue
        neighbors = []
        if index:
            neighbors.append(blocks[index - 1][1])
        if index + 1 < len(blocks):
            neighbors.append(blocks[index + 1][1])
        if any(
            _column_slid_down(block, neighbor, WHOLESALE_PRICE_COLUMN)
            for neighbor in neighbors
        ):
            row_numbers.extend(row_number for row_number, _ in block)
    return set(row_numbers)


# The audit's one retail case: Los Mina's Ago-2017 S2 column slid with no dominant
# row-level signature (shifted 14% at one row, 30% at the same row, 35% the other
# way), so no threshold singles it out; the user chose to void that column for that
# week instead of leaving an absurd column.
AUDIT_SLID_MARKET_COLUMNS = ((("segunda", "agosto", "2017"), 6),)


def shifted_market_cells(
    rows, date_columns, market_columns
) -> dict[int, set[int]]:
    """Per row number, the retail market columns that slid a row for that week (G1).

    Same detection as `shifted_price_blocks`, one market column at a time, plus the
    audit's explicit Los Mina case. The fired cells should be voided (missing), and
    the rest of the row loads normally.
    """
    blocks = _blocks_with_numbers(rows, date_columns)
    fired: set[tuple[tuple, int]] = set(AUDIT_SLID_MARKET_COLUMNS)
    for index, (label, block) in enumerate(blocks):
        if len(block) < MIN_AVERAGE_BLOCK_ROWS:
            continue
        neighbors = []
        if index:
            neighbors.append(blocks[index - 1][1])
        if index + 1 < len(blocks):
            neighbors.append(blocks[index + 1][1])
        for column in market_columns:
            if any(
                _column_slid_down(block, neighbor, column) for neighbor in neighbors
            ):
                fired.add((label, column))
    cells: dict[int, set[int]] = {}
    for label, block in blocks:
        voided = {column for fired_label, column in fired if fired_label == label}
        if not voided:
            continue
        for row_number, row in block:
            cells.setdefault(row_number, set()).update(
                column for column in voided if column < len(row)
            )
    return cells


DRIFT_TOLERANCE = Decimal("0.08")


@dataclass(frozen=True)
class UnitSighting:
    """One priced cell as the ledger remembers it: amount, base unit, and Unit used."""

    amount: Decimal
    base_unit: object
    unit: object


class UnitLedger:
    """Per (price type, product) memory of each market's last priced row.

    The service feeds it in file order, so a row's reference is the product's previous
    row. That is what unit labels drift against: the audit's M1 asks to trust a stable
    price over a changed label. Only the last sighting per market is kept — recovery
    and drift compare consecutive rows, not whole history.
    """

    def __init__(self) -> None:
        self._markets: dict[tuple[object, str], dict[str, UnitSighting]] = {}

    def record(self, price_type, name_key: str, market: str, amount, base_unit, unit) -> None:
        seen = self._markets.setdefault((price_type, name_key), {})
        seen[market] = UnitSighting(amount, base_unit, unit)

    def last(self, price_type, name_key: str) -> dict[str, UnitSighting]:
        return self._markets.get((price_type, name_key), {})


def unit_in_effect(parsed, amounts, previous: dict[str, UnitSighting]):
    """What a priced row scales in, given its label and the product's previous row.

    parsed is the row's own label (or None when no size is stated), amounts the row's
    (market, amount) pairs, previous the same markets' UnitSighting from the ledger.

    A label the parser cannot read is trusted when the money stays put: within the
    drift tolerance of the previous row the previous unit applies — "Saco/1N/DN/D lb"
    is the same sack the row above priced (audit M1: "ignorar cambios de etiqueta que
    no mueven el precio"). A readable label of another size is only a nominal relabel
    under the same rule (Lechuga moves its box from 18 to 20 lb at the same box price);
    when the price moves the new label is real and prices in it. The amount is never
    rescaled — no convertir por tamaño a ciegas. Several markets decide together by
    the middle ratio: every cell of a row carries the same label change, while a stray
    cell cannot.
    """
    candidates = [
        (market, amount, previous[market])
        for market, amount in amounts
        if market in previous
        and previous[market].unit is not None
        and (parsed is None or previous[market].base_unit == parsed.base_unit)
    ]
    if not candidates:
        return parsed
    candidates.sort(key=lambda pair: abs(pair[1] / pair[2].amount - 1))
    middle = candidates[len(candidates) // 2]
    if abs(middle[1] / middle[2].amount - 1) > DRIFT_TOLERANCE:
        return parsed
    return middle[2].unit


def peer_median(values):
    """A row's mid price level, or None when the row has no pair to judge against.

    A spatial reference (audit M3) needs at least two peers: with a single priced
    row cell nothing distinguishes a slipped digit from a real one-off price.
    """
    if len(values) < 2:
        return None
    return median(values)
