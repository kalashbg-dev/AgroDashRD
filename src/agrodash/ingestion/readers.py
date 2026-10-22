"""Read the cells of a price report, whatever format it was exported in."""

import csv
import io
from pathlib import PurePath

import openpyxl
import xlrd

from agrodash.ingestion.models import SourceLayout
from agrodash.ingestion.normalize import fold

SUPPORTED_SUFFIXES = (".csv", ".xlsx", ".xls")


class UnreadableFile(ValueError):
    pass


def read_rows(content: bytes, file_name: str) -> list[list[object]]:
    """Cells of the first sheet, row by row. CSV files use ';' and UTF-8 or Latin-1."""
    suffix = PurePath(file_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnreadableFile(f"Unsupported file type '{suffix}'; use .csv, .xlsx or .xls")
    try:
        if suffix == ".csv":
            return _read_csv(content)
        if suffix == ".xlsx":
            return _read_xlsx(content)
        return _read_xls(content)
    except Exception as error:
        raise UnreadableFile(f"Could not read {file_name}: {error}") from error


def _read_csv(content: bytes) -> list[list[object]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        # The official exports are Windows Latin-1.
        text = content.decode("cp1252", errors="replace")
    return [list(row) for row in csv.reader(io.StringIO(text), delimiter=";")]


def _read_xlsx(content: bytes) -> list[list[object]]:
    # openpyxl parses XML with defusedxml when it is installed (it is a dependency).
    workbook = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    try:
        return [list(row) for row in workbook.worksheets[0].iter_rows(values_only=True)]
    finally:
        workbook.close()


def _read_xls(content: bytes) -> list[list[object]]:
    sheet = xlrd.open_workbook(file_contents=content).sheet_by_index(0)
    return [sheet.row_values(index) for index in range(sheet.nrows)]


def detect_layout(rows: list[list[object]]) -> SourceLayout:
    """Retail reports start with a header naming the markets; wholesale ones have none."""
    first = [fold(cell) for cell in rows[0]] if rows else []
    if first and first[0] == "productos" and any("mercado" in cell for cell in first):
        return SourceLayout.RETAIL
    return SourceLayout.WHOLESALE


def is_blank(row: list[object]) -> bool:
    return all(cell is None or str(cell).strip() == "" for cell in row)
