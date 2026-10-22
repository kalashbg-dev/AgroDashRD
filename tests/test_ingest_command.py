import io
import urllib.error
from email.message import Message
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from agrodash import cli
from agrodash.ingestion import drive
from agrodash.ingestion.drive import DriveError, download
from agrodash.ingestion.models import IngestionRun, IngestionTrigger
from agrodash.prices.models import Price

FIXTURES = Path(__file__).parent / "fixtures"
FILE_ID = "17K3F_hI0OV_eWwNFo0WESPsxrnhuQKKK"


class FakeResponse(io.BytesIO):
    def __init__(self, content: bytes, content_type: str, file_name: str | None = None):
        super().__init__(content)
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        if file_name:
            self.headers["Content-Disposition"] = f'attachment; filename="{file_name}"'


def fake_urlopen(response, requested=None):
    def urlopen(request, timeout):
        if requested is not None:
            requested.append((request.full_url, timeout))
        if isinstance(response, Exception):
            raise response
        return response

    return urlopen


def test_ingest_command_loads_files_and_reports_counts(cli_database, capsys):
    path = str(FIXTURES / "mayorista.csv")

    assert cli.main(["ingest", path]) == 0
    assert cli.main(["ingest", path]) == 0

    out = capsys.readouterr().out.splitlines()
    assert out == [
        "mayorista.csv: loaded 5 prices, quarantined 6 rows, 1 missing values (run 1)",
        "mayorista.csv: already loaded (run 1)",
    ]
    with Session(cli_database) as session:
        assert session.query(Price).count() == 5


def test_ingest_command_reports_failures_and_keeps_going(cli_database, tmp_path, capsys):
    bad = tmp_path / "precios.pdf"
    bad.write_bytes(b"%PDF")

    exit_code = cli.main(
        ["ingest", str(bad), str(tmp_path / "missing.csv"), str(FIXTURES / "minorista.csv")]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error: precios.pdf: Unsupported file type '.pdf'" in captured.err
    assert "error: missing.csv: file not found" in captured.err
    assert "minorista.csv: loaded 13 prices" in captured.out


def test_ingest_command_downloads_the_configured_drive_files(cli_database, monkeypatch, capsys):
    monkeypatch.setenv("AGRODASH_DRIVE_FILE_IDS", f" {FILE_ID} ")
    content = (FIXTURES / "minorista.xls").read_bytes()
    response = FakeResponse(content, "application/vnd.ms-excel", "Minorista 2026.xls")
    monkeypatch.setattr(drive, "urlopen", fake_urlopen(response))

    assert cli.main(["ingest", "--drive"]) == 0

    assert "Minorista 2026.xls: loaded 13 prices" in capsys.readouterr().out
    with Session(cli_database) as session:
        assert session.query(IngestionRun).one().trigger is IngestionTrigger.DRIVE


def test_ingest_command_needs_files_or_drive(cli_database, capsys):
    with pytest.raises(SystemExit):
        cli.main(["ingest"])

    assert "give files to load or --drive" in capsys.readouterr().err


def test_download_uses_the_public_download_url_and_file_name(monkeypatch):
    requested = []
    response = FakeResponse(b"data", "application/octet-stream", "../../Mayorista.xlsx")
    monkeypatch.setattr(drive, "urlopen", fake_urlopen(response, requested))

    assert download(FILE_ID) == ("Mayorista.xlsx", b"data")
    assert requested == [(f"https://drive.google.com/uc?export=download&id={FILE_ID}", 60)]


@pytest.mark.parametrize("file_id", ["", "abc", "x/../../y", "id?x=1&y=2", "a" * 200])
def test_download_refuses_malformed_ids(file_id):
    with pytest.raises(DriveError, match="Invalid Drive file id"):
        download(file_id)


def test_download_explains_a_file_that_is_not_public(monkeypatch):
    response = FakeResponse(b"<html>Sign in</html>", "text/html")
    monkeypatch.setattr(drive, "urlopen", fake_urlopen(response))

    with pytest.raises(DriveError, match="shared with 'Anyone with the link'"):
        download(FILE_ID)


def test_download_reports_network_errors(monkeypatch):
    monkeypatch.setattr(drive, "urlopen", fake_urlopen(urllib.error.URLError("timed out")))

    with pytest.raises(DriveError, match=f"Could not download {FILE_ID}: timed out"):
        download(FILE_ID)


def test_download_refuses_files_over_the_limit(monkeypatch):
    monkeypatch.setattr(drive, "urlopen", fake_urlopen(FakeResponse(b"12345", "text/csv", "a.csv")))

    with pytest.raises(DriveError, match="larger than"):
        download(FILE_ID, max_bytes=4)


def test_ingest_help_names_the_sources(capsys):
    with pytest.raises(SystemExit):
        cli.main(["ingest", "--help"])

    assert "Load official price reports (.csv, .xlsx, .xls) from files or Google Drive" in (
        capsys.readouterr().out
    )


def test_ingest_command_reports_unreadable_paths_and_keeps_going(cli_database, tmp_path, capsys):
    exit_code = cli.main(["ingest", str(tmp_path), str(FIXTURES / "mayorista.csv")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert f"error: {tmp_path.name}: cannot read file" in captured.err
    assert "mayorista.csv: loaded 5 prices" in captured.out


def test_ingest_drive_needs_configured_ids(cli_database, monkeypatch, capsys):
    monkeypatch.setenv("AGRODASH_DRIVE_FILE_IDS", " , ")

    assert cli.main(["ingest", "--drive"]) == 1

    assert "error: AGRODASH_DRIVE_FILE_IDS is empty" in capsys.readouterr().err


def test_download_reports_timeouts_while_reading(monkeypatch):
    class SlowResponse(FakeResponse):
        def read(self, *_):
            raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(drive, "urlopen", fake_urlopen(SlowResponse(b"", "text/csv", "a.csv")))

    with pytest.raises(
        DriveError, match=f"Could not download {FILE_ID}: The read operation timed out"
    ):
        download(FILE_ID)
