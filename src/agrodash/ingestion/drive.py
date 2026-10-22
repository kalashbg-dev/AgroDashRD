"""Download the official price reports from a public Google Drive folder."""

import re
import urllib.error
from pathlib import PurePath
from urllib.request import Request, urlopen

from agrodash.ingestion.service import MAX_FILE_BYTES

DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={file_id}"
TIMEOUT_SECONDS = 60
_FILE_ID = re.compile(r"[A-Za-z0-9_-]{10,100}")


class DriveError(Exception):
    pass


def download(file_id: str, max_bytes: int = MAX_FILE_BYTES) -> tuple[str, bytes]:
    """File name and content of a file shared with "Anyone with the link"."""
    if not _FILE_ID.fullmatch(file_id):
        raise DriveError(f"Invalid Drive file id {file_id!r}")
    # Fixed https host; the id is validated above, so the URL cannot be redirected.
    url = DOWNLOAD_URL.format(file_id=file_id)
    request = Request(url, headers={"User-Agent": "agrodash"})  # noqa: S310
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            if response.headers.get_content_type() == "text/html":
                raise DriveError(
                    f"Drive returned a web page for {file_id}; check that the file is "
                    "shared with 'Anyone with the link'"
                )
            content = response.read(max_bytes + 1)
            file_name = response.headers.get_filename() or file_id
    except urllib.error.URLError as error:
        raise DriveError(f"Could not download {file_id}: {error.reason}") from error
    if len(content) > max_bytes:
        raise DriveError(f"{file_id} is larger than {max_bytes} bytes")
    # The name only labels the run and picks the reader; never use it as a path.
    return PurePath(file_name.replace("\\", "/")).name, content
