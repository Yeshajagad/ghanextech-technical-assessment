"""Read PDFs and return their text page by page."""

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class IngestError(Exception):
    """Raised with a message that is safe to show to the user."""


def read_pdf(file_name, data):
    """Return a list of (page_number, text) for one PDF.

    Pages with no text are skipped. If nothing can be extracted at all
    (for example a scanned PDF) an IngestError is raised.
    """
    if not file_name.lower().endswith(".pdf"):
        raise IngestError(f"{file_name}: only PDF files are supported.")

    try:
        reader = PdfReader(BytesIO(data))
    except (PdfReadError, ValueError, OSError):
        raise IngestError(f"{file_name}: this file is not a valid PDF or is corrupted.")

    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty password, so try that first.
        try:
            unlocked = reader.decrypt("")
        except Exception:
            unlocked = 0
        if not unlocked:
            raise IngestError(f"{file_name}: the PDF is password-protected.")

    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))

    if not pages:
        raise IngestError(
            f"{file_name}: no text found. It may be a scanned PDF (images only), "
            "which this app cannot read because it has no OCR."
        )
    return pages