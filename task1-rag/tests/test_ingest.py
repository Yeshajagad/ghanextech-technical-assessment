from io import BytesIO

import pytest
from pypdf import PdfWriter

from rag.ingest import IngestError, read_pdf


def make_text_pdf(text):
    """Build a tiny one-page PDF by hand, with a correct cross-reference table."""
    stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET"
    objects = [
        "<</Type/Catalog/Pages 2 0 R>>",
        "<</Type/Pages/Kids[3 0 R]/Count 1>>",
        "<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        "/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        f"<</Length {len(stream)}>>\nstream\n{stream}\nendstream",
        "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    body = "%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{number} 0 obj\n{obj}\nendobj\n"

    xref_start = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for offset in offsets:
        body += f"{offset:010d} 00000 n \n"
    body += f"trailer\n<</Root 1 0 R/Size {len(objects) + 1}>>\nstartxref\n{xref_start}\n%%EOF"
    return body.encode("latin-1")


def make_blank_pdf(password=None):
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    if password:
        writer.encrypt(password, algorithm="RC4-128")
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_reads_text_pdf():
    pages = read_pdf("hello.pdf", make_text_pdf("Hello assessment world"))
    assert pages[0][0] == 1
    assert "Hello assessment world" in pages[0][1]


def test_rejects_non_pdf_name():
    with pytest.raises(IngestError, match="only PDF"):
        read_pdf("notes.txt", b"hello")


def test_rejects_corrupt_pdf():
    with pytest.raises(IngestError, match="not a valid PDF"):
        read_pdf("bad.pdf", b"this is not a pdf at all")


def test_rejects_scanned_or_empty_pdf():
    with pytest.raises(IngestError, match="no text found"):
        read_pdf("scan.pdf", make_blank_pdf())


def test_rejects_password_protected_pdf():
    with pytest.raises(IngestError, match="password-protected"):
        read_pdf("locked.pdf", make_blank_pdf(password="secret"))