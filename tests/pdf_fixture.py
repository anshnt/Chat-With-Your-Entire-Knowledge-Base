"""Builds a minimal, valid, text-extractable PDF for tests.

Hand-writing the PDF avoids a heavyweight generation dependency and — more
usefully — keeps the fixture honest: the bytes are a real PDF with a real
content stream, so :class:`~kb.ingest.pdf.PDFConnector` and pypdf do their actual
work rather than being handed something pre-digested.
"""

from __future__ import annotations


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str], *, leading: int = 16, font_size: int = 12) -> bytes:
    body = ["BT", f"/F1 {font_size} Tf", f"{leading} TL", "72 720 Td"]
    for index, line in enumerate(lines):
        body.append(f"({_escape(line)}) Tj" if index == 0 else f"T* ({_escape(line)}) Tj")
    body.append("ET")
    return "\n".join(body).encode("latin-1")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Assemble a PDF from ``pages``, each a list of text lines.

    Object layout: 1 = Catalog, 2 = Pages, 3 = Font, then one Page and one
    content stream per page.
    """
    objects: dict[int, bytes] = {}
    page_ids: list[int] = []
    next_id = 4

    for lines in pages:
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_ids.append(page_id)
        stream = _content_stream(lines)
        objects[page_id] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> " + f"/Contents {content_id} 0 R >>".encode()
        )
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode() + objects[obj_id] + b"\nendobj\n"

    xref_offset = len(out)
    count = max(objects) + 1
    out += f"xref\n0 {count}\n".encode()
    out += b"0000000000 65535 f \n"
    for obj_id in range(1, count):
        offset = offsets.get(obj_id, 0)
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(out)
