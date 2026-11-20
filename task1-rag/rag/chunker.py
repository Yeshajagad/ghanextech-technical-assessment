"""Split page text into overlapping chunks.

The splitter is recursive: it first tries to cut on paragraph breaks, then
line breaks, then sentence ends, then single spaces. A bigger separator is
only abandoned when a piece is still longer than chunk_size.
"""

import re

CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 150  # characters

# Tried in this order. Each one is a regex.
SEPARATORS = [r"\n\s*\n", r"\n", r"(?<=[.!?])\s+", r"\s+"]


def _break_up(text, size, separators):
    """Cut text into pieces that are each at most `size` characters."""
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i:i + size] for i in range(0, len(text), size)]

    parts = re.split(separators[0], text)
    pieces = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= size:
            pieces.append(part)
        else:
            pieces.extend(_break_up(part, size, separators[1:]))
    return pieces


def _merge(pieces, size, overlap):
    """Join small pieces into chunks, repeating a tail of each chunk in the next."""
    chunks = []
    current = []
    current_len = 0

    for piece in pieces:
        extra = len(piece) + (1 if current else 0)
        if current and current_len + extra > size:
            chunks.append(" ".join(current))
            # Keep only the last few pieces, up to `overlap` characters.
            while current and (current_len > overlap or current_len + len(piece) + 1 > size):
                dropped = current.pop(0)
                current_len -= len(dropped) + (1 if current else 0)
        current_len += len(piece) + (1 if current else 0)
        current.append(piece)

    if current:
        chunks.append(" ".join(current))
    return chunks


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    pieces = _break_up(text, chunk_size, SEPARATORS)
    return _merge(pieces, chunk_size, overlap)


def chunk_pages(pages, source):
    """Turn [(page, text)] into a list of chunk dicts that remember where they came from."""
    chunks = []
    for page_number, text in pages:
        for piece in split_text(text):
            chunks.append(
                {
                    "text": piece,
                    "source": source,
                    "page": page_number,
                }
            )
    return chunks