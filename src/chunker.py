"""Document ingestion and preprocessing: split resume / JD text into chunks.

Retrieval works on small, self-contained pieces of text. A bullet point or a
short sentence is a good unit: large enough to carry meaning, small enough that
one embedding represents one idea.
"""
from dataclasses import dataclass
import re

try:
    from .parser import detect_heading
except ImportError:
    from parser import detect_heading

_BULLET = re.compile(r"^\s*(?:[-*\u2022\u25aa\u25e6\u00b7]|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str  # section key from parser.SECTION_HEADINGS, or "other"
    index: int


def split_units(text: str) -> list[tuple[str, str]]:
    """Split text into (section, unit) pairs.

    A bullet starts a new unit. A non-bullet line that starts in lower case and
    follows an unfinished line is treated as a wrapped continuation (common in
    PDF extraction) and is merged into the previous unit.
    """
    units: list[tuple[str, str]] = []
    section = "other"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = detect_heading(line)
        if heading:
            section = heading
            continue
        is_bullet = bool(_BULLET.match(line))
        line = _BULLET.sub("", line)
        if (units and not is_bullet and units[-1][0] == section
                and line[:1].islower() and not units[-1][1].endswith((".", "!", "?"))):
            units[-1] = (section, units[-1][1] + " " + line)
        else:
            units.append((section, line))
    return units


def _split_long(unit: str, max_words: int) -> list[str]:
    if len(unit.split()) <= max_words:
        return [unit]
    pieces, current = [], []
    for sentence in _SENTENCE_END.split(unit):
        words = sentence.split()
        if current and len(current) + len(words) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(text: str, min_words: int = 4, max_words: int = 60) -> list[Chunk]:
    """Split ``text`` into section-tagged chunks.

    Units shorter than ``min_words`` (stray headings, dates, single skills in a
    column) are dropped; units longer than ``max_words`` are split on sentence
    boundaries.
    """
    chunks: list[Chunk] = []
    for section, unit in split_units(text):
        for piece in _split_long(unit, max_words):
            if len(piece.split()) >= min_words:
                chunks.append(Chunk(text=piece, section=section, index=len(chunks)))
    return chunks
