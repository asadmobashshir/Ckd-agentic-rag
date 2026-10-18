"""Structure-aware chunking for medical documents.

Rather than cutting every ``N`` characters, the chunker:

1. Segments the document by Markdown headings and tracks the heading path so a
   retrieved chunk can be traced back to its section.
2. Splits each section into atomic blocks (paragraph, list, table, fenced code,
   dosage/reference block).
3. Packs blocks into chunks up to ``chunk_size``, never splitting a table, list
   or dosage block unless that single block is itself larger than the budget.
4. Adds sentence-boundary overlap between consecutive chunks for context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

BlockKind = Literal["paragraph", "list", "table", "code", "dosage", "heading"]

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
FENCE_RE = re.compile(r"^\s*```")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

# Blocks that read like dosage / posology / administration reference material and
# should be kept intact so a number is never separated from its qualifier.
DOSAGE_HINT_RE = re.compile(
    r"\b(dose|doses|dosage|dosing|posology|administration|mg\b|mcg\b|µg|ml\b|"
    r"units?/kg|mg/kg|titrat|maximum daily|max\.? daily|bid\b|tid\b|qid\b|"
    r"once daily|twice daily)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Block:
    """An atomic, non-splittable unit of document text."""

    text: str
    kind: BlockKind
    heading_path: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class Chunk:
    """A retrievable chunk with structural provenance."""

    chunk_id: str
    document_id: str
    text: str
    index: int
    section: str | None = None
    heading_path: tuple[str, ...] = ()
    block_kinds: tuple[str, ...] = ()
    char_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _classify(lines: list[str]) -> BlockKind:
    if any(TABLE_RE.match(line) for line in lines):
        return "table"
    if any(LIST_RE.match(line) for line in lines):
        return "list"
    text = "\n".join(lines)
    if DOSAGE_HINT_RE.search(text):
        return "dosage"
    return "paragraph"


def split_blocks(text: str) -> list[Block]:
    """Split a document into heading-aware atomic blocks."""
    blocks: list[Block] = []
    heading_path: list[str] = []
    buffer: list[str] = []
    in_fence = False

    def flush(kind: BlockKind | None = None) -> None:
        nonlocal buffer
        if not buffer:
            return
        content = "\n".join(buffer).strip()
        if content:
            blocks.append(
                Block(content, kind or _classify(buffer), tuple(heading_path))
            )
        buffer = []

    for line in text.splitlines():
        if FENCE_RE.match(line):
            if in_fence:
                buffer.append(line)
                flush("code")
                in_fence = False
            else:
                flush()
                buffer.append(line)
                in_fence = True
            continue

        if in_fence:
            buffer.append(line)
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush()
            level, title = len(heading.group(1)), heading.group(2)
            del heading_path[level - 1 :]
            heading_path.append(title)
            blocks.append(Block(line.strip(), "heading", tuple(heading_path)))
            continue

        if not line.strip():
            flush()
            continue

        # A table or list must not be merged with an adjacent paragraph.
        if buffer:
            current_is_structured = any(
                TABLE_RE.match(b) or LIST_RE.match(b) for b in buffer
            )
            line_is_structured = bool(TABLE_RE.match(line) or LIST_RE.match(line))
            if current_is_structured != line_is_structured:
                flush()
        buffer.append(line)

    flush()
    return blocks


def _overlap_tail(text: str, overlap: int) -> str:
    """Return up to ``overlap`` trailing characters, cut on a sentence boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    sentences = SENTENCE_SPLIT_RE.split(tail)
    return (sentences[-1] if len(sentences) > 1 else tail).strip()


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Last-resort splitter for a single oversized block."""
    pieces, start = [], 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        pieces.append(text[start : start + chunk_size].strip())
        start += step
    return [p for p in pieces if p]


def chunk_blocks(
    blocks: Iterable[Block],
    document_id: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Pack blocks into overlapping chunks that respect structural boundaries."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    current: list[Block] = []
    current_len = 0
    pending_heading: Block | None = None

    def emit() -> None:
        nonlocal current, current_len
        if not current:
            return
        body = "\n\n".join(b.text for b in current).strip()
        if not body:
            current, current_len = [], 0
            return
        index = len(chunks)
        # A chunk may span several sections; record every one so a retrieved
        # passage can always be traced back to the heading(s) it came from.
        distinct_paths: list[tuple[str, ...]] = []
        for block in current:
            if block.heading_path and block.heading_path not in distinct_paths:
                distinct_paths.append(block.heading_path)
        heading_path = distinct_paths[0] if distinct_paths else ()
        section = "; ".join(" > ".join(path) for path in distinct_paths) or None
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}::c{index:04d}",
                document_id=document_id,
                text=body,
                index=index,
                section=section,
                heading_path=heading_path,
                block_kinds=tuple(dict.fromkeys(b.kind for b in current)),
                char_count=len(body),
            )
        )
        current, current_len = [], 0

    for block in blocks:
        # A heading alone should lead the next chunk, not trail the previous one.
        if block.kind == "heading":
            if current_len >= chunk_size * 0.6:
                emit()
            pending_heading = block
            current.append(block)
            current_len += block.size + 2
            continue

        if block.size > chunk_size:
            emit()
            prefix = f"{pending_heading.text}\n\n" if pending_heading else ""
            for i, piece in enumerate(_hard_split(block.text, chunk_size, chunk_overlap)):
                index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}::c{index:04d}",
                        document_id=document_id,
                        text=(prefix if i == 0 else "") + piece,
                        index=index,
                        section=" > ".join(block.heading_path) or None,
                        heading_path=block.heading_path,
                        block_kinds=(block.kind, "split"),
                        char_count=len(piece),
                    )
                )
            continue

        if current_len + block.size > chunk_size and current:
            tail = "\n\n".join(b.text for b in current)
            emit()
            carry = _overlap_tail(tail, chunk_overlap)
            if carry:
                current.append(Block(carry, "paragraph", block.heading_path))
                current_len += len(carry) + 2

        current.append(block)
        current_len += block.size + 2

    emit()
    return chunks


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Convenience wrapper: document text -> structural chunks."""
    return chunk_blocks(split_blocks(text), document_id, chunk_size, chunk_overlap)
