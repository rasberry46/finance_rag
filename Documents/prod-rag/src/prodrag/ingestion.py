"""
ingestion.py
============
STEP 1 of the pipeline: turn PDFs into retrievable Chunks.

Uses pdfplumber for REAL table extraction (it detects ruled/space tables and
returns cell grids), then applies table-aware chunking so each table becomes ONE
atomic Markdown chunk with per-row children — headers never split from values.
Prose is windowed with overlap.

This is the offline/local ingestion path. `textract_ingest()` shows the AWS
Textract swap for scanned docs (same Chunk output).

Chunk carries metadata used later by confidence scoring:
    source, page, chunk_type, is_table, doc_date, source_type
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber

from .config import CONFIG


@dataclass
class Chunk:
    text: str
    chunk_type: str          # "table" | "table_row" | "prose"
    doc_id: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)


def _rows_to_markdown(rows: list[list[str]]) -> str:
    rows = [[(c if c is not None else "") for c in r] for r in rows]
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    header, body = padded[0], padded[1:]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * width) + " |\n"
    for r in body:
        md += "| " + " | ".join(str(x) for x in r) + " |\n"
    return md


def _window_prose(text: str, doc_id: str, page: int, doc_date: datetime,
                  source_type: str) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    size, overlap = CONFIG.prose_chunk_words, CONFIG.prose_overlap_words
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if not window:
            break
        chunks.append(Chunk(
            text=" ".join(window), chunk_type="prose", doc_id=doc_id,
            metadata={"source": doc_id, "page": page, "doc_date": doc_date,
                      "source_type": source_type},
        ))
        if start + size >= len(words):
            break
    return chunks


def ingest_pdf(path: str | Path, source_type: str = "audited_filing",
               doc_date: datetime | None = None) -> list[Chunk]:
    path = Path(path)
    doc_id = path.stem
    doc_date = doc_date or datetime.now(timezone.utc)
    chunks: list[Chunk] = []

    with pdfplumber.open(str(path)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []

            # Tables → atomic chunks + row children
            for tbl in tables:
                clean = [[(c or "").strip() for c in row] for row in tbl if any(row)]
                if len(clean) < 2:
                    continue
                parent_id = str(uuid.uuid4())[:8]
                header = clean[0]
                chunks.append(Chunk(
                    text=_rows_to_markdown(clean), chunk_type="table", doc_id=doc_id,
                    chunk_id=parent_id,
                    metadata={"source": doc_id, "page": pno, "is_table": True,
                              "n_rows": len(clean) - 1, "columns": header,
                              "doc_date": doc_date, "source_type": source_type},
                ))
                for row in clean[1:]:
                    verbalized = f"In table from {doc_id} (p{pno}): " + "; ".join(
                        f"{header[c]} = {row[c]}"
                        for c in range(min(len(header), len(row))) if header[c]
                    )
                    chunks.append(Chunk(
                        text=verbalized, chunk_type="table_row", doc_id=doc_id,
                        parent_id=parent_id,
                        metadata={"source": doc_id, "page": pno, "row_label": row[0],
                                  "doc_date": doc_date, "source_type": source_type},
                    ))

            # Prose → remove table text to avoid duplication, then window
            page_text = page.extract_text() or ""
            # crude but effective: drop lines that are mostly table cells
            table_cells = set()
            for tbl in tables:
                for row in tbl:
                    for c in row:
                        if c:
                            table_cells.add(c.strip())
            prose_lines = [
                ln for ln in page_text.splitlines()
                if ln.strip() and ln.strip() not in table_cells
                and not _is_mostly_table_row(ln, table_cells)
            ]
            prose = " ".join(prose_lines)
            chunks.extend(_window_prose(prose, doc_id, pno, doc_date, source_type))

    return chunks


def _is_mostly_table_row(line: str, table_cells: set[str]) -> bool:
    parts = line.split()
    if not parts:
        return False
    hits = sum(1 for p in parts if p in table_cells)
    return hits / len(parts) > 0.5


def ingest_dir(directory: str | Path = None) -> list[Chunk]:
    directory = Path(directory) if directory else CONFIG.sample_docs_dir
    all_chunks = []
    for pdf in sorted(directory.glob("*.pdf")):
        all_chunks.extend(ingest_pdf(pdf))
    return all_chunks


if __name__ == "__main__":
    chunks = ingest_dir()
    by_type = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    print(f"Ingested {len(chunks)} chunks from sample_docs: {by_type}")
    print("\n--- first table chunk ---")
    for c in chunks:
        if c.chunk_type == "table":
            print(c.text)
            break
    print("--- a table_row chunk ---")
    for c in chunks:
        if c.chunk_type == "table_row":
            print(c.text)
            break
