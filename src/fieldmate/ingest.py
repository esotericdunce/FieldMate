"""
FieldMate Technical Document & Knowledge Ingestion CLI.

Features:
- Handles Markdown (.md), Plain Text (.txt), JSON (.json), and PDF (.pdf) manuals/books.
- Automatic semantic chunking (splits by markdown headings / ~1500-char paragraphs with overlap).
- Automatic OEM detection (Lenovo, Dell, HP, ASUS) and Fault Code extraction (WHEA, BSOD, 0x800..., etc.).
- Uses Qdrant Cloud Inference (dense + sparse BM25) for automatic server-side embeddings.

Usage:
    uv run python -m fieldmate.ingest --dir /home/hdd/projects/something/contents/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from fieldmate.brain.memory.models import (
    EvidenceType,
    MemoryEvidence,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
)
from fieldmate.brain.qdrant.config import QdrantConfig
from fieldmate.brain.qdrant.repository import QdrantMemoryRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fieldmate.ingest")

_OEM_PATTERNS = {
    "Lenovo": re.compile(r"\b(lenovo|thinkpad|ideapad|legion|yoga)\b", re.IGNORECASE),
    "Dell": re.compile(r"\b(dell|xps|latitude|inspiron|alienware|precision)\b", re.IGNORECASE),
    "HP": re.compile(r"\b(hp|hewlett-packard|spectre|envy|elitebook|pavilion|omen)\b", re.IGNORECASE),
    "ASUS": re.compile(r"\b(asus|rog|zenbook|vivobook|tuf)\b", re.IGNORECASE),
}

_FAULT_CODE_PATTERN = re.compile(r"\b(WHEA[A-Z_]*|BSOD|ERR-[0-9A-Z]+|0x[0-9A-Fa-f]{8}|STOP:\s*0x[0-9A-Fa-f]+)\b", re.IGNORECASE)


def detect_oem(text: str) -> str | None:
    for oem, pattern in _OEM_PATTERNS.items():
        if pattern.search(text):
            return oem
    return None


def extract_fault_codes(text: str) -> list[str]:
    matches = _FAULT_CODE_PATTERN.findall(text)
    return list(set(matches))


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 150) -> list[str]:
    """Semantic text chunker splitting by Markdown headers or character windows."""
    if not text or not text.strip():
        return []

    # First attempt: split by markdown headers if available
    headers = re.split(r"\n(?=#{1,4}\s+)", text)
    chunks: list[str] = []

    for section in headers:
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            # Paragraph splitting
            paragraphs = section.split("\n\n")
            current = ""
            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue
                if len(current) + len(p) + 2 <= max_chars:
                    current = f"{current}\n\n{p}".strip()
                else:
                    if current:
                        chunks.append(current)
                    if len(p) <= max_chars:
                        current = p
                    else:
                        # Character sliding window as fallback
                        start = 0
                        while start < len(p):
                            end = min(start + max_chars, len(p))
                            chunks.append(p[start:end].strip())
                            start += max_chars - overlap
                        current = ""
            if current:
                chunks.append(current)

    return [c for c in chunks if len(c) >= 50]  # ignore tiny noise chunks


async def ingest_record(
    repo: QdrantMemoryRepository,
    content: str,
    *,
    memory_type: MemoryType = MemoryType.PROCEDURAL,
    equipment_model: str | None = None,
    equipment_family: str | None = None,
    fault_codes: list[str] | None = None,
    source_ref: str = "ingest",
) -> str:
    """Ingest a single chunk into Qdrant Cloud via Cloud Inference."""
    evidence = MemoryEvidence(
        evidence_type=EvidenceType.MANUAL,
        reference_id=source_ref,
        description=f"Ingested technical document: {source_ref}",
    )

    detected_oem = equipment_family or detect_oem(content)
    detected_faults = fault_codes or extract_fault_codes(content)

    record = MemoryRecord(
        memory_type=memory_type,
        status=MemoryStatus.VERIFIED,
        content=content.strip(),
        confidence=1.0,
        equipment_model=equipment_model,
        equipment_family=detected_oem,
        fault_codes=detected_faults,
        evidence=[evidence],
    )

    point_id = await repo.upsert_memory(record)
    return point_id


async def ingest_pdf(repo: QdrantMemoryRepository, file_path: Path) -> int:
    """Extract and chunk PDF text pages into Qdrant Cloud."""
    if not HAS_PYPDF:
        logger.error("pypdf is not installed. Run `uv add pypdf` first.")
        return 0

    try:
        reader = PdfReader(str(file_path))
        full_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                full_text.append(text)
        
        doc_text = "\n\n".join(full_text)
        chunks = chunk_text(doc_text)
        logger.info("PDF %s extracted: %d pages, %d chunks", file_path.name, len(reader.pages), len(chunks))

        ingested_count = 0
        for idx, chunk in enumerate(chunks):
            await ingest_record(
                repo,
                chunk,
                source_ref=f"{file_path.name}#chunk{idx+1}",
            )
            ingested_count += 1
            if ingested_count % 25 == 0:
                logger.info("  -> Ingested %d/%d chunks for %s", ingested_count, len(chunks), file_path.name)
        
        return ingested_count
    except Exception:
        logger.exception("Failed to ingest PDF %s", file_path.name)
        return 0


async def ingest_file(
    repo: QdrantMemoryRepository,
    file_path: Path,
    *,
    equipment_model: str | None = None,
    fault_codes: list[str] | None = None,
) -> int:
    """Ingest a text, markdown, JSON, or PDF file into Qdrant Cloud."""
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return 0

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return await ingest_pdf(repo, file_path)

    if suffix == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        count = 0
        for item in items:
            content = item.get("content") or item.get("text") or ""
            if not content:
                continue
            m_type_str = item.get("memory_type", "procedural").lower()
            try:
                m_type = MemoryType(m_type_str)
            except ValueError:
                m_type = MemoryType.PROCEDURAL

            await ingest_record(
                repo,
                content,
                memory_type=m_type,
                equipment_model=item.get("equipment_model") or equipment_model,
                fault_codes=item.get("fault_codes") or fault_codes,
                source_ref=file_path.name,
            )
            count += 1
        return count

    # Markdown or Plain Text
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(raw_text)
    logger.info("Processing %s (%d chars -> %d chunks)", file_path.name, len(raw_text), len(chunks))

    count = 0
    for idx, chunk in enumerate(chunks):
        await ingest_record(
            repo,
            chunk,
            equipment_model=equipment_model,
            fault_codes=fault_codes,
            source_ref=f"{file_path.name}#chunk{idx+1}",
        )
        count += 1
        if count % 50 == 0:
            logger.info("  -> Ingested %d/%d chunks for %s", count, len(chunks), file_path.name)

    return count


async def main() -> None:
    parser = argparse.ArgumentParser(description="FieldMate Qdrant Knowledge Base Ingestion Engine")
    parser.add_argument("--file", type=str, help="Path to document file (markdown, text, json, pdf)")
    parser.add_argument("--dir", type=str, help="Path to directory containing documents")
    parser.add_argument("--text", type=str, help="Direct text snippet to ingest")
    parser.add_argument("--equipment", type=str, help="Target equipment model (e.g. 'ThinkPad X1 Carbon')")
    parser.add_argument("--fault-code", type=str, action="append", help="Fault code filter (e.g. 'ERR-17', 'BSOD')")
    parser.add_argument("--type", type=str, default="procedural", choices=["procedural", "equipment", "resolution", "pattern"])

    args = parser.parse_args()

    load_dotenv()
    config = QdrantConfig.from_env()
    repo = QdrantMemoryRepository(config)

    try:
        await repo.ensure_collection()
        fault_codes = args.fault_code or []

        if args.text:
            m_type = MemoryType(args.type)
            await ingest_record(
                repo,
                args.text,
                memory_type=m_type,
                equipment_model=args.equipment,
                fault_codes=fault_codes,
                source_ref="cli_text",
            )
            logger.info("Direct text snippet ingested successfully.")

        if args.file:
            count = await ingest_file(
                repo,
                Path(args.file),
                equipment_model=args.equipment,
                fault_codes=fault_codes,
            )
            logger.info("Ingestion complete for file: %s (%d chunks)", args.file, count)

        if args.dir:
            dir_path = Path(args.dir)
            if dir_path.is_dir():
                total_chunks = 0
                files = [p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in (".md", ".txt", ".json", ".pdf", ".epub")]
                logger.info("Starting ingestion for %d document(s) in %s", len(files), dir_path)
                for p in files:
                    count = await ingest_file(
                        repo,
                        p,
                        equipment_model=args.equipment,
                        fault_codes=fault_codes,
                    )
                    total_chunks += count
                logger.info("=== TOTAL INGESTION COMPLETE: %d chunks ingested into Qdrant Cloud ===", total_chunks)

        if not args.text and not args.file and not args.dir:
            parser.print_help()

    finally:
        await repo.close()


if __name__ == "__main__":
    asyncio.run(main())
