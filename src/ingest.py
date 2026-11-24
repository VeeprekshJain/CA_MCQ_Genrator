from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import fitz

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.json"
SOURCES_METADATA_PATH = PROCESSED_DATA_DIR / "sources_metadata.json"


def ensure_data_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def save_chunks(chunks: list[dict[str, Any]], path: Path = CHUNKS_PATH) -> None:
    ensure_data_dirs()
    with path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)


def save_sources_metadata(metadata: list[dict[str, Any]]) -> None:
    ensure_data_dirs()
    with SOURCES_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    for line in lines:
        if not line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if len(line) < 3:
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def split_into_word_chunks(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step
    return chunks


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            cleaned = clean_text(raw_text)
            if cleaned:
                pages.append({"page_number": page_index, "text": cleaned})
    return pages


def infer_subject_and_chapter(filename: str) -> tuple[str, str]:
    normalized = filename.lower()
    if "idt" in normalized or "gst" in normalized or "indirect" in normalized:
        return "Indirect Tax Laws (IDT)", "General"
    if "direct" in normalized or "dt" in normalized or "capital" in normalized:
        return "Direct Tax Laws (DT)", "General"
    return "Indirect Tax Laws (IDT)", "General"


def _build_chunk_id(document_title: str, page_number: int, text: str) -> str:
    digest = hashlib.md5(f"{document_title}|{page_number}|{text}".encode("utf-8")).hexdigest()
    return f"chunk_{digest[:12]}"


def _build_chunks_from_pages(
    pages: list[dict[str, Any]],
    document_title: str,
    source_url: str,
    subject: str,
    chapter: str,
    edition_or_attempt: str,
    topic: str | None = None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    topic_value = topic or chapter or "General"

    for page in pages:
        page_number = int(page["page_number"])
        for chunk_text in split_into_word_chunks(str(page["text"])):
            chunk = {
                "chunk_id": _build_chunk_id(document_title, page_number, chunk_text),
                "text": chunk_text,
                "subject": subject,
                "chapter": chapter,
                "topic": topic_value,
                "document_title": document_title,
                "source_url": source_url,
                "edition_or_attempt": edition_or_attempt,
                "page_number": page_number,
            }
            chunks.append(chunk)
    return chunks


def _merge_chunks(existing: list[dict[str, Any]], new_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {chunk["chunk_id"]: chunk for chunk in existing if "chunk_id" in chunk}
    for chunk in new_chunks:
        merged[chunk["chunk_id"]] = chunk
    return list(merged.values())


def ingest_pdf_file(
    pdf_path: Path,
    subject: str,
    chapter: str,
    source_url: str,
    edition_or_attempt: str = "CA Final",
    topic: str | None = None,
) -> list[dict[str, Any]]:
    pages = extract_pdf_pages(pdf_path)
    document_title = pdf_path.stem
    return _build_chunks_from_pages(
        pages=pages,
        document_title=document_title,
        source_url=source_url,
        subject=subject,
        chapter=chapter,
        edition_or_attempt=edition_or_attempt,
        topic=topic,
    )


def ingest_raw_pdfs(
    default_subject: str | None = None,
    default_chapter: str | None = None,
    default_source_url: str | None = None,
    edition_or_attempt: str = "CA Final",
) -> list[dict[str, Any]]:
    ensure_data_dirs()
    existing = load_chunks()
    new_chunks: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []

    for pdf_path in sorted(RAW_DATA_DIR.glob("*.pdf")):
        inferred_subject, inferred_chapter = infer_subject_and_chapter(pdf_path.name)
        subject = default_subject or inferred_subject
        chapter = default_chapter or inferred_chapter
        source_url = default_source_url or "https://boslive.icai.org/"

        file_chunks = ingest_pdf_file(
            pdf_path=pdf_path,
            subject=subject,
            chapter=chapter,
            source_url=source_url,
            edition_or_attempt=edition_or_attempt,
            topic=chapter,
        )
        if file_chunks:
            new_chunks.extend(file_chunks)
            metadata.append(
                {
                    "document_title": pdf_path.stem,
                    "source_url": source_url,
                    "subject": subject,
                    "chapter": chapter,
                    "edition_or_attempt": edition_or_attempt,
                    "chunk_count": len(file_chunks),
                }
            )

    merged = _merge_chunks(existing, new_chunks)
    save_chunks(merged)
    if metadata:
        save_sources_metadata(metadata)
    return merged


def seed_sample_chunks(save: bool = True) -> list[dict[str, Any]]:
    ensure_data_dirs()

    sample_chunks = [
        {
            "chunk_id": "seed_idt_supply_1",
            "text": (
                "Under GST, supply includes all forms of supply of goods or services such as sale, transfer, "
                "barter, exchange, license, rental, lease or disposal made or agreed to be made for a "
                "consideration by a person in the course or furtherance of business. Activities listed in "
                "Schedule I are treated as supply even without consideration. Activities in Schedule III are "
                "neither supply of goods nor supply of services."
            ),
            "subject": "Indirect Tax Laws (IDT)",
            "chapter": "Supply under GST",
            "topic": "Scope of Supply",
            "document_title": "ICAI_CA_Final_IDT_Study_Material",
            "source_url": "https://boslive.icai.org/",
            "edition_or_attempt": "May/Nov 2026",
            "page_number": 42,
        },
        {
            "chunk_id": "seed_idt_supply_2",
            "text": (
                "Time of supply determines when liability to pay GST arises. For goods, time of supply is generally "
                "the earlier of date of issue of invoice or last date on which invoice is required to be issued, "
                "or date of receipt of payment. For services, similar principles apply with timelines linked to "
                "issue of invoice and receipt of payment."
            ),
            "subject": "Indirect Tax Laws (IDT)",
            "chapter": "Time and Value of Supply",
            "topic": "Time of Supply",
            "document_title": "ICAI_CA_Final_IDT_Study_Material",
            "source_url": "https://boslive.icai.org/",
            "edition_or_attempt": "May/Nov 2026",
            "page_number": 88,
        },
        {
            "chunk_id": "seed_idt_itc_1",
            "text": (
                "Input Tax Credit can be availed when the registered person is in possession of tax invoice, "
                "has received goods or services, tax has been paid to the government by the supplier, and return "
                "has been furnished. ITC is blocked on specific items under section 17(5), including certain motor "
                "vehicles, personal consumption, and works contract in specified cases."
            ),
            "subject": "Indirect Tax Laws (IDT)",
            "chapter": "Input Tax Credit (ITC)",
            "topic": "Eligibility and Blocked Credit",
            "document_title": "ICAI_CA_Final_IDT_Study_Material",
            "source_url": "https://boslive.icai.org/",
            "edition_or_attempt": "May/Nov 2026",
            "page_number": 135,
        },
        {
            "chunk_id": "seed_dt_capital_gains_1",
            "text": (
                "Capital gains arise when a capital asset is transferred. Chargeability is governed by section 45. "
                "Computation generally considers full value of consideration less expenditure incurred wholly and "
                "exclusively in connection with transfer and less indexed cost of acquisition or improvement, where "
                "indexation is applicable."
            ),
            "subject": "Direct Tax Laws (DT)",
            "chapter": "Capital Gains",
            "topic": "Charge and Computation",
            "document_title": "ICAI_CA_Final_DT_Study_Material",
            "source_url": "https://boslive.icai.org/",
            "edition_or_attempt": "May/Nov 2026",
            "page_number": 211,
        },
        {
            "chunk_id": "seed_dt_capital_gains_2",
            "text": (
                "For immovable property being land or building or both, period of holding of more than 24 months "
                "typically classifies the asset as long-term capital asset, subject to statutory exceptions. "
                "Short-term or long-term classification impacts rate and indexation treatment."
            ),
            "subject": "Direct Tax Laws (DT)",
            "chapter": "Capital Gains",
            "topic": "Period of Holding",
            "document_title": "ICAI_CA_Final_DT_Study_Material",
            "source_url": "https://boslive.icai.org/",
            "edition_or_attempt": "May/Nov 2026",
            "page_number": 223,
        },
        {
            "chunk_id": "seed_dt_capital_gains_3",
            "text": (
                "Exemptions from capital gains are available under various sections when conditions are fulfilled, "
                "such as reinvestment in specified assets within prescribed timelines. The exemption can be denied "
                "if lock-in conditions are violated or consideration is not utilized as required by the relevant "
                "section."
            ),
            "subject": "Direct Tax Laws (DT)",
            "chapter": "Capital Gains",
            "topic": "Exemptions",
            "document_title": "ICAI_CA_Final_DT_Study_Material",
            "source_url": "https://boslive.icai.org/",
            "edition_or_attempt": "May/Nov 2026",
            "page_number": 246,
        },
    ]

    if save:
        existing = load_chunks()
        merged = _merge_chunks(existing, sample_chunks)
        save_chunks(merged)
        save_sources_metadata(
            [
                {
                    "document_title": "ICAI_CA_Final_IDT_Study_Material",
                    "source_url": "https://boslive.icai.org/",
                    "subject": "Indirect Tax Laws (IDT)",
                    "chapter": "Supply under GST / ITC / Time and Value of Supply",
                    "edition_or_attempt": "May/Nov 2026",
                    "chunk_count": 3,
                },
                {
                    "document_title": "ICAI_CA_Final_DT_Study_Material",
                    "source_url": "https://boslive.icai.org/",
                    "subject": "Direct Tax Laws (DT)",
                    "chapter": "Capital Gains",
                    "edition_or_attempt": "May/Nov 2026",
                    "chunk_count": 3,
                },
            ]
        )

    return sample_chunks


def ensure_knowledge_base(seed_sample: bool = True) -> list[dict[str, Any]]:
    ensure_data_dirs()
    chunks = load_chunks()
    if chunks:
        return chunks

    chunks = ingest_raw_pdfs()
    if chunks:
        return chunks

    if seed_sample:
        seed_sample_chunks(save=True)
        return load_chunks()

    return []
