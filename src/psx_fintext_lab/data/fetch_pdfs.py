"""Download announcement PDFs, extract a text excerpt, and build the final `text` field.

Many PSX announcements are scanned letters with no text layer. With --ocr, those are read
with Tesseract OCR. OCR results are cached in data/interim/ocr/, so re-runs are fast.

"Financial results" announcements keep their profit/EPS figures in annexures on later pages.
For those rows more pages are read, and the excerpt is the letter header plus snippets around
key terms such as "earnings per share" (see key_figures_excerpt).

Output: data/processed/dataset.parquet
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from pypdf import PdfReader
from tqdm import tqdm

from psx_fintext_lab.data.clean import CLEAN_PATH, normalize_text
from psx_fintext_lab.data.common import (
    DATASET_PATH,
    INTERIM_DIR,
    RAW_DIR,
    load_data_config,
    make_session,
    request_with_retry,
)

PDF_DIR = RAW_DIR / "pdfs"
OCR_DIR = INTERIM_DIR / "ocr"


def read_pdf_text(pdf_path: Path, max_pages: int, stop_after_chars: int | None = None) -> str:
    """Normalized text of the first `max_pages` pages, stopping early once `stop_after_chars`
    characters are collected. Scanned (image-only) or unreadable PDFs return an empty string."""
    try:
        reader = PdfReader(str(pdf_path))
        parts: list[str] = []
        for page in reader.pages[:max_pages]:
            parts.append(page.extract_text() or "")
            if stop_after_chars is not None and sum(len(p) for p in parts) >= stop_after_chars:
                break
        return normalize_text(" ".join(parts))
    except Exception:  # corrupt or unreadable PDF: treat as having no text
        return ""


def extract_excerpt(pdf_path: Path, max_chars: int, max_pages: int = 3) -> str:
    """Text from the first few pages. Scanned (image-only) PDFs return an empty string."""
    return read_pdf_text(pdf_path, max_pages, stop_after_chars=max_chars)[:max_chars]


def ocr_cache_path(pdf_path: Path, pages: int | None = None) -> Path:
    """Cache file for OCR text. The default (short) read uses <id>.txt; longer reads use
    <id>_p<pages>.txt so they never overwrite or reuse the short cache."""
    suffix = f"_p{pages}" if pages is not None else ""
    return OCR_DIR / f"{pdf_path.stem}{suffix}.txt"


def ocr_text(pdf_path: Path, pages: int, dpi: int, cache: Path) -> str:
    """Read the first `pages` pages of a scanned PDF with OCR. Results are cached in `cache`."""
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    # Imported here so the rest of the pipeline works even without the OCR tools installed
    import pytesseract
    from pdf2image import convert_from_path

    try:
        images = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=pages)
        text = " ".join(pytesseract.image_to_string(img, lang="eng") for img in images)
        text = normalize_text(text)
    except Exception as exc:  # unreadable file or a page Poppler can't render
        print(f"  OCR failed for {pdf_path.name}: {exc}")
        text = ""

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")  # cache empty results too, so we don't retry them
    return text


def ocr_excerpt(pdf_path: Path, max_chars: int, pages: int, dpi: int) -> str:
    """OCR excerpt for ordinary rows, using the short <id>.txt cache."""
    return ocr_text(pdf_path, pages, dpi, ocr_cache_path(pdf_path))[:max_chars]


def is_results_title(title: object) -> bool:
    """True for results announcements: "financial result" anywhere in the title, so prefixed
    or singular titles ("KTML-Financial Results ...", "Financial Result for ...") count too.
    Board meeting notices ("Board Meeting Other Than Financial Results") are excluded."""
    if not isinstance(title, str):
        return False
    lowered = title.lower()
    return "financial result" in lowered and "board meeting" not in lowered


def key_term_pattern(terms: list[str]) -> re.Pattern[str]:
    """Case-insensitive regex for the key terms. Word boundaries are added only where the term
    starts or ends with a letter/digit, so "eps" doesn't match "steps" but "(loss)" still works."""
    parts = []
    for term in terms:
        start = r"\b" if term[:1].isalnum() else ""
        end = r"\b" if term[-1:].isalnum() else ""
        parts.append(f"{start}{re.escape(term)}{end}")
    return re.compile("|".join(parts), re.IGNORECASE)


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching (start, end) ranges, returned in document order."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def key_figures_excerpt(
    text: str,
    pattern: re.Pattern[str],
    header_chars: int,
    snippet_chars: int,
    max_chars: int,
) -> str:
    """The first `header_chars` characters, then " ... "-joined windows of ±`snippet_chars`
    around each key-term match (overlaps merged), cut to `max_chars` in total.
    With no matches it falls back to the plain first `max_chars` characters."""
    matches = [(m.start(), m.end()) for m in pattern.finditer(text)]
    if not matches:
        return text[:max_chars]

    windows = [(0, min(header_chars, len(text)))]
    windows += [(max(0, s - snippet_chars), min(len(text), e + snippet_chars)) for s, e in matches]
    excerpt = " ... ".join(text[s:e].strip() for s, e in merge_windows(windows))
    return excerpt[:max_chars]


def results_excerpt(path: Path, cfg: dict, use_ocr: bool) -> tuple[str, str]:
    """Key-figures excerpt for a financial results PDF, reading up to results_pages pages.
    Returns (excerpt, source) where source is "pdf" or "ocr".

    OCR is used when pypdf finds no text, and also when the text layer has no key terms:
    often only the cover letter has a text layer and the annexed statements are scanned."""
    pages = cfg["results_pages"]
    pattern = key_term_pattern(cfg["key_terms"])
    text, source = read_pdf_text(path, pages), "pdf"
    if use_ocr and not pattern.search(text):
        ocr = ocr_text(path, pages, cfg["ocr_dpi"], ocr_cache_path(path, pages))
        if not text or pattern.search(ocr):  # keep the text layer unless OCR finds the figures
            text, source = ocr, "ocr"
    excerpt = key_figures_excerpt(
        text, pattern, cfg["header_chars"], cfg["snippet_chars"], cfg["pdf_excerpt_chars"]
    )
    return excerpt, source


def build_text(title: str, summary: str, pdf_text: str) -> str:
    return " | ".join(part for part in (title, summary, pdf_text) if part)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PDFs and build the text field")
    parser.add_argument("--skip-download", action="store_true", help="Only use PDFs already saved")
    parser.add_argument("--ocr", action="store_true", help="OCR scanned PDFs that have no text")
    args = parser.parse_args()

    if not CLEAN_PATH.exists():
        raise SystemExit("Run clean-data first.")
    cfg = load_data_config()["psx"]
    df = pd.read_parquet(CLEAN_PATH)
    session = make_session()
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    if args.ocr:
        import pytesseract

        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            raise SystemExit(
                "Tesseract not found. Install it with: brew install tesseract poppler"
            ) from None

    excerpts, sources = [], []
    for row in tqdm(df.itertuples(), total=len(df), desc="pdfs"):
        if not (isinstance(row.pdf_url, str) and row.pdf_url):
            excerpts.append("")
            sources.append("title_only")
            continue

        path = PDF_DIR / f"{row.id}.pdf"
        if not path.exists() and not args.skip_download:
            try:
                url = urljoin(cfg["pdf_base_url"], row.pdf_url)
                path.write_bytes(request_with_retry(session, "GET", url).content)
                time.sleep(cfg.get("delay_seconds", 1.5))
            except requests.RequestException as exc:
                print(f"  failed to download {row.pdf_url}: {exc}")
        if not path.exists():
            excerpts.append("")
            sources.append("title_only")
            continue

        if is_results_title(row.title):
            text, source = results_excerpt(path, cfg, args.ocr)
        else:
            text = extract_excerpt(path, cfg["pdf_excerpt_chars"])
            source = "pdf"
            if not text and args.ocr:
                text = ocr_excerpt(path, cfg["pdf_excerpt_chars"], cfg["ocr_pages"], cfg["ocr_dpi"])
                source = "ocr"
        excerpts.append(text)
        sources.append(source if text else "title_only")

    df["pdf_text"] = excerpts
    df["text_source"] = sources
    df["text"] = [
        build_text(t, s, p)
        for t, s, p in zip(df["title"], df["summary"], df["pdf_text"], strict=True)
    ]

    print("\nWhere each row's text came from:")
    print(df["text_source"].value_counts().to_string())

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {DATASET_PATH}")


if __name__ == "__main__":
    main()
