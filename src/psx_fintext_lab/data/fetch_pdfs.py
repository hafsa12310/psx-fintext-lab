"""Download announcement PDFs, extract a text excerpt, and build the final `text` field.

Many PSX announcements are scanned letters with no text layer. With --ocr, those are read
with Tesseract OCR. OCR results are cached in data/interim/ocr/, so re-runs are fast.

Output: data/processed/dataset.parquet
"""

from __future__ import annotations

import argparse
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


def extract_excerpt(pdf_path: Path, max_chars: int, max_pages: int = 3) -> str:
    """Text from the first few pages. Scanned (image-only) PDFs return an empty string."""
    try:
        reader = PdfReader(str(pdf_path))
        parts: list[str] = []
        for page in reader.pages[:max_pages]:
            parts.append(page.extract_text() or "")
            if sum(len(p) for p in parts) >= max_chars:
                break
        return normalize_text(" ".join(parts))[:max_chars]
    except Exception:  # corrupt or unreadable PDF: treat as having no text
        return ""


def ocr_excerpt(pdf_path: Path, max_chars: int, pages: int, dpi: int) -> str:
    """Read a scanned PDF with OCR. Results are cached as text files."""
    cache = OCR_DIR / f"{pdf_path.stem}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")[:max_chars]

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
    return text[:max_chars]


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
