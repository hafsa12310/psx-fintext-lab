from pathlib import Path

from psx_fintext_lab.data.fetch_pdfs import (
    is_results_title,
    key_figures_excerpt,
    key_term_pattern,
    merge_windows,
    ocr_cache_path,
)

TERMS = ["earnings per share", "eps", "net profit", "(loss)"]


def test_merge_windows_merges_overlaps_and_sorts() -> None:
    windows = [(50, 60), (0, 10), (5, 20), (20, 25), (40, 45)]
    assert merge_windows(windows) == [(0, 25), (40, 45), (50, 60)]


def test_key_term_pattern_respects_word_boundaries() -> None:
    pattern = key_term_pattern(TERMS)
    assert pattern.search("Basic EPS (Rupees) 4.20")
    assert pattern.search("Profit / (Loss) for the period")
    assert pattern.search("EARNINGS PER SHARE")
    assert not pattern.search("next steps for the company")  # "eps" inside a word


def test_excerpt_keeps_header_and_snippets_in_order() -> None:
    header = "HEADER cash dividend Rs. 5 per share. "
    filler = "x" * 200
    text = header + filler + " net profit 1,000 " + filler + " EPS 2.50 " + filler
    excerpt = key_figures_excerpt(
        text, key_term_pattern(TERMS), header_chars=20, snippet_chars=10, max_chars=4000
    )

    parts = excerpt.split(" ... ")
    assert parts[0] == text[:20]
    assert len(parts) == 3
    assert "net profit" in parts[1] and "EPS" in parts[2]
    assert excerpt.index("net profit") < excerpt.index("EPS")


def test_excerpt_merges_nearby_matches_into_one_snippet() -> None:
    text = "H" * 100 + " net profit 10 and EPS 1.0 " + "y" * 100
    excerpt = key_figures_excerpt(
        text, key_term_pattern(TERMS), header_chars=10, snippet_chars=20, max_chars=4000
    )
    assert excerpt.count(" ... ") == 1  # header + one merged snippet


def test_excerpt_merges_match_inside_header() -> None:
    text = "net profit up " + "z" * 300
    excerpt = key_figures_excerpt(
        text, key_term_pattern(TERMS), header_chars=50, snippet_chars=10, max_chars=4000
    )
    assert " ... " not in excerpt
    assert excerpt == text[:50]


def test_excerpt_is_capped_at_max_chars() -> None:
    text = "".join(f"block {i} net profit {i} " + "w" * 300 for i in range(50))
    excerpt = key_figures_excerpt(
        text, key_term_pattern(TERMS), header_chars=100, snippet_chars=50, max_chars=500
    )
    assert len(excerpt) == 500


def test_excerpt_without_matches_falls_back_to_start() -> None:
    text = "Board meeting notice " * 50
    excerpt = key_figures_excerpt(
        text, key_term_pattern(TERMS), header_chars=10, snippet_chars=10, max_chars=120
    )
    assert excerpt == text[:120]


def test_is_results_title() -> None:
    assert is_results_title("Financial Results for the Year Ended June 30, 2026")
    assert is_results_title("  FINANCIAL RESULTS FOR THE HALF YEAR")
    assert is_results_title("KTML-Financial Results for the year")
    assert is_results_title("FINANCIAL RESULT FOR THE YEAR ENDED JUNE 30, 2026")
    assert is_results_title("Announcement of Financial Results for the Half Year")
    assert not is_results_title("Board Meeting Other Than Financial Results")
    assert not is_results_title("Board Meeting for the Announcement of the Financial Results")
    assert not is_results_title("Credit of Interim Cash Dividend")
    assert not is_results_title(None)


def test_ocr_cache_names_by_page_count() -> None:
    pdf = Path("data/raw/pdfs/abc123.pdf")
    assert ocr_cache_path(pdf).name == "abc123.txt"
    assert ocr_cache_path(pdf, 8).name == "abc123_p8.txt"
