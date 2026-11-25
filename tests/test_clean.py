import pandas as pd

from psx_fintext_lab.data.clean import clean_psx, dedupe, make_id, normalize_text, sample_rows


def test_normalize_text() -> None:
    assert normalize_text("  Notice\n\tof   AGM ") == "Notice of AGM"
    assert normalize_text(None) == ""


def test_make_id_is_stable() -> None:
    assert make_id("psx", "HBL", "2026-10-01") == make_id("psx", "HBL", "2026-10-01")
    assert make_id("psx", "HBL", "2026-10-01") != make_id("psx", "HBL", "2026-10-02")


def test_clean_psx_maps_and_normalizes_columns() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["Oct 5, 2026"],
            "Symbol": [" ggl "],
            "Name": ["Ghani"],
            "Title": ["Notice  of AGM"],
        }
    )
    row = clean_psx(raw, {"name": "company"}).iloc[0]
    assert row["symbol"] == "GGL"
    assert row["title"] == "Notice of AGM"
    assert row["company"] == "Ghani"
    assert row["date"].year == 2026


def test_dedupe_ignores_case_on_same_day() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-10-05", "2026-10-05", "2026-10-06"],
            "symbol": ["HBL", "HBL", "HBL"],
            "title": ["Board Meeting", "BOARD MEETING", "Board Meeting"],
        }
    )
    out, removed = dedupe(clean_psx(raw, {}))
    assert removed == 1
    assert len(out) == 2


def test_sampling_caps_routine_titles_but_keeps_priority_ones() -> None:
    titles = ["Board Meeting"] * 10 + ["Financial Results for the Quarter"] * 10
    raw = pd.DataFrame(
        {
            "date": [f"2026-10-{d:02d}" for d in range(1, 21)],
            "symbol": [f"S{i}" for i in range(20)],  # different companies, so no per-symbol cap
            "title": titles,
        }
    )
    out = sample_rows(
        clean_psx(raw, {}),
        max_rows=100,
        max_per_symbol=30,
        max_per_title=3,
        priority_patterns=["^financial results"],
        seed=0,
    )
    counts = out["title"].value_counts()
    assert counts["Board Meeting"] == 3  # routine title capped
    assert counts["Financial Results for the Quarter"] == 10  # priority title never capped
