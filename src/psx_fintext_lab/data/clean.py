from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata

import pandas as pd

from psx_fintext_lab.data.common import INTERIM_DIR, RAW_DIR, load_data_config

COLUMNS = ["id", "source", "date", "symbol", "company", "title", "summary", "pdf_url", "url"]
CLEAN_PATH = INTERIM_DIR / "clean.parquet"


def normalize_text(value: object) -> str:
    """Unicode-normalize, collapse all whitespace to single spaces, and strip."""
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value).strip()


def title_key(title: str) -> str:
    """Group titles that differ only by case or numbers, e.g. dates in report titles."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", title.lower())).strip()


def make_id(*parts: object) -> str:
    """Stable ID from the fields that identify a row, so re-importing gives the same IDs."""
    joined = "|".join(str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def clean_psx(raw: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    df = raw.rename(columns=lambda c: str(c).strip().lower()).rename(columns=column_map)
    missing = {"date", "symbol", "title"} - set(df.columns)
    if missing:
        raise SystemExit(
            f"PSX data is missing columns {sorted(missing)}. Found {list(df.columns)}. "
            "Update psx.column_map in configs/data.yaml."
        )
    for col in ("company", "pdf_url"):
        if col not in df.columns:
            df[col] = None

    df["title"] = df["title"].map(normalize_text)
    df["company"] = df["company"].map(normalize_text)
    df["symbol"] = df["symbol"].map(normalize_text).str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["source"] = "psx"
    df["summary"] = ""
    df["url"] = df["pdf_url"]
    df["id"] = [
        make_id("psx", symbol, d.date() if pd.notna(d) else "", title)
        for symbol, d, title in zip(df["symbol"], df["date"], df["title"], strict=True)
    ]
    return df[COLUMNS]


def clean_news(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["source"] = df["feed"].fillna("news")
    df["title"] = df["title"].map(normalize_text)
    df["summary"] = df["summary"].map(normalize_text)
    dates = pd.to_datetime(df["published"], errors="coerce", utc=True, format="mixed")
    df["date"] = dates.dt.tz_localize(None)
    df["symbol"] = ""
    df["company"] = ""
    df["pdf_url"] = None
    df["id"] = [make_id("news", u, t) for u, t in zip(df["url"], df["title"], strict=True)]
    return df[COLUMNS]


def drop_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows we can't use: no date, no title, or a PSX row with no company symbol."""
    keep = df["date"].notna() & (df["title"] != "")
    keep &= ~((df["source"] == "psx") & (df["symbol"] == ""))
    return df[keep].copy(), int((~keep).sum())


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact repeats, then the same title for the same symbol on the same day."""
    before = len(df)
    df = df.drop_duplicates("id")
    key = (
        df["title"].str.lower()
        + "|"
        + df["symbol"].fillna("")
        + "|"
        + df["date"].dt.date.astype(str)
    )
    df = df[~key.duplicated()]
    return df.copy(), before - len(df)


def sample_rows(
    df: pd.DataFrame,
    max_rows: int,
    max_per_symbol: int,
    max_per_title: int,
    priority_patterns: list[str],
    seed: int,
) -> pd.DataFrame:
    """Pick a balanced subset: keep all news, prefer informative titles, cap routine ones."""
    shuffled = df.sample(frac=1, random_state=seed)
    news = shuffled[shuffled["source"] != "psx"]
    psx = shuffled[shuffled["source"] == "psx"].copy()

    pattern = "|".join(f"(?:{p.lower()})" for p in priority_patterns)
    psx["priority"] = psx["title"].str.lower().str.contains(pattern) if pattern else False

    # Per company: keep priority rows first, then fill up to the cap with the rest
    psx = psx.sort_values("priority", ascending=False, kind="stable")
    psx = psx.groupby("symbol", group_keys=False).head(max_per_symbol)

    # Routine titles are capped; priority titles are not
    routine = psx[~psx["priority"]]
    routine = routine.groupby(routine["title"].map(title_key), group_keys=False).head(max_per_title)

    budget = max(max_rows - len(news), 0)
    chosen = pd.concat([psx[psx["priority"]], routine]).head(budget)
    out = pd.concat([chosen, news]).drop(columns="priority")
    return out.sort_values("date").reset_index(drop=True)


def main() -> None:
    argparse.ArgumentParser(description="Clean, dedupe and sample raw data").parse_args()
    cfg = load_data_config()

    frames = []
    psx_path = RAW_DIR / "psx_announcements.csv"
    if psx_path.exists():
        raw_psx = pd.read_csv(psx_path, dtype=str)
        frames.append(clean_psx(raw_psx, cfg["psx"].get("column_map") or {}))
    news_paths = sorted((RAW_DIR / "news").glob("*.csv"))
    if news_paths:
        frames.append(clean_news(pd.concat([pd.read_csv(p, dtype=str) for p in news_paths])))
    if not frames:
        raise SystemExit("No raw data found. Run import-psx and/or scrape-news first.")

    df = pd.concat(frames, ignore_index=True)
    print(f"Raw rows: {len(df)}")
    print("Missing or empty values per column:")
    print(df.replace("", pd.NA).isna().sum().to_string())

    df, n_missing = drop_missing(df)
    print(f"\nDropped {n_missing} rows with no date, title or symbol")
    df, n_dupes = dedupe(df)
    print(f"Dropped {n_dupes} duplicate rows")

    s = cfg["sampling"]
    df = sample_rows(
        df,
        s["max_rows"],
        s["max_per_symbol"],
        s["max_per_title"],
        s["priority_patterns"],
        s["seed"],
    )
    print(f"\nKept {len(df)} rows after sampling:")
    print(df["source"].value_counts().to_string())
    print("\nMost common titles after sampling:")
    print(df["title"].map(title_key).value_counts().head(10).to_string())

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PATH, index=False)
    print(f"\nSaved to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
