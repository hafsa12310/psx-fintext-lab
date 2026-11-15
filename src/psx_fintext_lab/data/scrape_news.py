"""Collect business news headlines from RSS feeds into data/raw/news/.

RSS feeds only contain the latest items, so run this regularly (for example once a day)
while you work on the project. Each run adds new items and skips ones already saved.
"""

from __future__ import annotations

import argparse

import feedparser
import pandas as pd
from bs4 import BeautifulSoup

from psx_fintext_lab.data.common import RAW_DIR, load_data_config, make_session, request_with_retry

OUT_DIR = RAW_DIR / "news"


def strip_html(value: str) -> str:
    """RSS summaries often contain HTML tags; keep only the text."""
    return BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True)


def main() -> None:
    argparse.ArgumentParser(description="Collect news headlines from RSS feeds").parse_args()
    cfg = load_data_config()["news"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    for feed in cfg["feeds"]:
        response = request_with_retry(session, "GET", feed["url"])
        parsed = feedparser.parse(response.content)
        new = pd.DataFrame(
            [
                {
                    "feed": feed["name"],
                    "title": entry.get("title"),
                    "summary": strip_html(entry.get("summary", "")),
                    "url": entry.get("link"),
                    "published": entry.get("published"),
                }
                for entry in parsed.entries
            ]
        )
        path = OUT_DIR / f"{feed['name']}.csv"
        existing = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame()
        combined = pd.concat([existing, new], ignore_index=True).drop_duplicates("url")
        combined.to_csv(path, index=False)
        print(f"{feed['name']}: +{len(combined) - len(existing)} new, {len(combined)} total")


if __name__ == "__main__":
    main()
