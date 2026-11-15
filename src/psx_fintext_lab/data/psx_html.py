"""Parse the PSX Data Portal announcements table (HTML) into a DataFrame."""

from __future__ import annotations

import re

import pandas as pd
from bs4 import BeautifulSoup

from psx_fintext_lab.data.common import RAW_DIR

COMBINED_PATH = RAW_DIR / "psx_announcements.csv"


def parse_html_table(html: str) -> pd.DataFrame:
    """Turn table rows into a DataFrame, using <th> text as column names and keeping PDF links."""
    soup = BeautifulSoup(html, "html.parser")
    scope = soup.find("table") or soup
    headers = [th.get_text(" ", strip=True).lower() for th in scope.find_all("th")]
    rows = []
    for tr in scope.find_all("tr"):
        cells = tr.find_all("td")
        if not cells:
            continue  # the header row has <th>, not <td>
        row = {
            (headers[i] if i < len(headers) else f"col_{i}"): td.get_text(" ", strip=True)
            for i, td in enumerate(cells)
        }
        link = tr.find("a", href=lambda h: bool(h) and ".pdf" in h.lower())
        row["pdf_url"] = link["href"] if link else None
        rows.append(row)
    return pd.DataFrame(rows)


def parse_total(html: str) -> int | None:
    """Read the total row count from the pagination buttons, e.g. data-total="997"."""
    match = re.search(r'data-total="(\d+)"', html)
    return int(match.group(1)) if match else None
