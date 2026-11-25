"""Blind manual verification: label a random sample yourself WITHOUT seeing the LLM's label.

Progress is saved after every answer, so you can quit (q) and resume any time.
"""

from __future__ import annotations

import argparse
import textwrap
from datetime import datetime

import pandas as pd

from psx_fintext_lab.data.common import (
    DATASET_PATH,
    HUMAN_LABELS_PATH,
    SAMPLE_IDS_PATH,
    load_data_config,
    load_human_labels,
)

KEYS = {"p": "positive", "n": "negative", "u": "neutral", "s": "skip"}


def get_sample_ids(df: pd.DataFrame, n: int, seed: int) -> list[str]:
    """Draw the random sample once and save it, so it never changes between sessions."""
    if SAMPLE_IDS_PATH.exists():
        return pd.read_csv(SAMPLE_IDS_PATH, dtype=str)["id"].tolist()
    ids = df.sample(n=min(n, len(df)), random_state=seed)["id"].tolist()
    SAMPLE_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids}).to_csv(SAMPLE_IDS_PATH, index=False)
    return ids


def save(labelled: list[dict]) -> None:
    HUMAN_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = ["id", "sentiment", "labelled_at"]
    pd.DataFrame(labelled, columns=columns).to_csv(HUMAN_LABELS_PATH, index=False)


def show(row: pd.Series, done: int, total: int) -> None:
    print("\033c", end="")  # clear the terminal
    symbol = row["symbol"] or "-"
    print(f"[{done}/{total}]  {row['source']}  {row['date']:%Y-%m-%d}  {symbol}\n")
    print(textwrap.fill(row["text"][:3000], width=100))
    print("\n[p] positive   [n] negative   [u] neutral   [s] skip   [z] undo   [q] quit")


def main() -> None:
    argparse.ArgumentParser(description="Blind manual label verification").parse_args()
    cfg = load_data_config()["verification"]
    df = pd.read_parquet(DATASET_PATH)
    sample_ids = get_sample_ids(df, cfg["sample_size"], cfg["seed"])
    rows = df.set_index("id")

    labelled = load_human_labels().to_dict("records")
    done_ids = {r["id"] for r in labelled}
    queue = [i for i in sample_ids if i not in done_ids and i in rows.index]

    i = 0
    while i < len(queue):
        show(rows.loc[queue[i]], len(labelled), len(sample_ids))
        key = input("> ").strip().lower()
        if key == "q":
            break
        if key == "z":
            if labelled:
                queue.insert(i, labelled.pop()["id"])  # show the previous item again
                save(labelled)
            continue
        if key not in KEYS:
            continue
        labelled.append(
            {
                "id": queue[i],
                "sentiment": KEYS[key],
                "labelled_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        save(labelled)
        i += 1

    print(f"Saved {len(labelled)} of {len(sample_ids)} labels to {HUMAN_LABELS_PATH}")


if __name__ == "__main__":
    main()
