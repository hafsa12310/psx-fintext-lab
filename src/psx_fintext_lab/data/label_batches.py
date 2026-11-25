"""Label the dataset with Claude Code, in batches.

1. uv run export-batches   -> data/interim/label_batches/batch_0001.jsonl, ...
2. In Claude Code, label each batch into batch_0001.labels.jsonl (see docs/claude_code_labelling.md)
3. uv run import-labels    -> validates the labels and adds them to data/interim/llm_labels.jsonl
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from psx_fintext_lab.data.common import (
    DATASET_PATH,
    INTERIM_DIR,
    LLM_LABELS_PATH,
    load_data_config,
    load_llm_labels,
)

BATCH_DIR = INTERIM_DIR / "label_batches"


def export_batches() -> None:
    parser = argparse.ArgumentParser(description="Write unlabelled rows into batch files")
    parser.add_argument("--size", type=int, default=50, help="Rows per batch file")
    args = parser.parse_args()

    cfg = load_data_config()["labelling"]
    df = pd.read_parquet(DATASET_PATH)
    done = set(load_llm_labels()["id"])
    already_exported: set[str] = set()
    for path in BATCH_DIR.glob("batch_*[0-9].jsonl"):
        already_exported |= set(pd.read_json(path, lines=True, dtype={"id": str})["id"])
    todo = df[~df["id"].isin(done | already_exported)]

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    start_number = len(list(BATCH_DIR.glob("batch_*[0-9].jsonl"))) + 1
    for i, start in enumerate(range(0, len(todo), args.size)):
        batch = todo.iloc[start : start + args.size]
        path = BATCH_DIR / f"batch_{start_number + i:04d}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in batch.itertuples():
                text = row.text[: cfg["max_text_chars"]]
                f.write(json.dumps({"id": row.id, "text": text}, ensure_ascii=False) + "\n")
    n_files = -(-len(todo) // args.size)  # ceiling division
    print(f"Wrote {len(todo)} rows into {n_files} batch files in {BATCH_DIR}")


def validate_row(item: dict, labels: list[str], event_types: list[str]) -> dict | None:
    """Return a clean label record, or None if the row is invalid."""
    sentiment = str(item.get("sentiment", "")).strip().lower()
    if not item.get("id") or sentiment not in labels:
        return None
    event = item.get("event_type")
    try:
        confidence = min(max(float(item.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "id": str(item["id"]),
        "sentiment": sentiment,
        "event_type": event if event in event_types else "other",
        "confidence": confidence,
        "model": "claude-code",
    }


def import_labels() -> None:
    argparse.ArgumentParser(description="Validate Claude Code labels and import them").parse_args()
    cfg = load_data_config()["labelling"]
    done = set(load_llm_labels()["id"])
    added = invalid = missing = 0

    LLM_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LLM_LABELS_PATH.open("a", encoding="utf-8") as out:
        for labels_path in sorted(BATCH_DIR.glob("batch_*.labels.jsonl")):
            batch_path = labels_path.with_name(labels_path.name.replace(".labels", ""))
            batch_ids = set(pd.read_json(batch_path, lines=True, dtype={"id": str})["id"])
            labelled_ids = set()
            for line in labels_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = validate_row(json.loads(line), cfg["labels"], cfg["event_types"])
                except json.JSONDecodeError:
                    record = None
                if record is None or record["id"] not in batch_ids:
                    invalid += 1
                    continue
                labelled_ids.add(record["id"])
                if record["id"] in done:
                    continue  # imported on an earlier run
                out.write(json.dumps(record) + "\n")
                done.add(record["id"])
                added += 1
            gap = batch_ids - labelled_ids
            if gap:
                missing += len(gap)
                print(f"  {labels_path.name}: {len(gap)} rows have no valid label")

    total = load_llm_labels()
    print(f"Added {added} | invalid lines {invalid} | rows still missing {missing}")
    print(f"Total labelled: {len(total)}")
    if not total.empty:
        print(total["sentiment"].value_counts(normalize=True).round(3).to_string())
