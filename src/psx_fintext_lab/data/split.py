from __future__ import annotations

import argparse
import hashlib
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from psx_fintext_lab.data.common import (
    DATASET_PATH,
    PROCESSED_DIR,
    SAMPLE_IDS_PATH,
    SPLITS_DIR,
    load_data_config,
    load_human_labels,
    load_llm_labels,
)

FINGERPRINT_PATH = SPLITS_DIR / "test_fingerprint.txt"


def text_key(text: str) -> str:
    """Lowercased text with punctuation removed, to catch near-identical duplicates."""
    return re.sub(r"\W+", " ", text.lower()).strip()


def fingerprint_test_set(test: pd.DataFrame) -> str:
    """Hash of (id, label) pairs. Changes if any test item or test label changes."""
    pairs = sorted(f"{i},{s}" for i, s in zip(test["id"], test["sentiment"], strict=True))
    return hashlib.sha256("\n".join(pairs).encode("utf-8")).hexdigest()


def make_splits(
    dataset: pd.DataFrame,
    llm: pd.DataFrame,
    human: pd.DataFrame,
    excluded_ids: set[str],
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    human = human.drop_duplicates("id", keep="last")  # your last answer wins
    human = human[human["sentiment"] != "skip"]
    test = dataset.merge(human[["id", "sentiment"]], on="id", how="inner")
    test["label_source"] = "human"

    # Everything in the verification sample (including skipped items) stays out of training
    pool = dataset[~dataset["id"].isin(excluded_ids | set(test["id"]))]
    pool = pool.merge(llm[["id", "sentiment", "confidence"]], on="id", how="inner")

    # Leakage guard: drop training rows whose text duplicates a test row
    test_keys = set(test["text"].map(text_key))
    leaked = pool["text"].map(text_key).isin(test_keys)
    pool = pool[~leaked].copy()
    pool["label_source"] = "llm"

    train, val = train_test_split(
        pool, test_size=val_size, stratify=pool["sentiment"], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        int(leaked.sum()),
    )


def check_frozen(test: pd.DataFrame, force: bool) -> None:
    fingerprint = fingerprint_test_set(test)
    if FINGERPRINT_PATH.exists():
        if FINGERPRINT_PATH.read_text().strip() == fingerprint:
            return
        if not force:
            raise SystemExit(
                "The test set differs from the frozen one. Re-freezing means every model "
                "result so far is no longer comparable. If you really mean it, use --force."
            )
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_PATH.write_text(fingerprint + "\n")
    print(f"Test set frozen: {fingerprint[:16]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make train/val/test splits")
    parser.add_argument("--force", action="store_true", help="Allow re-freezing the test set")
    args = parser.parse_args()

    cfg = load_data_config()["split"]
    human = load_human_labels()
    if human.empty:
        raise SystemExit("No human labels yet. Run verify-labels first.")
    dataset = pd.read_parquet(DATASET_PATH)
    excluded = set(pd.read_csv(SAMPLE_IDS_PATH, dtype=str)["id"])

    train, val, test, n_leaked = make_splits(
        dataset, load_llm_labels(), human, excluded, cfg["val_size"], cfg["seed"]
    )
    check_frozen(test, args.force)

    splits = {"train": train, "val": val, "test": test}
    for name, df in splits.items():
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        df[["id"]].to_csv(SPLITS_DIR / f"{name}_ids.csv", index=False)

    print(f"Removed {n_leaked} training rows that duplicated test texts")
    print(pd.Series({name: len(df) for name, df in splits.items()}, name="rows").to_string())
    dist = pd.concat(
        {name: df["sentiment"].value_counts(normalize=True) for name, df in splits.items()},
        axis=1,
    )
    print("\nLabel distribution:\n" + dist.round(3).to_string())


if __name__ == "__main__":
    main()
