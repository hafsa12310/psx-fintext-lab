"""Measure how often the LLM agreed with your blind human labels.

Writes reports/label_quality.md (safe to commit: no article text) and
data/interim/label_disagreements.csv (for you to read: contains text, not committed).
"""

from __future__ import annotations

import argparse
import math

import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

from psx_fintext_lab.data.common import (
    DATASET_PATH,
    INTERIM_DIR,
    REPORTS_DIR,
    load_data_config,
    load_human_labels,
    load_llm_labels,
)

FENCE = "~~~"  # code-block marker for the Markdown report


def merge_labels(human: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    """Join human and LLM labels on id. Keeps your last answer if an id was labelled twice."""
    human = human.drop_duplicates("id", keep="last")
    human = human[human["sentiment"] != "skip"]
    return human[["id", "sentiment"]].merge(
        llm[["id", "sentiment", "event_type", "confidence"]],
        on="id",
        suffixes=("_human", "_llm"),
    )


def agreement_stats(merged: pd.DataFrame, labels: list[str]) -> dict:
    """Treat the human label as ground truth and the LLM as the predictor."""
    y_true, y_pred = merged["sentiment_human"], merged["sentiment_llm"]
    n = len(merged)
    accuracy = float((y_true == y_pred).mean())
    ci95 = 1.96 * math.sqrt(accuracy * (1 - accuracy) / n)  # normal approximation
    confusion = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"human: {lab}" for lab in labels],
        columns=[f"llm: {lab}" for lab in labels],
    )
    return {
        "n": n,
        "accuracy": accuracy,
        "ci95": ci95,
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "confusion": confusion,
        "per_class": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }


def accuracy_by(merged: pd.DataFrame, column: str) -> pd.DataFrame:
    correct = merged["sentiment_human"] == merged["sentiment_llm"]
    table = correct.groupby(merged[column], observed=True).agg(accuracy="mean", n="size")
    return table.round(3).sort_values("n", ascending=False)


def section(title: str, body: str) -> str:
    return f"## {title}\n\n{FENCE}\n{body}\n{FENCE}\n"


def main() -> None:
    argparse.ArgumentParser(description="LLM vs human label agreement report").parse_args()
    labels = load_data_config()["labelling"]["labels"]
    merged = merge_labels(load_human_labels(), load_llm_labels())
    if merged.empty:
        raise SystemExit("No overlap between human and LLM labels yet.")

    dataset = pd.read_parquet(
        DATASET_PATH, columns=["id", "source", "text_source", "title", "text"]
    )
    merged = merged.merge(dataset, on="id", how="left")
    merged["confidence_bin"] = pd.cut(
        merged["confidence"], [0, 0.6, 0.8, 0.9, 1.0], include_lowest=True
    )

    stats = agreement_stats(merged, labels)
    header = (
        "# LLM label quality\n\n"
        f"Human-verified items (blind): **{stats['n']}**\n\n"
        f"- Accuracy (LLM vs human): **{stats['accuracy']:.1%} "
        f"± {stats['ci95']:.1%}** (95% CI)\n"
        f"- Cohen's kappa: **{stats['kappa']:.3f}**\n"
    )
    report = "\n".join(
        [
            header,
            section("Confusion matrix", stats["confusion"].to_string()),
            section("Per-class (human = truth)", stats["per_class"]),
            section(
                "Accuracy by LLM self-reported confidence",
                accuracy_by(merged, "confidence_bin").to_string(),
            ),
            section(
                "Accuracy by event type (LLM's tag)", accuracy_by(merged, "event_type").to_string()
            ),
            section("Accuracy by text source", accuracy_by(merged, "text_source").to_string()),
        ]
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "label_quality.md").write_text(report)

    disagreements = merged[merged["sentiment_human"] != merged["sentiment_llm"]]
    columns = [
        "id",
        "sentiment_human",
        "sentiment_llm",
        "event_type",
        "confidence",
        "title",
        "text",
    ]
    disagreements[columns].to_csv(INTERIM_DIR / "label_disagreements.csv", index=False)

    print(report)
    print(f"{len(disagreements)} disagreements saved to data/interim/label_disagreements.csv")


if __name__ == "__main__":
    main()
