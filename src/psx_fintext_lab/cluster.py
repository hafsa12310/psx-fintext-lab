from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files without opening windows

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

ROOT = Path.cwd()
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FENCE = "~~~"


def load_train(embeddings_file: str) -> tuple[np.ndarray, pd.DataFrame]:
    X = np.load(PROCESSED / embeddings_file)
    df = pd.read_parquet(PROCESSED / "train.parquet")
    llm = pd.read_json(ROOT / "data/interim/llm_labels.jsonl", lines=True, dtype={"id": str})
    df = df.merge(llm[["id", "event_type"]], on="id", how="left")  # keeps row order
    assert len(df) == len(X), "embeddings and train.parquet are out of sync; re-run embed"
    return X, df


def scan_k(X: np.ndarray, k_values: range, seed: int) -> pd.DataFrame:
    """Fit K-means for each k and record inertia (elbow) and silhouette."""
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sil = silhouette_score(X, km.labels_, random_state=seed)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"  k={k:2d}  inertia={km.inertia_:8.1f}  silhouette={sil:.3f}")
    return pd.DataFrame(rows)


def top_terms(titles: pd.Series, labels: np.ndarray, n: int = 8) -> dict[int, str]:
    """The words that most distinguish each cluster's titles (highest average TF-IDF)."""
    vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    matrix = vec.fit_transform(titles.fillna(""))
    terms = np.array(vec.get_feature_names_out())
    out = {}
    for c in sorted(set(labels)):
        mean = np.asarray(matrix[labels == c].mean(axis=0)).ravel()
        out[c] = ", ".join(terms[mean.argsort()[::-1][:n]])
    return out


def plot_scan(scan: pd.DataFrame, chosen_k: int) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(scan["k"], scan["inertia"], "o-", color="tab:blue")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("inertia (lower = tighter)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(scan["k"], scan["silhouette"], "s--", color="tab:orange")
    ax2.set_ylabel("silhouette (higher = cleaner)", color="tab:orange")
    ax1.axvline(chosen_k, color="grey", linestyle=":")
    plt.title("Choosing k: elbow and silhouette")
    plt.tight_layout()
    plt.savefig(FIGURES / "kmeans_k_scan.png", dpi=150)
    plt.close()


def plot_pca(coords: np.ndarray, colour: pd.Series, title: str, filename: str) -> None:
    categories = colour.astype("category")
    plt.figure(figsize=(8, 6))
    for code, name in enumerate(categories.cat.categories):
        mask = (categories.cat.codes == code).to_numpy()
        plt.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.6, label=str(name))
    plt.legend(markerscale=3, fontsize=7, loc="best")
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=150)
    plt.close()


def section(title: str, body: str) -> str:
    return f"## {title}\n\n{FENCE}\n{body}\n{FENCE}\n"


def main() -> None:
    argparse.ArgumentParser(description="K-means clustering of embeddings").parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "clustering.yaml").read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)

    X, df = load_train(cfg.get("embeddings", "embeddings_train.npy"))
    print(f"Scanning k = {cfg['k_min']}..{cfg['k_max']} on {len(X)} training texts")
    scan = scan_k(X, range(cfg["k_min"], cfg["k_max"] + 1), cfg["seed"])
    k = cfg["k"] or int(scan.loc[scan["silhouette"].idxmax(), "k"])
    print(f"\nChosen k = {k}")
    plot_scan(scan, k)

    km = KMeans(n_clusters=k, n_init=10, random_state=cfg["seed"]).fit(X)
    labels = km.labels_
    df["cluster"] = labels

    events = df["event_type"].fillna("other")
    ari = adjusted_rand_score(events, labels)
    nmi = normalized_mutual_info_score(events, labels)

    pca = PCA(n_components=2, random_state=cfg["seed"])
    coords = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    plot_pca(coords, df["cluster"], f"K-means clusters (k={k}), PCA view", "pca_clusters.png")
    plot_pca(coords, events, "LLM event types, same PCA view", "pca_event_types.png")
    plot_pca(coords, df["sentiment"], "Labels, same PCA view", "pca_sentiment.png")

    terms = top_terms(df["title"], labels)
    sizes = pd.Series(labels).value_counts().sort_index()
    term_table = "\n".join(f"cluster {c:2d} ({sizes[c]:4d} rows): {terms[c]}" for c in terms)

    report = "\n".join(
        [
            f"# Clustering report\n\nTraining texts: **{len(X)}**, chosen **k = {k}**\n\n"
            f"- Silhouette at k={k}: **{scan.set_index('k').loc[k, 'silhouette']:.3f}**\n"
            f"- Agreement with LLM event types: ARI **{ari:.3f}**, NMI **{nmi:.3f}**\n"
            f"- PCA: 2 components explain **{explained:.1%}** of the variance\n",
            section("k scan", scan.round(3).to_string(index=False)),
            section("Top title words per cluster", term_table),
            section("Cluster vs event type", pd.crosstab(df["cluster"], events).to_string()),
            section("Cluster vs label", pd.crosstab(df["cluster"], df["sentiment"]).to_string()),
        ]
    )
    (REPORTS / "clustering.md").write_text(report)
    print(report)

    print("\nExample titles per cluster (not saved to the report):")
    for c in range(k):
        print(f"\n[cluster {c}]")
        for t in df.loc[df["cluster"] == c, "title"].head(3):
            print("   ", t)


if __name__ == "__main__":
    main()
