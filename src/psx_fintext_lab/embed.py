from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

from psx_fintext_lab.train import get_device

ROOT = Path.cwd()
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "configs" / "embeddings.yaml"
SPLITS = ["train", "val", "test"]


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int, normalize: bool):
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute sentence embeddings for each split")
    parser.add_argument("--force", action="store_true", help="Recompute even if files exist")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    device = get_device()
    print(f"Loading {cfg['model']} on {device}")
    model = SentenceTransformer(cfg["model"], device=str(device))

    for split in SPLITS:
        suffix = "" if cfg.get("input", "text") == "text" else f"_{cfg['input']}"
        out = PROCESSED / f"embeddings_{split}{suffix}.npy"
        if out.exists() and not args.force:
            print(f"{split}: already done ({out.name})")
            continue
        df = pd.read_parquet(PROCESSED / f"{split}.parquet")
        texts = [str(t)[: cfg["max_chars"]] for t in df[cfg.get("input", "text")]]
        vectors = embed_texts(model, texts, cfg["batch_size"], cfg["normalize"])
        np.save(out, vectors.astype(np.float32))
        print(f"{split}: {vectors.shape[0]} texts -> {vectors.shape[1]} dimensions")


if __name__ == "__main__":
    main()
