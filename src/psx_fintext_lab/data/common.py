"""Shared paths and helpers for the data pipeline. Run every command from the repo root."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
SPLITS_DIR = DATA_DIR / "splits"
EXTERNAL_DIR = DATA_DIR / "external"
REPORTS_DIR = ROOT / "reports"
DATA_CONFIG = ROOT / "configs" / "data.yaml"

DATASET_PATH = PROCESSED_DIR / "dataset.parquet"
LLM_LABELS_PATH = INTERIM_DIR / "llm_labels.jsonl"
HUMAN_LABELS_PATH = LABELS_DIR / "human_labels.csv"
SAMPLE_IDS_PATH = LABELS_DIR / "verification_sample_ids.csv"

USER_AGENT = "psx-fintext-lab/0.1 (personal ML research project)"


def load_data_config(path: Path = DATA_CONFIG) -> dict:
    if not path.exists():
        raise SystemExit(f"Config not found at {path}. Run commands from the repo root.")
    return yaml.safe_load(path.read_text())


def make_session(extra_headers: dict | None = None) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    if extra_headers:
        session.headers.update(extra_headers)
    return session


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    retries: int = 3,
    backoff: float = 2.0,
    **kwargs,
) -> requests.Response:
    """Send a request, retrying with exponential backoff on network or HTTP errors."""
    for attempt in range(1, retries + 1):
        try:
            response = session.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = backoff**attempt
            print(f"  request failed ({exc}); retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def load_llm_labels(path: Path = LLM_LABELS_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["id", "sentiment", "event_type", "confidence", "model"])
    return pd.read_json(path, lines=True, dtype={"id": str})


def load_human_labels(path: Path = HUMAN_LABELS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["id", "sentiment", "labelled_at"])
    return pd.read_csv(path, dtype={"id": str})
