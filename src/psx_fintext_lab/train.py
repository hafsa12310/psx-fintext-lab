import argparse
from pathlib import Path

import torch
import yaml


def get_device() -> torch.device:
    """Pick the best available device: CUDA (Kaggle/Colab), MPS (Mac), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Path, overrides: dict) -> dict:
    """Load a YAML config, then apply CLI overrides that were actually provided."""
    cfg = yaml.safe_load(path.read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a PSX FinText model")
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    p.add_argument("--model", choices=["logreg", "mlp", "cnn", "transformer"])
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int, dest="batch_size")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(
        args.config,
        {
            "model": args.model,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
    )
    device = get_device()
    print(
        f"Training {cfg['model']} | lr={cfg['lr']} | epochs={cfg['epochs']} "
        f"| batch_size={cfg['batch_size']} | device={device}"
    )


if __name__ == "__main__":
    main()
