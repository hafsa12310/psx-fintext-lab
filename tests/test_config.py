from pathlib import Path

from psx_fintext_lab.train import load_config


def test_cli_overrides_config(tmp_path: Path) -> None:
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("model: logreg\nlr: 0.01\nepochs: 5\n")

    cfg = load_config(cfg_file, {"model": "cnn", "lr": None})

    assert cfg["model"] == "cnn"  # provided override wins
    assert cfg["lr"] == 0.01  # None does not override
    assert cfg["epochs"] == 5  # untouched keys stay


def test_base_config_loads() -> None:
    cfg = load_config(Path("configs/base.yaml"), {})
    assert {"seed", "model", "lr", "batch_size", "epochs"} <= cfg.keys()
