"""Where everything lives.

The raw data and the checkpoints sit outside the repo, so we look for them in
three places, in order: an env var if you set one, a folder inside the repo, or
a folder next to it (which is how it is laid out on the machine we developed on).

Set CROP_DATA_DIR and CROP_EXPERIMENTS_DIR if yours are somewhere else.
Run this file directly to see what it picked up.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _resolve(env_var: str, *candidates: Path) -> Path:
    """First existing candidate wins; the environment variable always wins."""
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]          # nothing exists yet, use the documented default


# Raw Ministry of Commerce price files, one JSON per product.
DATA_DIR = _resolve(
    "CROP_DATA_DIR",
    ROOT / "data" / "historical_data_2018",
    ROOT.parent / "historical_data_2018",
)

# Model checkpoints and saved per-window predictions. Large, not version
# controlled; see README for how to regenerate or download.
EXPERIMENTS_DIR = _resolve(
    "CROP_EXPERIMENTS_DIR",
    ROOT / "experiments_results",
    ROOT.parent / "experiments_results",
)

# Version-controlled outputs.
RESULTS_DIR = ROOT / "results"
PAPER_DIR = ROOT / "paper"
IMAGE_DIR = PAPER_DIR / "images"

# Convenience handles used throughout the analysis scripts.
DATA_GLOB = str(DATA_DIR / "*.json")
CHAMPION_CKPT = EXPERIMENTS_DIR / "tft_hw_quantile_gamma_4_5.ckpt"
CHAMPION_NPZ = EXPERIMENTS_DIR / "tft_hw_quantile_gamma_4_5_predictions.npz"


def describe() -> str:
    return "\n".join(
        f"{name:18s} {value}"
        for name, value in [
            ("ROOT", ROOT),
            ("DATA_DIR", DATA_DIR),
            ("EXPERIMENTS_DIR", EXPERIMENTS_DIR),
            ("RESULTS_DIR", RESULTS_DIR),
            ("IMAGE_DIR", IMAGE_DIR),
        ]
    )


if __name__ == "__main__":
    print(describe())
    for label, p in [("DATA_DIR", DATA_DIR), ("EXPERIMENTS_DIR", EXPERIMENTS_DIR)]:
        print(f"{label} exists: {p.exists()}")
