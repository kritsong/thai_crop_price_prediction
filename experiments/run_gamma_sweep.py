"""Run the full horizon-weighted loss gamma sweep and aggregate results.

Usage (from the package root):
    python run_gamma_sweep.py

The default is the complete 17-point grid reported in Table 6. Pass
``--gammas`` only for an intentional partial or incremental rerun.

Each configuration trains a fresh TFT via src/models/train_tft.py and writes
experiments_results/tft_hw_quantile_gamma_<g>_metrics.csv. After all runs, the
per-gamma files are concatenated into results/gamma_sweep_summary.csv.
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
# make the repository root importable regardless of how this is invoked
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import EXPERIMENTS_DIR

RESULTS_DIR = EXPERIMENTS_DIR
SUMMARY_PATH = PACKAGE_ROOT / "results" / "gamma_sweep_summary.csv"

DEFAULT_GAMMAS = [i / 2 for i in range(17)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    args = parser.parse_args()

    failures = []
    for gamma in args.gammas:
        print(f"\n===== Training TFT with gamma={gamma} =====", flush=True)
        cmd = [sys.executable, "-m", "src.models.train_tft", "--gamma", str(gamma)]
        result = subprocess.run(cmd, cwd=PACKAGE_ROOT)
        if result.returncode != 0:
            print(f"FAILED: gamma={gamma} exited with code {result.returncode}", flush=True)
            failures.append(gamma)

    # aggregate every completed gamma on disk, not just this invocation's list,
    # so incremental grid extensions keep the summary complete
    frames = []
    for path in sorted(glob.glob(str(RESULTS_DIR / "tft_hw_quantile_gamma_*_metrics.csv"))):
        frames.append(pd.read_csv(path))
    if not frames:
        print("WARNING: no per-gamma metrics files found", flush=True)

    if frames:
        summary = pd.concat(frames, ignore_index=True)
        if "Seed" in summary.columns:
            summary["Seed"] = summary["Seed"].fillna(42).astype(int)
        summary = summary.sort_values(["Gamma", "Horizon"])
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(SUMMARY_PATH, index=False)
        summary.to_csv(RESULTS_DIR / "gamma_sweep_summary.csv", index=False)
        print(f"\nWrote sweep summary ({len(summary)} rows) to {SUMMARY_PATH}", flush=True)

    if failures:
        print(f"Sweep finished with failures for gammas: {failures}", flush=True)
        sys.exit(1)
    print("Sweep finished successfully.", flush=True)


if __name__ == "__main__":
    main()
