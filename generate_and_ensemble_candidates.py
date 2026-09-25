"""
Generate multiple final portfolio candidates and build a self-ensemble.

Use from inside the ml-competition-sp26 project folder.

Example for phase 2 after updating data through 2026-05-08:
    python generate_and_ensemble_candidates.py \
      --as-of 20260508 \
      --model-script model_v2_portfolio_aware_safe.py \
      --final-out submissions/final_self_ensemble.csv

What it does:
  1. Runs model_v2_portfolio_aware_safe.py for a set of strong candidate configs.
  2. Validates each candidate CSV.
  3. Averages candidate weights.
  4. Keeps the top N aggregate names.
  5. Re-caps at 10%, normalizes to sum to 1, validates final output.

This is for final/live submissions where future scoring data is not available yet.
For historical testing, use sweep_portfolio_aware_checked.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

MIN_STOCKS = 30
MAX_WEIGHT = 0.10

# Good defaults based on the sweep: Huber + rank dominated; k=30/40/50 were useful.
# The no-filter k30 is included because it had the highest mean; filtered variants reduce downside.
DEFAULT_CONFIGS: list[dict[str, Any]] = [
    {"name": "k30_filter_rank", "top_k": 30, "objective": "huber", "weight_method": "rank", "filter": True,  "weight": 1.25},
    {"name": "k30_nofilter_rank", "top_k": 30, "objective": "huber", "weight_method": "rank", "filter": False, "weight": 1.00},
    {"name": "k40_filter_rank", "top_k": 40, "objective": "huber", "weight_method": "rank", "filter": True,  "weight": 1.00},
    {"name": "k50_filter_rank", "top_k": 50, "objective": "huber", "weight_method": "rank", "filter": True,  "weight": 0.75},
    {"name": "k40_filter_voladj", "top_k": 40, "objective": "huber", "weight_method": "vol_adjusted", "filter": True, "weight": 0.50},
]


def run(cmd: list[str], check: bool = True) -> str:
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed with return code {p.returncode}: {' '.join(cmd)}")
    return p.stdout


def cap_and_normalize(w: pd.Series, cap: float = MAX_WEIGHT) -> pd.Series:
    w = w.clip(lower=0).astype(float)
    if w.sum() <= 0:
        raise ValueError("All ensemble weights are zero.")
    w = w / w.sum()

    for _ in range(100):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w.loc[over] = cap
        free = ~over
        if not free.any():
            break
        free_sum = float(w.loc[free].sum())
        if free_sum <= 0:
            break
        w.loc[free] += excess * w.loc[free] / free_sum

    return w / w.sum()


def load_portfolio(path: Path, col_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].str.zfill(6)
    if not {"stock_code", "weight"}.issubset(df.columns):
        raise ValueError(f"{path} must contain stock_code,weight columns")
    return df[["stock_code", "weight"]].rename(columns={"weight": col_name})


def read_configs(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_CONFIGS
    with open(path, "r") as f:
        cfgs = json.load(f)
    if not isinstance(cfgs, list):
        raise ValueError("Config JSON must be a list of config dictionaries.")
    return cfgs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--as-of", required=True, help="YYYYMMDD prediction date, e.g. 20260508")
    p.add_argument("--model-script", default="model_v2_portfolio_aware_safe.py")
    p.add_argument("--validate-script", default="validate_submission.py")
    p.add_argument("--candidate-dir", default="submissions/final_candidates")
    p.add_argument("--final-out", default="submissions/final_self_ensemble.csv")
    p.add_argument("--configs-json", default=None, help="Optional JSON list of candidate configs")
    p.add_argument("--ensemble-top-n", type=int, default=50, help="Keep top N aggregate names. Must be >=30.")
    p.add_argument("--skip-training", action="store_true", help="Only ensemble existing candidate files.")
    p.add_argument("--no-ensemble", action="store_true", help="Pass --no-ensemble to model script for speed/testing.")
    args = p.parse_args()

    if args.ensemble_top_n < MIN_STOCKS:
        raise ValueError("--ensemble-top-n must be at least 30")

    candidate_dir = Path(args.candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    final_out = Path(args.final_out)
    final_out.parent.mkdir(parents=True, exist_ok=True)

    configs = read_configs(args.configs_json)
    candidate_paths: list[tuple[Path, float, str]] = []

    for cfg in configs:
        name = cfg["name"]
        out_path = candidate_dir / f"{args.as_of}_{name}.csv"
        candidate_paths.append((out_path, float(cfg.get("weight", 1.0)), name))

        if args.skip_training:
            continue

        cmd = [
            sys.executable, args.model_script,
            "--as-of", args.as_of,
            "--top-k", str(cfg["top_k"]),
            "--objective", cfg.get("objective", "huber"),
            "--weight-method", cfg.get("weight_method", "rank"),
            "--out", str(out_path),
        ]
        if cfg.get("filter", False):
            cmd.append("--filter-universe")
        if args.no_ensemble:
            cmd.append("--no-ensemble")
        if "score_power" in cfg:
            cmd += ["--score-power", str(cfg["score_power"])]

        run(cmd)
        run([sys.executable, args.validate_script, str(out_path)])

    merged: pd.DataFrame | None = None
    total_cfg_weight = 0.0
    overlap_rows = []

    for path, cfg_weight, name in candidate_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate file: {path}")
        col = f"w_{name}"
        df = load_portfolio(path, col)
        df[col] *= cfg_weight
        total_cfg_weight += cfg_weight
        merged = df if merged is None else merged.merge(df, on="stock_code", how="outer")

    assert merged is not None
    weight_cols = [c for c in merged.columns if c.startswith("w_")]
    merged[weight_cols] = merged[weight_cols].fillna(0.0)
    merged["appears"] = (merged[weight_cols] > 0).sum(axis=1)
    merged["avg_weight"] = merged[weight_cols].sum(axis=1) / total_cfg_weight

    # Keep top N consensus names, then normalize/cap.
    chosen = merged.sort_values("avg_weight", ascending=False).head(args.ensemble_top_n).copy()
    weights = cap_and_normalize(chosen.set_index("stock_code")["avg_weight"])

    out = weights.reset_index().rename(columns={"avg_weight": "weight", 0: "weight"})
    out.columns = ["stock_code", "weight"]
    out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
    out = out.sort_values("weight", ascending=False)
    out.to_csv(final_out, index=False)

    # Diagnostics
    diag = merged.sort_values("avg_weight", ascending=False).copy()
    diag_path = final_out.with_name(final_out.stem + "_diagnostics.csv")
    diag.to_csv(diag_path, index=False)

    print("\nFinal self-ensemble:")
    print(f"  wrote: {final_out}")
    print(f"  names: {len(out)}")
    print(f"  sum:   {out['weight'].sum():.8f}")
    print(f"  max:   {out['weight'].max():.4f}")
    print(f"  diagnostics: {diag_path}")
    print("\nConsensus counts among candidate portfolios:")
    print(diag["appears"].value_counts().sort_index().to_string())
    print("\nTop holdings:")
    print(out.head(15).to_string(index=False))

    run([sys.executable, args.validate_script, str(final_out)])


if __name__ == "__main__":
    main()
