"""
Leakage-checked grid search for portfolio-aware CSI500 LightGBM models.

This script is an orchestration wrapper. It does NOT train the model itself.
It repeatedly calls model_v2_portfolio_aware_safe.py, validates the generated
portfolio, scores it with score_submission.py, and records the results.

Leakage controls happen at two layers:
  1. Sweep layer:
     - Each test window must be ASOF:START:END.
     - ASOF must be strictly before START.
     - START and END must be valid trading dates in data/index.parquet.
     - ASOF must be available in the local index data, unless --allow-asof-snap
       is provided.
     - By default, the sweep requires the model log to contain the phrase
       'latest label-safe training/validation date', proving the safe model
       applied the label cutoff.

  2. Model layer:
     - model_v2_portfolio_aware_safe.py internally caps train/validation labels
       at ASOF - FORWARD_HORIZON trading days, so early stopping does not see
       future labels.

Example:
  python sweep_portfolio_aware_checked.py \
    --model-script model_v2_portfolio_aware_safe.py \
    --windows 20260403:20260407:20260410 20260410:20260414:20260417 20260424:20260427:20260430 \
    --top-k 30 40 50 60 80 \
    --objectives huber lambdarank \
    --weight-methods rank vol_adjusted equal \
    --filter-modes none filter \
    --no-ensemble
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SAFE_LOG_PHRASE = "latest label-safe training/validation date"


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def parse_percent_line(label: str, text: str) -> float | None:
    pattern = rf"{re.escape(label)}\s*:\s*([+-]?\d+(?:\.\d+)?)%"
    m = re.search(pattern, text)
    if not m:
        return None
    return float(m.group(1)) / 100.0


def parse_window(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Bad window '{spec}'. Expected ASOF:START:END, e.g. 20260424:20260427:20260430"
        )
    for p in parts:
        datetime.strptime(p, "%Y%m%d")
    as_of, start, end = parts
    if pd.Timestamp(as_of) >= pd.Timestamp(start):
        raise ValueError(f"Leaky window '{spec}': ASOF must be strictly before START.")
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError(f"Bad window '{spec}': START must be <= END.")
    return as_of, start, end


def load_trading_dates(index_path: str) -> set[str]:
    idx = pd.read_parquet(index_path)
    idx["date"] = pd.to_datetime(idx["date"])
    return set(idx["date"].dt.strftime("%Y%m%d"))


def validate_windows_against_data(
    windows: list[tuple[str, str, str]],
    trading_dates: set[str],
    allow_asof_snap: bool = False,
) -> None:
    missing: list[str] = []
    for as_of, start, end in windows:
        if not allow_asof_snap and as_of not in trading_dates:
            missing.append(f"as_of={as_of}")
        if start not in trading_dates:
            missing.append(f"start={start}")
        if end not in trading_dates:
            missing.append(f"end={end}")
    if missing:
        sample = ", ".join(missing[:10])
        raise RuntimeError(
            "Some sweep dates are not in data/index.parquet. "
            f"First missing entries: {sample}. "
            "Run download_data.py --update or choose valid trading dates."
        )


def safe_name(**kwargs: Any) -> str:
    return "_".join(f"{k}{v}" for k, v in kwargs.items()).replace(".", "p")


def write_reports(results: list[dict[str, Any]], report_csv: str, report_json: str) -> None:
    df = pd.DataFrame(results)
    Path(report_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(report_json).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_csv, index=False)
    with open(report_json, "w") as f:
        json.dump(results, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-checked sweep for portfolio-aware LightGBM configs.")
    parser.add_argument("--model-script", default="model_v2_portfolio_aware_safe.py")
    parser.add_argument("--score-script", default="score_submission.py")
    parser.add_argument("--validate-script", default="validate_submission.py")
    parser.add_argument("--index", default="data/index.parquet")
    parser.add_argument("--windows", nargs="+", required=True,
                        help="One or more ASOF:START:END specs, e.g. 20260424:20260427:20260430")
    parser.add_argument("--top-k", nargs="+", type=int, default=[30, 40, 50, 60, 80])
    parser.add_argument("--objectives", nargs="+", choices=["huber", "lambdarank"], default=["huber", "lambdarank"])
    parser.add_argument("--weight-methods", nargs="+", choices=["rank", "equal", "vol_adjusted", "score_rank"],
                        default=["rank", "vol_adjusted", "equal"])
    parser.add_argument("--filter-modes", nargs="+", choices=["none", "filter"], default=["none", "filter"])
    parser.add_argument("--score-powers", nargs="+", type=float, default=[1.0],
                        help="Only relevant for weight_method=score_rank.")
    parser.add_argument("--no-ensemble", action="store_true", help="Use single model for speed.")
    parser.add_argument("--no-extra-data", action="store_true")
    parser.add_argument("--allow-asof-snap", action="store_true",
                        help="Allow model to snap non-trading ASOF to latest prior available date.")
    parser.add_argument("--allow-unsafe-model", action="store_true",
                        help="Do not require the safe-model cutoff log phrase. Use only if you know what you're doing.")
    parser.add_argument("--out-dir", default="submissions/sweep_checked")
    parser.add_argument("--report-csv", default="reports/portfolio_sweep_checked_results.csv")
    parser.add_argument("--report-json", default="reports/portfolio_sweep_checked_results.json")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    windows = [parse_window(w) for w in args.windows]
    trading_dates = load_trading_dates(args.index)
    validate_windows_against_data(windows, trading_dates, allow_asof_snap=args.allow_asof_snap)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs: list[dict[str, Any]] = []
    for top_k, objective, weight_method, filter_mode in itertools.product(
        args.top_k, args.objectives, args.weight_methods, args.filter_modes
    ):
        powers = args.score_powers if weight_method == "score_rank" else [1.0]
        for score_power in powers:
            configs.append({
                "top_k": top_k,
                "objective": objective,
                "weight_method": weight_method,
                "filter_mode": filter_mode,
                "score_power": score_power,
            })

    total = len(windows) * len(configs)
    print(f"Running {total} checked tests: {len(configs)} configs × {len(windows)} windows")
    print("Leakage checks enabled:")
    print("  ✓ ASOF < START <= END")
    print("  ✓ START/END exist in index data")
    print("  ✓ safe-model cutoff log required" if not args.allow_unsafe_model else "  ⚠ safe-model cutoff log NOT required")

    results: list[dict[str, Any]] = []
    counter = 0

    for as_of, start, end in windows:
        for cfg in configs:
            counter += 1
            tag = safe_name(
                asof=as_of,
                start=start,
                end=end,
                k=cfg["top_k"],
                obj=cfg["objective"],
                wt=cfg["weight_method"],
                filt=cfg["filter_mode"],
                pow=cfg["score_power"],
            )
            submission = out_dir / f"{tag}.csv"

            model_cmd = [
                sys.executable, args.model_script,
                "--as-of", as_of,
                "--top-k", str(cfg["top_k"]),
                "--objective", cfg["objective"],
                "--weight-method", cfg["weight_method"],
                "--score-power", str(cfg["score_power"]),
                "--out", str(submission),
                "--save-importance", "",
            ]
            if cfg["filter_mode"] == "filter":
                model_cmd.append("--filter-universe")
            if args.no_ensemble:
                model_cmd.append("--no-ensemble")
            if args.no_extra_data:
                model_cmd.append("--no-extra-data")

            print("\n" + "=" * 96)
            print(f"[{counter}/{total}] as_of={as_of}, score={start}->{end}, {cfg}")
            print("Generating portfolio...")
            model_rc, model_out = run_cmd(model_cmd)

            row: dict[str, Any] = {
                "as_of": as_of,
                "start": start,
                "end": end,
                **cfg,
                "submission": str(submission),
                "model_returncode": model_rc,
                "validate_returncode": None,
                "score_returncode": None,
                "portfolio_return": None,
                "benchmark_return": None,
                "excess_return": None,
                "status": "ok",
            }

            if model_rc != 0:
                row["status"] = "model_failed"
                row["model_tail"] = model_out[-2500:]
                print("MODEL FAILED")
                print(model_out[-2500:])
                results.append(row)
                write_reports(results, args.report_csv, args.report_json)
                if args.stop_on_error:
                    return
                continue

            if not args.allow_unsafe_model and SAFE_LOG_PHRASE not in model_out:
                row["status"] = "unsafe_model_log_missing"
                row["model_tail"] = model_out[-2500:]
                print("UNSAFE / UNVERIFIED MODEL RUN: safe cutoff phrase missing from model output.")
                print("Expected phrase:", SAFE_LOG_PHRASE)
                print(model_out[-2500:])
                results.append(row)
                write_reports(results, args.report_csv, args.report_json)
                if args.stop_on_error:
                    return
                continue

            print("Validating...")
            val_rc, val_out = run_cmd([sys.executable, args.validate_script, str(submission)])
            row["validate_returncode"] = val_rc
            if val_rc != 0:
                row["status"] = "validation_failed"
                row["validation_tail"] = val_out[-2500:]
                print("VALIDATION FAILED")
                print(val_out[-2500:])
                results.append(row)
                write_reports(results, args.report_csv, args.report_json)
                if args.stop_on_error:
                    return
                continue

            print("Scoring...")
            score_rc, score_out = run_cmd([
                sys.executable, args.score_script, str(submission),
                "--start", start,
                "--end", end,
            ])
            row["score_returncode"] = score_rc
            row["score_output"] = score_out
            if score_rc != 0:
                row["status"] = "score_failed"
                row["score_tail"] = score_out[-2500:]
                print("SCORE FAILED")
                print(score_out[-2500:])
                results.append(row)
                write_reports(results, args.report_csv, args.report_json)
                if args.stop_on_error:
                    return
                continue

            row["portfolio_return"] = parse_percent_line("portfolio return", score_out)
            row["benchmark_return"] = parse_percent_line("benchmark return", score_out)
            row["excess_return"] = parse_percent_line("excess return", score_out)
            print(score_out.strip())
            if row["excess_return"] is not None:
                print(f"Parsed excess_return={row['excess_return']:+.4%}")
            else:
                print("Could not parse excess return")

            results.append(row)
            write_reports(results, args.report_csv, args.report_json)

    df = pd.DataFrame(results)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("\nNo successful runs.")
        return

    summary = (
        ok.groupby(["top_k", "objective", "weight_method", "filter_mode", "score_power"], dropna=False)
        .agg(
            mean_excess=("excess_return", "mean"),
            median_excess=("excess_return", "median"),
            min_excess=("excess_return", "min"),
            max_excess=("excess_return", "max"),
            hit_rate=("excess_return", lambda x: float((x > 0).mean())),
            n=("excess_return", "count"),
        )
        .reset_index()
        .sort_values(["mean_excess", "hit_rate", "min_excess"], ascending=[False, False, False])
    )

    summary_path = Path(args.report_csv).with_name("portfolio_sweep_checked_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 96)
    print(f"Saved detailed results to {args.report_csv}")
    print(f"Saved summary to {summary_path}")
    print("\nTop configs:")
    print(summary.head(15).to_string(index=False, formatters={
        "mean_excess": "{:+.4%}".format,
        "median_excess": "{:+.4%}".format,
        "min_excess": "{:+.4%}".format,
        "max_excess": "{:+.4%}".format,
        "hit_rate": "{:.1%}".format,
    }))


if __name__ == "__main__":
    main()
