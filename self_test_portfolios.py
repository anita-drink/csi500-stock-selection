"""Run leakage-safe portfolio self-tests over historical windows.

This script scores already-generated candidate portfolios or ensembles over one
or more historical windows, using score_submission.py. It is useful for report
self-test tables once you have generated portfolios with the correct as-of dates.

Window format:
  NAME:ASOF:START:END

The script does not train models. It assumes each candidate portfolio was created
using only data available as of ASOF. It records portfolio return, benchmark
return, excess return, trading days, and rank by window.

Example:
  python self_test_portfolios.py \
    --windows w1:20260403:20260407:20260410 w2:20260410:20260414:20260417 \
    --portfolios k30=submissions/k30_0403.csv k40=submissions/k40_0403.csv \
    --score-script score_submission.py \
    --out reports/self_test_portfolio_results.csv

For multi-window testing, use distinct portfolio CSVs per as-of/window, or run
the sweep_portfolio_aware_checked.py script, which generates and scores them.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
import pandas as pd


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout


def parse_pct(label: str, text: str) -> float | None:
    m = re.search(rf"{re.escape(label)}\s*:\s*([+-]?\d+(?:\.\d+)?)%", text)
    return None if not m else float(m.group(1)) / 100.0


def parse_days(text: str) -> int | None:
    m = re.search(r"\((\d+) trading days\)", text)
    return None if not m else int(m.group(1))


def parse_window(spec: str) -> dict[str, str]:
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError("Window must be NAME:ASOF:START:END")
    return {"window": parts[0], "as_of": parts[1], "start": parts[2], "end": parts[3]}


def parse_portfolio(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError("Portfolio must be NAME=path/to/file.csv")
    name, path = spec.split("=", 1)
    return name, path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", required=True, help="NAME:ASOF:START:END specs")
    ap.add_argument("--portfolios", nargs="+", required=True, help="NAME=portfolio.csv specs")
    ap.add_argument("--score-script", default="score_submission.py")
    ap.add_argument("--out", default="reports/self_test_portfolio_results.csv")
    args = ap.parse_args()

    windows = [parse_window(w) for w in args.windows]
    portfolios = [parse_portfolio(p) for p in args.portfolios]

    rows = []
    for win in windows:
        for model_name, path in portfolios:
            cmd = [sys.executable, args.score_script, path, "--start", win["start"], "--end", win["end"]]
            rc, out = run_cmd(cmd)
            row = {
                **win,
                "model": model_name,
                "portfolio_path": path,
                "returncode": rc,
                "portfolio_return": parse_pct("portfolio return", out),
                "benchmark_return": parse_pct("benchmark return", out),
                "excess_return": parse_pct("excess return", out),
                "trading_days": parse_days(out),
                "raw_output": out,
            }
            rows.append(row)
            print(f"{win['window']} | {model_name} | excess={row['excess_return']}")

    df = pd.DataFrame(rows)
    if not df.empty and "excess_return" in df.columns:
        df["rank_in_window"] = df.groupby("window")["excess_return"].rank(ascending=False, method="min")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    summary = (
        df.groupby("model", as_index=False)
        .agg(
            mean_excess=("excess_return", "mean"),
            median_excess=("excess_return", "median"),
            min_excess=("excess_return", "min"),
            max_excess=("excess_return", "max"),
            hit_rate=("excess_return", lambda x: float((x > 0).mean())),
            n=("excess_return", "count"),
        )
        .sort_values(["mean_excess", "hit_rate", "min_excess"], ascending=False)
    )
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved detailed results to {out_path}")
    print(f"Saved summary to {summary_path}")
    print(summary.to_string(index=False, formatters={
        "mean_excess": "{:+.4%}".format,
        "median_excess": "{:+.4%}".format,
        "min_excess": "{:+.4%}".format,
        "max_excess": "{:+.4%}".format,
        "hit_rate": "{:.1%}".format,
    }))


if __name__ == "__main__":
    main()
