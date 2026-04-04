#!/usr/bin/env python3
"""Analyze Lunar Lander episode outcomes and parameter search behavior.

Useful for deciding when to move between optimization phases.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional


def parse_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round((len(sorted_vals) - 1) * q))))
    return sorted_vals[idx]


def load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            lv = parse_float(r.get("landing_velocity_fps", ""))
            if lv is None:
                continue
            params_raw = r.get("policy_params_json", "{}")
            try:
                params = json.loads(params_raw)
            except json.JSONDecodeError:
                params = {}
            rows.append(
                {
                    "episode_id": r.get("episode_id", ""),
                    "timestamp": r.get("timestamp", ""),
                    "landing_velocity_fps": lv,
                    "reward": parse_float(r.get("reward", "")),
                    "fuel_remaining_units": parse_float(r.get("fuel_remaining_units", "")),
                    "turns": parse_float(r.get("turns", "")),
                    "params": params,
                }
            )
    return rows


def summarize(rows: List[Dict], label: str) -> None:
    vals = sorted(r["landing_velocity_fps"] for r in rows)
    print(f"\n=== {label} ===")
    print(f"episodes: {len(vals)}")
    if not vals:
        return
    print(f"best(min): {vals[0]:.4f}")
    print(f"p10:       {quantile(vals, 0.10):.4f}")
    print(f"p25:       {quantile(vals, 0.25):.4f}")
    print(f"median:    {median(vals):.4f}")
    print(f"p75:       {quantile(vals, 0.75):.4f}")
    print(f"p90:       {quantile(vals, 0.90):.4f}")
    print(f"worst(max):{vals[-1]:.4f}")
    print(f"mean:      {mean(vals):.4f}")


def print_top(rows: List[Dict], top_n: int) -> None:
    top = sorted(rows, key=lambda r: r["landing_velocity_fps"])[:top_n]
    print(f"\n=== Top {len(top)} episodes (lowest landing velocity) ===")
    for r in top:
        print(
            f"episode={r['episode_id']} vel={r['landing_velocity_fps']:.4f} "
            f"fuel={r['fuel_remaining_units']} turns={r['turns']}"
        )


def summarize_top_params(rows: List[Dict], top_n: int) -> None:
    top = sorted(rows, key=lambda r: r["landing_velocity_fps"])[:top_n]
    agg: Dict[str, List[float]] = {}
    for r in top:
        for k, v in r["params"].items():
            if isinstance(v, (int, float)):
                agg.setdefault(k, []).append(float(v))

    print(f"\n=== Mean params over top {len(top)} episodes ===")
    for k in sorted(agg):
        vals = agg[k]
        print(f"{k:16s} mean={mean(vals):8.4f} min={min(vals):8.4f} max={max(vals):8.4f}")


def boundary_hits(rows: List[Dict], config_path: Path, eps: float = 1e-6) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    ranges = cfg.get("optimizer", {}).get("param_ranges", {})
    if not ranges:
        print("\n(no optimizer param_ranges found in config; skipping boundary analysis)")
        return

    counts = {k: {"low": 0, "high": 0, "n": 0} for k in ranges}
    for r in rows:
        p = r["params"]
        for k, lim in ranges.items():
            if k not in p or not isinstance(p[k], (int, float)):
                continue
            v = float(p[k])
            lo = float(lim["min"])
            hi = float(lim["max"])
            c = counts[k]
            c["n"] += 1
            if v <= lo + eps:
                c["low"] += 1
            if v >= hi - eps:
                c["high"] += 1

    print("\n=== Boundary-hit rates vs optimizer ranges ===")
    for k in sorted(counts):
        c = counts[k]
        n = c["n"]
        if n == 0:
            continue
        low_pct = 100.0 * c["low"] / n
        high_pct = 100.0 * c["high"] / n
        print(f"{k:16s} low={c['low']:5d} ({low_pct:6.2f}%) high={c['high']:5d} ({high_pct:6.2f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Lunar Lander episodes.csv")
    ap.add_argument("--episodes-csv", default="logs/episodes.csv")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--recent", type=int, default=200, help="Recent-episode window size")
    args = ap.parse_args()

    rows = load_rows(Path(args.episodes_csv))
    if not rows:
        raise SystemExit(f"No parseable rows found in {args.episodes_csv}")

    summarize(rows, "Overall")
    recent = rows[-args.recent :] if args.recent > 0 else rows
    summarize(recent, f"Recent ({len(recent)})")
    print_top(rows, args.top_n)
    summarize_top_params(rows, min(args.top_n, len(rows)))
    boundary_hits(rows, Path(args.config))


if __name__ == "__main__":
    main()
