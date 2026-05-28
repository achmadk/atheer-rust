#!/usr/bin/env python3
"""Compare two perf-bench JSON reports and output the delta.

Usage:
    python tools/bench-compare.py before.json after.json [--markdown]

Output shows:
  - Avg tok/s delta (positive = improvement)
  - Peak tok/s delta
  - Per-batch-size throughput delta
  - Sustained sample count delta
  - Thermal state changes between runs
"""

import argparse
import json
import sys


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt(val: float, signed: bool = True) -> str:
    """Format a float delta with sign and 1 decimal place."""
    prefix = ""
    if signed:
        prefix = "+" if val > 0 else ""
    return f"{prefix}{val:.1f}"


def compare(before: dict, after: dict) -> list[str]:
    lines = []
    lines.append(f"{'Metric':<40} {'Before':>10} {'After':>10} {'Delta':>10}")
    lines.append("-" * 72)

    # Aggregate metrics
    for key, label in [
        ("avg_tokens_per_second", "Avg tok/s"),
        ("peak_tokens_per_second", "Peak tok/s"),
    ]:
        b = before.get(key, 0.0)
        a = after.get(key, 0.0)
        delta = a - b
        lines.append(
            f"{label:<40} {b:>10.1f} {a:>10.1f} {fmt(delta):>10}"
        )

    # Sustained sample count
    b_sc = before.get("sustained_sample_count", 0)
    a_sc = after.get("sustained_sample_count", 0)
    delta_sc = a_sc - b_sc
    prefix = "+" if delta_sc > 0 else ""
    lines.append(
        f"{'Sustained samples':<40} {b_sc:>10} {a_sc:>10} {prefix}{delta_sc:>9}"
    )

    # Sustained duration
    b_sd = before.get("sustained_duration_secs", 0)
    a_sd = after.get("sustained_duration_secs", 0)
    lines.append(
        f"{'Sustained duration (s)':<40} {b_sd:>10} {a_sd:>10} {'':>10}"
    )

    # Backend info
    lines.append(
        f"{'Backend':<40} {before.get('backend', '?'):>10} {after.get('backend', '?'):>10}"
    )
    lines.append(
        f"{'Device':<40} {before.get('device_name', '?'):>10} {after.get('device_name', '?'):>10}"
    )

    # Per-batch results
    b_batches = {r["batch_size"]: r for r in before.get("batch_results", [])}
    a_batches = {r["batch_size"]: r for r in after.get("batch_results", [])}
    all_sizes = sorted(set(list(b_batches.keys()) + list(a_batches.keys())))

    if all_sizes:
        lines.append("")
        lines.append(f"{'Batch Size':<40} {'Before tok/s':>10} {'After tok/s':>10} {'Delta':>10}")
        lines.append("-" * 72)
        for size in all_sizes:
            b_tps = b_batches.get(size, {}).get("tokens_per_second", 0.0)
            a_tps = a_batches.get(size, {}).get("tokens_per_second", 0.0)
            delta = a_tps - b_tps
            lines.append(
                f"{f'Batch {size}':<40} {b_tps:>10.1f} {a_tps:>10.1f} {fmt(delta):>10}"
            )

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Compare two perf-bench JSON reports"
    )
    parser.add_argument("before", help="Path to baseline bench-report.json")
    parser.add_argument("after", help="Path to new bench-report.json")
    parser.add_argument(
        "--markdown", action="store_true", help="Output as Markdown table"
    )
    args = parser.parse_args()

    try:
        before = load_report(args.before)
        after = load_report(args.after)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    lines = compare(before, after)

    if args.markdown:
        # Convert to markdown table
        print("| Metric | Before | After | Delta |")
        print("|--------|--------|-------|-------|")
        for line in lines[2:]:  # skip header and separator
            parts = [p.strip() for p in line.split("  ") if p.strip()]
            if len(parts) == 4:
                print(f"| {' | '.join(parts)} |")
            elif line.strip() == "":
                print()
    else:
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
