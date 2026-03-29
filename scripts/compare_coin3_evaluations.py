#!/usr/bin/env python3
"""Compare multiple coin_3 evaluation runs.

This script expects each run directory to be produced by
`scripts/evaluate_coin3_model.py` and to contain at least:
  - run_config.json
  - summary_overall.csv
  - summary_family.csv
  - per_task_summary.csv (optional but used when present)
"""


import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


MAIN_METRICS = [
    "exact_match",
    "task_success",
    "ordered_ratio",
    "step_f1",
    "step_accuracy",
]


def read_csv(path: Path) -> list:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def maybe_float(value):
    if value is None:
        return value
    if isinstance(value, (float, int)):
        return value
    text = str(value).strip()
    if text == "":
        return value
    try:
        if "." in text or "e" in text.lower() or "-" in text[1:]:
            return float(text)
        return int(text)
    except Exception:
        return value


def normalize_rows(rows: list) -> list:
    return [{k: maybe_float(v) for k, v in row.items()} for row in rows]


def find_run_dirs(args) -> list:
    if args.run_dirs:
        dirs = [Path(p).resolve() for p in args.run_dirs]
    else:
        root = Path(args.input_root).resolve()
        dirs = sorted(
            path for path in root.iterdir()
            if path.is_dir() and (path / "summary_overall.csv").exists()
        )
    if not dirs:
        raise FileNotFoundError("No evaluation run directories found")
    return dirs


def format_run_label(lbl):
    # Remove junk substrings to normalize
    l = lbl.replace("coin3", "").replace("COIN3", "").replace("coin_3", "").replace("COIN-3", "").replace("_eval", "").replace("system1_", "").replace("best_model", "").replace("__", "_").strip("_")
    
    # Check for keywords reliably
    if "8b_galore" in l or "8B" in l:
        return "Perception-LM 8B (GaLore)"
    elif "plain" in l:
        return "Perception-LM 1B (Plain)"
    elif "grounded" in l or "latent" in l:
        return "Perception-LM 1B (Latent Proj)"
    
    # Fallback
    return l

def load_runs(run_dirs: list, scope: str) -> list:
    runs = []
    for run_dir in run_dirs:
        run_cfg = read_json(run_dir / "run_config.json") if (run_dir / "run_config.json").exists() else {}
        summary_name = "summary_overall.csv" if scope == "label" else "summary_family.csv"
        summary_rows = normalize_rows(read_csv(run_dir / summary_name))
        task_rows = []
        task_path = run_dir / "per_task_summary.csv"
        if task_path.exists():
            task_rows = normalize_rows(read_csv(task_path))
        runs.append(
            {
                "run_dir": run_dir,
                "run_label": format_run_label(run_cfg.get("model_tag", run_dir.name)),
                "run_id": run_cfg.get("run_id", run_dir.name),
                "config": run_cfg,
                "summary_rows": summary_rows,
                "task_rows": task_rows,
            }
        )
    return runs


def build_combined_rows(runs: list, configs: set) -> list:
    out = []
    for run in runs:
        for row in run["summary_rows"]:
            config = str(row.get("config", ""))
            if configs and config not in configs:
                continue
            merged = dict(row)
            merged["run_label"] = run["run_label"]
            merged["run_id"] = run["run_id"]
            out.append(merged)
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["empty"])
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list, headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("| empty |\n|---|\n| no rows |\n")
        return
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def comparison_table_rows(rows: list, scope: str) -> list:
    key_name = "condition_label" if scope == "label" else "condition_family"
    compact = []
    for row in rows:
        compact.append(
            {
                "run_label": row["run_label"],
                "config": row["config"],
                key_name: row.get(key_name, ""),
                "exact_match": row.get("exact_match", 0.0),
                "task_success": row.get("task_success", 0.0),
                "ordered_ratio": row.get("ordered_ratio", 0.0),
                "step_f1": row.get("step_f1", 0.0),
                "step_accuracy": row.get("step_accuracy", 0.0),
            }
        )
    compact.sort(key=lambda row: (row["config"], row[key_name], row["run_label"]))
    return compact


def best_rows(rows: list, scope: str) -> list:
    key_name = "condition_label" if scope == "label" else "condition_family"
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get(key_name, ""), row.get("config", ""))].append(row)

    out = []
    for (condition, config), group in sorted(grouped.items()):
        for metric in MAIN_METRICS:
            best = max(group, key=lambda row: float(row.get(metric, 0.0)))
            out.append(
                {
                    key_name: condition,
                    "config": config,
                    "metric": metric,
                    "best_run": best["run_label"],
                    "best_value": best.get(metric, 0.0),
                }
            )
    return out


def interpretation_gain_rows(runs: list) -> list:
    out = []
    for run in runs:
        by_cfg_family = {
            (row.get("config"), row.get("condition_family")): row
            for row in run["summary_rows"]
        }
        configs = sorted({row.get("config") for row in run["summary_rows"]})
        for config in configs:
            goal_only = by_cfg_family.get((config, "goal_only"))
            goal_interp = by_cfg_family.get((config, "goal_plus_interpretation"))
            if not goal_only or not goal_interp:
                continue
            row = {
                "run_label": run["run_label"],
                "config": config,
            }
            for metric in MAIN_METRICS:
                row[f"delta_{metric}"] = float(goal_interp.get(metric, 0.0)) - float(goal_only.get(metric, 0.0))
            out.append(row)
    return out


def plot_metric_grids(rows: list, out_dir: Path, scope: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: matplotlib unavailable ({exc})")
        return

    key_name = "condition_label" if scope == "label" else "condition_family"
    configs = sorted({str(row.get("config", "")) for row in rows})
    conditions = sorted({str(row.get(key_name, "")) for row in rows})
    run_labels = sorted({str(row.get("run_label", "")) for row in rows})

    for config in configs:
        cfg_rows = [row for row in rows if str(row.get("config", "")) == config]
        if not cfg_rows:
            continue
        fig, axes = plt.subplots(len(MAIN_METRICS), 1, figsize=(12, 3.4 * len(MAIN_METRICS)), sharex=True)
        if len(MAIN_METRICS) == 1:
            axes = [axes]
        x = np.arange(len(conditions))
        width = 0.8 / max(len(run_labels), 1)
        for ax, metric in zip(axes, MAIN_METRICS):
            for idx, run_label in enumerate(run_labels):
                vals = []
                for condition in conditions:
                    row = next(
                        (
                            item
                            for item in cfg_rows
                            if str(item.get("run_label", "")) == run_label and str(item.get(key_name, "")) == condition
                        ),
                        None,
                    )
                    vals.append(np.nan if row is None else float(row.get(metric, 0.0)))
                pos = x - 0.4 + width / 2.0 + idx * width
                ax.bar(pos, vals, width=width, label=run_label)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel(metric)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(conditions, rotation=20, ha="right")
        axes[0].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"Comparison by Condition: {config}")
        fig.tight_layout()
        fig.savefig(out_dir / f"comparison_{config}.png", dpi=180)
        plt.close(fig)


def plot_prefix_curves(rows: list, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    prefix_rows = [
        row
        for row in rows
        if str(row.get("condition_family", "")) == "goal_plus_interpretation_prefix"
    ]
    if not prefix_rows:
        return

    configs = sorted({str(row.get("config", "")) for row in prefix_rows})
    run_labels = sorted({str(row.get("run_label", "")) for row in prefix_rows})
    prefix_vals = sorted({int(row.get("prefix_len", 0)) for row in prefix_rows})
    metrics = ["step_accuracy", "ordered_ratio"]

    for config in configs:
        cfg_rows = [row for row in prefix_rows if str(row.get("config", "")) == config]
        fig, axes = plt.subplots(1, len(metrics), figsize=(11, 4.5), sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            for run_label in run_labels:
                xs = []
                ys = []
                for prefix_len in prefix_vals:
                    row = next(
                        (
                            item
                            for item in cfg_rows
                            if str(item.get("run_label", "")) == run_label and int(item.get("prefix_len", 0)) == prefix_len
                        ),
                        None,
                    )
                    if row is None:
                        continue
                    xs.append(prefix_len)
                    ys.append(float(row.get(metric, 0.0)))
                if xs:
                    ax.plot(xs, ys, marker="o", label=run_label)
            ax.set_title(metric)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_xlabel("Observed Prefix Steps")
        axes[0].set_ylabel("Metric")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Prefix Comparison: {config}")
        fig.tight_layout()
        fig.savefig(out_dir / f"prefix_comparison_{config}.png", dpi=180)
        plt.close(fig)


def plot_interpretation_gain(gain_rows: list, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not gain_rows:
        return

    configs = sorted({str(row.get("config", "")) for row in gain_rows})
    run_labels = sorted({str(row.get("run_label", "")) for row in gain_rows})
    metrics = ["delta_exact_match", "delta_task_success", "delta_ordered_ratio", "delta_step_f1"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 3.3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(run_labels))
    width = 0.8 / max(len(configs), 1)
    for ax, metric in zip(axes, metrics):
        for idx, config in enumerate(configs):
            vals = []
            for run_label in run_labels:
                row = next(
                    (
                        item
                        for item in gain_rows
                        if str(item.get("run_label", "")) == run_label and str(item.get("config", "")) == config
                    ),
                    None,
                )
                vals.append(0.0 if row is None else float(row.get(metric, 0.0)))
            pos = x - 0.4 + width / 2.0 + idx * width
            ax.bar(pos, vals, width=width, label=config)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(run_labels, rotation=20, ha="right")
    axes[0].legend(fontsize=8)
    fig.suptitle("Gain from Adding Interpretation (goal+interpretation - goal_only)")
    fig.tight_layout()
    fig.savefig(out_dir / "interpretation_gain.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple coin_3 evaluation runs")
    parser.add_argument("--input_root", default="outputs/coin_3")
    parser.add_argument("--run_dirs", nargs="*", default=None)
    parser.add_argument("--configs", default="",
                        help="Optional comma-separated config filter, e.g. system1_greedy,system1_critic_goal")
    parser.add_argument("--scope", choices=["label", "family"], default="label")
    parser.add_argument("--output_dir", default="outputs/coin_3/comparison")
    args = parser.parse_args()

    cfg_filter = {c.strip() for c in args.configs.split(",") if c.strip()} or None
    run_dirs = find_run_dirs(args)
    runs = load_runs(run_dirs, args.scope)
    combined_rows = build_combined_rows(runs, cfg_filter)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = comparison_table_rows(combined_rows, args.scope)
    best_rows_data = best_rows(combined_rows, args.scope)
    gain_rows = interpretation_gain_rows(runs)

    write_csv(out_dir / "comparison_summary.csv", combined_rows)
    write_json(out_dir / "comparison_summary.json", combined_rows)
    write_csv(out_dir / "comparison_table.csv", summary_rows)
    write_markdown(
        out_dir / "comparison_table.md",
        summary_rows,
        ["run_label", "config", "condition_label" if args.scope == "label" else "condition_family", "exact_match", "task_success", "ordered_ratio", "step_f1", "step_accuracy"],
    )
    write_csv(out_dir / "best_by_metric.csv", best_rows_data)
    write_json(out_dir / "interpretation_gain.json", gain_rows)
    write_csv(out_dir / "interpretation_gain.csv", gain_rows)

    plot_metric_grids(combined_rows, out_dir, args.scope)
    plot_prefix_curves(combined_rows, out_dir)
    plot_interpretation_gain(gain_rows, out_dir)

    print(f"Saved comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
