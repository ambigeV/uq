import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_SPECS: List[Tuple[str, str, str, str]] = [
    ("base", "dmpnn_bii_mc_dmpnn_balanced", "base::dmpnn_bii_mc_dmpnn_balanced", "Base MC"),
    ("base", "dmpnn_bii_new_dmpnn_balanced", "base::dmpnn_bii_new_dmpnn_balanced", "Base EVD"),
    ("base", "ensemble", "base::ensemble", "Base GBM"),
    ("stack", "uncertainty_inverse_isotonic", "stack::uncertainty_inverse_isotonic", "Ens Unc-ISO"),
    ("stack", "uniform", "stack::uniform", "Ens Uniform"),
]

METRICS: List[Tuple[str, str]] = [
    ("MCC", "MCC (higher is better)"),
    ("F1", "F1 (higher is better)"),
    ("AUC", "AUC (higher is better)"),
    ("Accuracy", "Accuracy (higher is better)"),
    ("Brier", "Brier (lower is better)"),
    ("NLL", "NLL (lower is better)"),
    ("ECE", "ECE (lower is better)"),
]


def _extract_mode_rows(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = df.copy()
    if "row_type" not in df.columns:
        return pd.DataFrame()
    if mode == "all_data":
        cand = df[df["row_type"] == "mean_std_over_runs"].copy()
        if "split" in cand.columns:
            cand = cand[cand["split"] == "test"].copy()
        return cand
    # CP modes
    cand = df[df["row_type"] == "conformal_mean_std_over_runs"].copy()
    if "split" in cand.columns:
        cand = cand[cand["split"] == "test_conformal_accepted"].copy()
    return cand


def _collect_mode_metric_table(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = _extract_mode_rows(df, mode)
    out_rows: List[Dict[str, object]] = []
    if rows.empty:
        return pd.DataFrame(out_rows)

    for family, name, key, label in METHOD_SPECS:
        rr = rows[(rows["model_family"] == family) & (rows["model_name"] == name)]
        if rr.empty:
            continue
        r = rr.iloc[0]
        for metric, _ in METRICS:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col not in rows.columns:
                continue
            out_rows.append(
                {
                    "mode": mode,
                    "method_key": key,
                    "method_label": label,
                    "metric": metric,
                    "value": float(r.get(mean_col, np.nan)),
                    "std": float(r.get(std_col, np.nan)) if std_col in rows.columns else float("nan"),
                }
            )
    return pd.DataFrame(out_rows)


def _plot_grouped_bars(plot_df: pd.DataFrame, out_png: Path) -> None:
    mode_order = ["all_data", "standard_cp", "weighted_cp_rf"]
    mode_label = {
        "all_data": "All Data",
        "standard_cp": "Standard CP",
        "weighted_cp_rf": "Weighted CP (RF)",
    }
    mode_color = {
        "all_data": "#4C78A8",
        "standard_cp": "#F58518",
        "weighted_cp_rf": "#54A24B",
    }

    method_labels = [x[3] for x in METHOD_SPECS]
    method_keys = [x[2] for x in METHOD_SPECS]
    metric_order = [m for m, _ in METRICS]
    metric_title = {m: t for m, t in METRICS}

    fig = plt.figure(figsize=(23, 13))
    gs = fig.add_gridspec(2, 12)
    axes: Dict[str, plt.Axes] = {}
    # Top row: 4 subplots
    for i, m in enumerate(metric_order[:4]):
        axes[m] = fig.add_subplot(gs[0, i * 3 : (i + 1) * 3])
    # Bottom row: 3 centered subplots
    for i, m in enumerate(metric_order[4:]):
        axes[m] = fig.add_subplot(gs[1, i * 4 : (i + 1) * 4])

    x = np.arange(len(method_keys), dtype=float)
    width = 0.23
    offsets = {
        "all_data": -width,
        "standard_cp": 0.0,
        "weighted_cp_rf": width,
    }

    for metric in metric_order:
        ax = axes[metric]
        for mode in mode_order:
            dd = plot_df[(plot_df["mode"] == mode) & (plot_df["metric"] == metric)]
            val_map = {k: np.nan for k in method_keys}
            std_map = {k: np.nan for k in method_keys}
            for _, row in dd.iterrows():
                val_map[str(row["method_key"])] = float(row["value"])
                std_map[str(row["method_key"])] = float(row["std"])
            vals = np.asarray([val_map[k] for k in method_keys], dtype=float)
            stds = np.asarray([std_map[k] for k in method_keys], dtype=float)
            yerr = np.where(np.isfinite(stds), stds, 0.0)
            ax.bar(
                x + offsets[mode],
                vals,
                width=width * 0.95,
                yerr=yerr,
                capsize=3,
                color=mode_color[mode],
                alpha=0.9,
                edgecolor="white",
                linewidth=0.6,
                label=mode_label[mode],
            )
        ax.set_title(metric_title[metric], fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=25, ha="right", fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[metric_order[0]].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=230, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified barplot for all-data / CP / weighted-CP comparison.")
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="inference_outputs/stacking/csv",
        help="Directory containing base.csv, cp.csv, wcp.csv",
    )
    parser.add_argument(
        "--base_csv",
        type=str,
        default="base.csv",
    )
    parser.add_argument(
        "--cp_csv",
        type=str,
        default="cp.csv",
    )
    parser.add_argument(
        "--wcp_csv",
        type=str,
        default="wcp.csv",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="inference_outputs/stacking/csv/unified_mode_comparison.png",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="inference_outputs/stacking/csv/unified_mode_comparison.csv",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    csv_dir = Path(args.csv_dir)
    if not csv_dir.is_absolute():
        csv_dir = repo_root / csv_dir

    base_path = csv_dir / args.base_csv
    cp_path = csv_dir / args.cp_csv
    wcp_path = csv_dir / args.wcp_csv

    df_base = pd.read_csv(base_path)
    df_cp = pd.read_csv(cp_path)
    df_wcp = pd.read_csv(wcp_path)

    t_base = _collect_mode_metric_table(df_base, "all_data")
    t_cp = _collect_mode_metric_table(df_cp, "standard_cp")
    t_wcp = _collect_mode_metric_table(df_wcp, "weighted_cp_rf")
    plot_df = pd.concat([t_base, t_cp, t_wcp], axis=0, ignore_index=True)
    if plot_df.empty:
        raise ValueError("No compatible summary rows found in base/cp/wcp CSV files.")

    out_png = Path(args.out_png)
    out_csv = Path(args.out_csv)
    if not out_png.is_absolute():
        out_png = repo_root / out_png
    if not out_csv.is_absolute():
        out_csv = repo_root / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(out_csv, index=False)
    _plot_grouped_bars(plot_df, out_png)
    print(f"Saved plot: {out_png}")
    print(f"Saved table: {out_csv}")


if __name__ == "__main__":
    main()
