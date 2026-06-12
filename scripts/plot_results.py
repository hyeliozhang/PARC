#!/usr/bin/env python3
"""Create publication-quality figures and paper tables for PARC.

The plotting code is intentionally conservative: vector PDF output, embedded
fonts, colorblind-safe palettes, line styles that survive grayscale printing,
and actual IEEE column widths so labels are legible after inclusion in LaTeX.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]

METHOD_LABELS = {
    "no_repair": "No repair",
    "majority": "Majority",
    "standard_fd": "FD repair",
    "rule_only": "Rule-only",
    "random_weighting": "Random weights",
    "outlier_removal": "Outlier removal",
    "provenance_unaware": "Prov.-unaware",
    "iterative_truth": "Iter. truth",
    "source_dependence": "Source-dep. vote",
    "dependency_truth": "Dep.+truth",
    "parc": "PARC",
    "parc_no_prov": "No provenance cap",
    "parc_no_constraints": "No constraints",
    "parc_no_trust": "No trust update",
    "parc_no_schema": "No schema repair",
    "parc_name_blocks": "Aggressive name ER",
}

# Okabe-Ito inspired palette with redundant markers/linestyles for print.
STYLE = {
    "parc": dict(color="#009E73", marker="o", linestyle="-", linewidth=1.70, markersize=4.2, zorder=8),
    "dependency_truth": dict(color="#0072B2", marker="o", linestyle="-", linewidth=1.25, markersize=3.4, zorder=6),
    "iterative_truth": dict(color="#CC79A7", marker="D", linestyle="--", linewidth=1.15, markersize=3.1, zorder=5),
    "source_dependence": dict(color="#56B4E9", marker="P", linestyle=(0, (3, 1, 1, 1)), linewidth=1.15, markersize=3.2, zorder=5),
    "provenance_unaware": dict(color="#D55E00", marker="v", linestyle="-.", linewidth=1.15, markersize=3.3, zorder=4),
    "majority": dict(color="#E69F00", marker="s", linestyle=(0, (5, 2)), linewidth=1.15, markersize=3.1, zorder=3),
    "outlier_removal": dict(color="#8A8A8A", marker="^", linestyle=":", linewidth=1.15, markersize=3.2, zorder=3),
    "no_repair": dict(color="#7A7A7A", marker="x", linestyle=":", linewidth=1.05, markersize=3.2, zorder=2),
}


def configure_plot_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 220,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.15,
        "axes.labelsize": 7.45,
        "axes.titlesize": 7.65,
        "axes.linewidth": 0.55,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 5.9,
        "legend.frameon": True,
        "legend.framealpha": 0.90,
        "legend.edgecolor": "#D0D0D0",
        "legend.borderpad": 0.25,
        "legend.labelspacing": 0.24,
        "legend.handlelength": 1.45,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _ensure_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible helper for older result CSVs."""
    df = df.copy()
    if "entity_f1_mean" not in df.columns:
        df["entity_f1_mean"] = 1.0
    if "integrated_quality_mean" not in df.columns:
        df["integrated_quality_mean"] = df.get("cell_accuracy_mean", 0.0) * df["entity_f1_mean"]
    if "entity_f1_std" not in df.columns:
        df["entity_f1_std"] = 0.0
    if "integrated_quality_std" not in df.columns:
        df["integrated_quality_std"] = df.get("cell_accuracy_std", 0.0)
    return df


def load_summary(results_dir: Path) -> pd.DataFrame:
    return _ensure_quality_columns(pd.read_csv(results_dir / "benchmark_summary.csv"))


def _label_method_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Method"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    return df


def _polish_axes(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.grid(True, which="major", axis=grid_axis, linewidth=0.30, color="#DADDE2", alpha=0.72)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(0.55)
    ax.tick_params(width=0.5, length=2.5, pad=1.6)


def _plot_line(ax: plt.Axes, df: pd.DataFrame, method: str, x_col: str, y_col: str, *, label: str | None = None, std_col: str | None = None, ci_n: int | None = None) -> None:
    g = df[df["method"] == method].sort_values(x_col)
    if g.empty:
        return
    style = STYLE.get(method, {})
    ax.plot(g[x_col], g[y_col], label=label or METHOD_LABELS.get(method, method),
            markeredgecolor="white" if method == "parc" else style.get("color", "black"),
            markeredgewidth=0.35 if method == "parc" else 0.25, **style)
    if std_col and std_col in g.columns and ci_n:
        x = g[x_col].to_numpy(dtype=float)
        y = g[y_col].to_numpy(dtype=float)
        sd = g[std_col].fillna(0.0).to_numpy(dtype=float)
        ci = 1.96 * sd / np.sqrt(ci_n)
        ax.fill_between(x, y - ci, y + ci, color=style.get("color", "black"), alpha=0.10, linewidth=0, zorder=style.get("zorder", 1)-1)


def _save(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.018)
    fig.savefig(fig_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.018)
    plt.close(fig)


def plot_corruption_model(fig_dir: Path) -> None:
    """Compile the architecture figure from its TikZ source.

    The conceptual figure is maintained as a standalone LaTeX/TikZ source file
    rather than as a raster drawing; it remains a TikZ vector diagram. Recompiling it keeps the paper PDF
    crisp while preserving the exact visual source in the artifact.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    tex_path = fig_dir / "corruption_model_tikz.tex"
    if not tex_path.exists():
        raise FileNotFoundError(f"missing TikZ source: {tex_path}")
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=fig_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    shutil.copyfile(fig_dir / "corruption_model_tikz.pdf", fig_dir / "corruption_model.pdf")
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-r", "600", "corruption_model.pdf", "corruption_model"],
            cwd=fig_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        generated = fig_dir / "corruption_model-1.png"
        if generated.exists():
            generated.replace(fig_dir / "corruption_model.png")
    except Exception:
        pass


def _direct_label(ax: plt.Axes, x: float, y: float, label: str, color: str, *, dy: float = 0.0) -> None:
    ax.text(x, y + dy, label, color=color, fontsize=5.9, va="center", ha="left", clip_on=False)


def plot_accuracy(summary: pd.DataFrame, fig_dir: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(3.48, 2.16))
    methods = ["no_repair", "majority", "provenance_unaware", "iterative_truth", "dependency_truth", "parc"]
    for m in methods:
        _plot_line(ax, summary, m, "budget", "integrated_quality_mean", std_col="integrated_quality_std" if m in {"parc", "dependency_truth"} else None, ci_n=5)
    ax.set_xlabel("Adversarial budget (%)")
    ax.set_ylabel("Integrated quality")
    ax.set_ylim(0.44, 1.015)
    ax.set_xlim(-0.006, 0.388)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{int(round(100*x))}"))
    ax.set_xticks([0, .05, .10, .15, .20, .25, .30, .35])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    _polish_axes(ax)
    # Direct labels avoid a boxed legend covering curves in the narrow IEEE column.
    last = summary.sort_values("budget").groupby("method").tail(1).set_index("method")
    offsets = {"parc": -0.010, "dependency_truth": 0.010, "iterative_truth": 0.000,
               "majority": 0.002, "provenance_unaware": -0.008, "no_repair": 0.000}
    for m in methods:
        if m in last.index:
            _direct_label(ax, 0.358, float(last.loc[m, "integrated_quality_mean"]),
                          METHOD_LABELS[m], STYLE[m]["color"], dy=offsets.get(m, 0.0))
    _save(fig, fig_dir, "accuracy_budget")


def plot_distortion(summary: pd.DataFrame, fig_dir: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(3.48, 2.16))
    methods = ["no_repair", "provenance_unaware", "iterative_truth", "dependency_truth", "parc"]
    floor = 0.0025
    plot_df = summary.copy()
    plot_df["aggregate_distortion_plot"] = plot_df["aggregate_distortion_mean"].clip(lower=floor)
    for m in methods:
        _plot_line(ax, plot_df, m, "budget", "aggregate_distortion_plot", label=METHOD_LABELS[m],
                   std_col=None)
    ax.set_xlabel("Adversarial budget (%)")
    ax.set_ylabel("Aggregate distortion")
    ax.set_yscale("log")
    ax.set_xlim(-0.006, 0.388)
    ax.set_ylim(floor*0.75, 1.7)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{int(round(100*x))}"))
    ax.set_xticks([0, .05, .10, .15, .20, .25, .30, .35])
    ax.yaxis.set_major_locator(FixedLocator([0.003, 0.01, 0.03, 0.1, 0.3, 1.0]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    _polish_axes(ax)
    ax.text(0.02, 0.98, "log scale; lower is better", transform=ax.transAxes, ha="left", va="top", fontsize=5.55, color="#555555")
    last = plot_df.sort_values("budget").groupby("method").tail(1).set_index("method")
    offsets = {"parc": -0.006, "dependency_truth": 0.008, "iterative_truth": 0.0,
               "provenance_unaware": 0.0, "no_repair": 0.0}
    for m in methods:
        if m in last.index:
            _direct_label(ax, 0.358, float(last.loc[m, "aggregate_distortion_plot"]),
                          METHOD_LABELS[m], STYLE[m]["color"], dy=offsets.get(m, 0.0))
    _save(fig, fig_dir, "aggregate_distortion_budget")


def plot_trust_sensitivity(results_dir: Path, fig_dir: Path) -> None:
    path = results_dir / "trust_sensitivity_summary.csv"
    if not path.exists():
        return
    configure_plot_style()
    df = pd.read_csv(path)
    df["fnr_pct"] = 100.0 * df["false_negative_rate_mean"]
    fig, ax = plt.subplots(figsize=(3.48, 2.03))
    for m in ["provenance_unaware", "dependency_truth", "parc"]:
        _plot_line(ax, df, m, "trust_alpha", "fnr_pct")
    ax.set_xlabel("Declared-trust manipulation")
    ax.set_ylabel("Dirty-claim FNR (pct. pts.)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0.0, 4.7)
    ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.set_yticks([0, 1, 2, 3, 4])
    _polish_axes(ax)
    ax.legend(loc="upper left", ncol=1, borderaxespad=0.25)
    _save(fig, fig_dir, "trust_sensitivity")


def _combine_scale_tables(results_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(results_dir / "scalability_summary.csv")
    rows = []
    for _, row in base.iterrows():
        if int(row["n_entities"]) in {150, 300, 600, 900}:
            rows.append({
                "n_entities": int(row["n_entities"]),
                "method": row["method"],
                "runtime_sec_mean": float(row["runtime_sec_mean"]),
                "runtime_sec_std": float(row.get("runtime_sec_std", 0.0) or 0.0),
            })
    big_path = results_dir / "large_scale_3000_summary.csv"
    if big_path.exists():
        big = pd.read_csv(big_path)
        for _, row in big.iterrows():
            rows.append({
                "n_entities": int(row["n_entities"]),
                "method": row["method"],
                "runtime_sec_mean": float(row["runtime_sec_mean"]),
                "runtime_sec_std": 0.0,
            })
    return pd.DataFrame(rows)


def plot_scalability(results_dir: Path, fig_dir: Path) -> None:
    if not (results_dir / "scalability_summary.csv").exists():
        return
    configure_plot_style()
    df = _combine_scale_tables(results_dir)
    fig, axes = plt.subplots(1, 2, figsize=(3.55, 1.92), gridspec_kw={"wspace": 0.30})

    ax = axes[0]
    for m in ["majority", "dependency_truth", "parc"]:
        _plot_line(ax, df, m, "n_entities", "runtime_sec_mean")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Entities")
    ax.set_ylabel("Runtime (s)")
    ax.set_xticks([150, 300, 600, 3000])
    ax.set_xticklabels(["150", "300", "600", "3K"], rotation=0)
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:g}"))
    _polish_axes(ax)
    ax.set_title("(a) Entity scale", pad=2)
    ax.legend(loc="upper left", ncol=1, borderaxespad=0.2, handlelength=1.2)

    ax = axes[1]
    src_path = results_dir / "source_scale_parc_summary.csv"
    if src_path.exists():
        src = pd.read_csv(src_path).sort_values("n_sources")
        ax.plot(src["n_sources"], src["runtime_sec_mean"], label="PARC", markeredgecolor="white", markeredgewidth=0.35, **STYLE["parc"])
        if "runtime_sec_std" in src.columns:
            x = src["n_sources"].to_numpy(dtype=float)
            y = src["runtime_sec_mean"].to_numpy(dtype=float)
            sd = src["runtime_sec_std"].fillna(0).to_numpy(dtype=float)
            ax.fill_between(x, y - sd, y + sd, color=STYLE["parc"]["color"], alpha=0.10, linewidth=0)
        ax.set_xlabel("Sources")
        ax.set_ylabel("Runtime (s)")
        ax.set_xlim(4.5, 37.5)
        ax.set_ylim(1.04, 1.92)
        ax.set_xticks([6, 12, 18, 24, 30, 36])
        ax.set_yticks([1.1, 1.3, 1.5, 1.7, 1.9])
        _polish_axes(ax)
        ax.set_title("(b) Source scale", pad=2)
        ax.text(0.98, 0.05, "36 feeds\n7.15K records", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.4, color="#555555")
    else:
        ax.axis("off")
    _save(fig, fig_dir, "runtime_scalability")


def make_tables(summary: pd.DataFrame, results_dir: Path, table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    sub = _label_method_column(summary[(summary["budget"].round(2) == 0.20)].copy())
    cols = [
        "Method", "integrated_quality_mean", "cell_accuracy_mean", "entity_f1_mean",
        "aggregate_distortion_mean", "attack_success_reduction_mean",
        "false_positive_rate_mean", "false_negative_rate_mean", "runtime_sec_mean",
    ]
    sub[cols].sort_values("integrated_quality_mean", ascending=False).to_csv(table_dir / "baseline_table_beta020.csv", index=False)

    neg = _label_method_column(summary[summary["budget"].round(2) == 0.35].copy())
    neg[["Method", "integrated_quality_mean", "cell_accuracy_mean", "entity_f1_mean", "aggregate_distortion_mean", "runtime_sec_mean"]].sort_values(
        "integrated_quality_mean", ascending=False
    ).to_csv(table_dir / "negative_results_beta035.csv", index=False)

    ab_path = results_dir / "ablation_summary.csv"
    if ab_path.exists():
        ab = _ensure_quality_columns(pd.read_csv(ab_path))
        ab = _label_method_column(ab)
        ab[["Method", "integrated_quality_mean", "cell_accuracy_mean", "entity_f1_mean", "aggregate_distortion_mean", "false_negative_rate_mean"]].sort_values(
            "integrated_quality_mean", ascending=False
        ).to_csv(table_dir / "ablation_table_beta030.csv", index=False)

    op_path = results_dir / "operator_ablation_summary.csv"
    if op_path.exists():
        op = _ensure_quality_columns(pd.read_csv(op_path))
        op = _label_method_column(op)
        pivot = op.pivot_table(index="operator_label", columns="Method", values="integrated_quality_mean")
        pivot.reset_index().to_csv(table_dir / "operator_ablation_table.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "data" / "results"))
    ap.add_argument("--figures", default=str(ROOT / "figures"))
    ap.add_argument("--tables", default=str(ROOT / "data" / "results"))
    args = ap.parse_args()
    results_dir = Path(args.results)
    fig_dir = Path(args.figures)
    table_dir = Path(args.tables)
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(results_dir)
    plot_corruption_model(fig_dir)
    plot_accuracy(summary, fig_dir)
    plot_distortion(summary, fig_dir)
    plot_trust_sensitivity(results_dir, fig_dir)
    plot_scalability(results_dir, fig_dir)
    make_tables(summary, results_dir, table_dir)
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
