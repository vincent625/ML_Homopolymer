#!/usr/bin/env python3
"""Generate the small, repository-tracked figures embedded in README.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SYSTEM_LABELS = {
    "ORIGINAL_ONCOMINE": "Original",
    "A1_VCF_CONTEXT": "VCF",
    "A2_VCF_BAM": "+ BAM",
    "A3_VCF_BAM_FLOW": "+ flow",
    "A4_PAIRED_NA": "+ paired NA",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "patch.linewidth": 0.7,
        }
    )


def _annotate_bars(ax, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def make_figure(metrics: pd.DataFrame, output_dir: Path) -> None:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    blue = "#0072B2"
    orange = "#D55E00"
    gray = "#7A7A7A"

    order = list(SYSTEM_LABELS)
    overall = metrics.loc[metrics["stratum"] == "ALL"].set_index("system").loc[order]
    x = np.arange(len(order))
    width = 0.36

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(180 / 25.4, 102 / 25.4),
        constrained_layout=True,
    )

    sensitivity = overall["sensitivity"].to_numpy() * 100
    precision = overall["precision"].to_numpy() * 100
    bars_sens = axes[0].bar(
        x - width / 2,
        sensitivity,
        width,
        color=blue,
        label="Sensitivity",
    )
    bars_ppv = axes[0].bar(
        x + width / 2,
        precision,
        width,
        color=orange,
        label="PPV",
    )
    _annotate_bars(axes[0], bars_sens)
    _annotate_bars(axes[0], bars_ppv)
    axes[0].set_title("A  All-hotspot performance")
    axes[0].set_ylabel("Performance (%)")
    axes[0].set_xticks(x, [SYSTEM_LABELS[item] for item in order])
    axes[0].set_ylim(0, 108)
    axes[0].set_yticks(np.arange(0, 101, 20))
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )

    strata = ["ALL", "HP", "NON_HP"]
    stratum_labels = ["All", "HP", "Non-HP"]
    original = (
        metrics.loc[metrics["system"] == "ORIGINAL_ONCOMINE"]
        .set_index("stratum")
        .loc[strata, "sensitivity"]
        .to_numpy()
        * 100
    )
    flow = (
        metrics.loc[metrics["system"] == "A3_VCF_BAM_FLOW"]
        .set_index("stratum")
        .loc[strata, "sensitivity"]
        .to_numpy()
        * 100
    )
    sx = np.arange(len(strata))
    bars_original = axes[1].bar(
        sx - width / 2,
        original,
        width,
        color=gray,
        label="Original Oncomine",
    )
    bars_flow = axes[1].bar(
        sx + width / 2,
        flow,
        width,
        color=blue,
        label="Flow add-on",
    )
    _annotate_bars(axes[1], bars_original)
    _annotate_bars(axes[1], bars_flow)
    axes[1].set_title("B  Sensitivity by indel context")
    axes[1].set_ylabel("Sensitivity (%)")
    axes[1].set_xticks(sx, stratum_labels)
    axes[1].set_ylim(0, 108)
    axes[1].set_yticks(np.arange(0, 101, 20))
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    axes[1].text(
        0.98,
        0.91,
        "Flow add-on: +2 TP, +1 FP\nSuccess gate not met",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=orange,
    )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Nested locus-grouped validation on 64 AOHC hotspot truth observations",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(output_dir / "hotspot_rescue_results.png")
    fig.savefig(output_dir / "hotspot_rescue_results.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        default="docs/results/hotspot_rescue_ablation.tsv",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="docs/figures",
        type=Path,
    )
    args = parser.parse_args()
    metrics = pd.read_csv(args.metrics, sep="\t")
    make_figure(metrics, args.output_dir)


if __name__ == "__main__":
    main()
