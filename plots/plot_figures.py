#!/usr/bin/env python3
"""
Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International

Generates all figures for the PEM multicore report. Figures can be produced individually with
--figure N or all at once without the flag. Outputs PDF (vector) and PNG
(300 dpi) for each figure, plus a derived_numbers.json with the computed
quantities to quote in the report text.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from data import (
    AggregatedCell,
    OUTPUT_DIR,
    aggregate,
    build_cuda_table,
    cuda_exclusion_reason,
    build_hyb_table,
    build_par1_baselines,
    build_seq_baselines,
    best_time_per_p,
    compute_derived_numbers,
    load_raw_rows,
    load_weak_rows,
    build_weak_table,
    parse_cuda_config,
    parse_hyb_config,
)

PALETTE: list[str] = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

SERIES_TESTS: list[str] = ["A_1M", "B_10", "A_100k", "A_20k", "C_5k"]

TEST_COLOURS: dict[str, str] = {
    test: PALETTE[i] for i, test in enumerate(SERIES_TESTS)
}

P_TICKS: list[int] = [1, 2, 4, 8, 16, 32, 64, 128]

CUDA_BRANCHES: list[str] = ["cuda-first", "cuda-1D", "cuda-shared", "main"]

BRANCH_COLOURS: dict[str, str] = {
    branch: PALETTE[i] for i, branch in enumerate(CUDA_BRANCHES)
}

BRANCH_LABELS: dict[str, str] = {
    "cuda-first": "cuda-first",
    "cuda-1D": "cuda-1D",
    "cuda-shared": "cuda-shared",
    "main": "main (historical)",
}


def setup_axes(ax: plt.Axes) -> None:
    """
    Applies the common style to an axes object: light grid behind the data,
    no top/right spines, legible font sizes for half-column width printing.
    """
    ax.grid(True, which="both", color="#d0d0d0", linewidth=0.5, linestyle="-",
            zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def save_figure(fig: plt.Figure, name: str) -> None:
    """
    Saves the figure as both PDF (vector) and PNG (300 dpi) into the plots
    output directory, then closes the figure to free memory.
    """
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}.pdf, {name}.png")


def plot_marker(
    ax: plt.Axes,
    x: float,
    y: float,
    colour: str,
    cell: AggregatedCell,
    marker: str = "o",
    size: float = 6.0,
) -> None:
    """
    Plots a single data marker: filled for clean cells, hollow (open) for
    flagged cells (contention fallback or below timing floor).
    """
    is_flagged: bool = cell.used_minimum or cell.below_floor
    ax.plot(
        x, y,
        marker=marker,
        markersize=size,
        color=colour,
        markerfacecolor="white" if is_flagged else colour,
        markeredgecolor=colour,
        markeredgewidth=1.5 if is_flagged else 1.0,
        zorder=5,
    )


def figure1_speedup(
    hyb_table: dict[str, dict[str, AggregatedCell]],
    seq_baselines: dict[str, AggregatedCell],
) -> None:
    """
    Figure 1: strong-scaling speedup S(p) = T_serial / T(p) for each test,
    taking the best configuration at each core count p. Log-log axes with a
    linear-speedup reference line.
    """
    print("Figure 1: Strong scaling speedup")
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    setup_axes(ax)

    p_range: np.ndarray = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=float)
    ax.plot(p_range, p_range, "--", color="grey", linewidth=1.0, alpha=0.6,
            label="Linear speedup", zorder=2)

    for test_id in SERIES_TESTS:
        if test_id not in seq_baselines:
            print(f"  WARNING: no SEQ baseline for {test_id}, skipping")
            continue
        t_serial: float = seq_baselines[test_id].value
        p_best: dict[int, AggregatedCell] = best_time_per_p(hyb_table, test_id)

        ps: list[int] = sorted(p_best.keys())
        speedups: list[float] = [t_serial / p_best[p].value for p in ps]

        colour: str = TEST_COLOURS[test_id]
        ax.plot(ps, speedups, "-", color=colour, linewidth=1.2, zorder=3,
                label=test_id)
        for p, s in zip(ps, speedups):
            plot_marker(ax, p, s, colour, p_best[p])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(P_TICKS)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Total cores (p)", fontsize=9)
    ax.set_ylabel("Speedup S(p) = T_serial / T(p)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

    save_figure(fig, "fig1_speedup")


def figure2_efficiency(
    hyb_table: dict[str, dict[str, AggregatedCell]],
    par1_baselines: dict[str, AggregatedCell],
) -> None:
    """
    Figure 2: strong-scaling efficiency E(p) = T_par(1) / (p * T(p)) for each
    test, with a horizontal reference at E=1 and a vertical annotation at the
    single-node to multi-node boundary (p=32 to p=64).
    """
    print("Figure 2: Strong scaling efficiency")
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    setup_axes(ax)

    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=1.0, alpha=0.6,
               label="Ideal efficiency", zorder=2)

    ax.axvline(x=np.sqrt(32.0 * 64.0), color="grey", linestyle=":",
               linewidth=0.8, alpha=0.5, zorder=2)
    ax.text(np.sqrt(32.0 * 64.0), 0.05, "single-node | multi-node",
            fontsize=6, color="grey", ha="center", rotation=90, va="bottom")

    for test_id in SERIES_TESTS:
        if test_id not in par1_baselines:
            print(f"  WARNING: no T_par(1) for {test_id}, skipping")
            continue
        t_par1: float = par1_baselines[test_id].value
        p_best: dict[int, AggregatedCell] = best_time_per_p(hyb_table, test_id)

        ps: list[int] = sorted(p_best.keys())
        efficiencies: list[float] = [
            t_par1 / (p * p_best[p].value) for p in ps
        ]

        colour: str = TEST_COLOURS[test_id]
        ax.plot(ps, efficiencies, "-", color=colour, linewidth=1.2, zorder=3)
        for p, e in zip(ps, efficiencies):
            plot_marker(ax, p, e, colour, p_best[p])

        label_p = ps[-1]
        ax.annotate(
            test_id, (label_p, efficiencies[-1]),
            textcoords="offset points", xytext=(6, 0),
            fontsize=7, color=colour, va="center",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(P_TICKS)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Total cores (p)", fontsize=9)
    ax.set_ylabel("Efficiency E(p) = T_par(1) / (p * T(p))", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.8)

    save_figure(fig, "fig2_efficiency")


def figure3_weak(weak_table: dict[int, AggregatedCell]) -> None:
    """
    Figure 3: weak-scaling time and efficiency. Primary y-axis shows median
    execution time (linear, starting at 0); secondary y-axis shows weak
    efficiency E_w(p) = T(4)/T(p). Annotates the inter-node step between
    p=32 and p=64 and draws min-max error bars.
    """
    print("Figure 3: Weak scaling")
    fig, ax1 = plt.subplots(figsize=(7.0, 5.0))
    setup_axes(ax1)

    ps: list[int] = sorted(weak_table.keys())
    times: list[float] = [weak_table[p].value for p in ps]
    t_base: float = weak_table[min(ps)].value

    mins: list[float] = [min(weak_table[p].raw_times) for p in ps]
    maxs: list[float] = [max(weak_table[p].raw_times) for p in ps]
    err_lo: list[float] = [t - lo for t, lo in zip(times, mins)]
    err_hi: list[float] = [hi - t for t, hi in zip(times, maxs)]

    ax1.errorbar(
        ps, times,
        yerr=[err_lo, err_hi],
        fmt="o-", color=PALETTE[0], markersize=6, linewidth=1.5,
        capsize=4, capthick=1.2, label="Execution time", zorder=4,
    )
    ax1.axhline(y=t_base, color="grey", linestyle="--", linewidth=1.0,
                alpha=0.6, label=f"Ideal (constant = {t_base:.3f} s)", zorder=2)

    ax1.axvline(x=np.sqrt(32.0 * 64.0), color="grey", linestyle=":",
                linewidth=0.8, alpha=0.5, zorder=2)
    ax1.text(np.sqrt(32.0 * 64.0), max(times) * 0.98,
             "single-node | multi-node",
             fontsize=6, color="grey", ha="center", rotation=90, va="top")

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ps)
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.set_ylim(0, max(maxs) * 1.15)
    ax1.set_xlabel("Total cores (p)", fontsize=9)
    ax1.set_ylabel("Execution time (s)", fontsize=9, color=PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])

    ax2: plt.Axes = ax1.twinx()
    efficiencies: list[float] = [t_base / t for t in times]
    ax2.plot(ps, efficiencies, "s--", color=PALETTE[1], markersize=5,
             linewidth=1.2, label="Weak efficiency $E_w$", zorder=3)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Weak efficiency $E_w(p) = T(p_{min})/T(p)$", fontsize=9,
                   color=PALETTE[1])
    ax2.tick_params(axis="y", labelcolor=PALETTE[1])
    ax2.spines["right"].set_visible(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
               loc="center left", framealpha=0.8)

    save_figure(fig, "fig3_weak_scaling")


def figure4_config(
    hyb_table: dict[str, dict[str, AggregatedCell]],
) -> None:
    """
    Figure 4: configuration study for A_1M. Panel (a) compares 4x8 vs 8x4
    rank/thread splits at p=32, 64, 128. Panel (b) compares packed vs spread
    layouts at p=32 (1n_8x4 vs 2n_4x4 vs 4n_2x4). Bars are annotated when
    the aggregation fell back to the minimum.
    """
    print("Figure 4: Configuration study")
    test_id: str = "A_1M"
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 4.5), sharey=True)
    setup_axes(ax_a)
    setup_axes(ax_b)

    panel_a_configs: dict[int, dict[str, str]] = {
        32: {"4x8": "hyb_1n_4x8", "8x4": "hyb_1n_8x4"},
        64: {"4x8": "hyb_2n_4x8", "8x4": "hyb_2n_8x4"},
        128: {"4x8": "hyb_4n_4x8", "8x4": "hyb_4n_8x4"},
    }

    x_labels_a: list[str] = []
    vals_4x8: list[float] = []
    vals_8x4: list[float] = []
    flags_4x8: list[bool] = []
    flags_8x4: list[bool] = []

    for p_val in [32, 64, 128]:
        cfg_map: dict[str, str] = panel_a_configs[p_val]
        x_labels_a.append(f"p={p_val}")

        cell_4x8: AggregatedCell = hyb_table[cfg_map["4x8"]][test_id]
        cell_8x4: AggregatedCell = hyb_table[cfg_map["8x4"]][test_id]
        vals_4x8.append(cell_4x8.value)
        vals_8x4.append(cell_8x4.value)
        flags_4x8.append(cell_4x8.used_minimum)
        flags_8x4.append(cell_8x4.used_minimum)

    x_pos: np.ndarray = np.arange(len(x_labels_a))
    bar_width: float = 0.35

    bars1 = ax_a.bar(x_pos - bar_width / 2, vals_4x8, bar_width,
                     color=PALETTE[0], label="4x8 (4 ranks, 8 threads)")
    bars2 = ax_a.bar(x_pos + bar_width / 2, vals_8x4, bar_width,
                     color=PALETTE[1], label="8x4 (8 ranks, 4 threads)")

    for i, (bar, flagged) in enumerate(zip(bars1, flags_4x8)):
        if flagged:
            ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      "*", ha="center", va="bottom", fontsize=10, color="red")
    for i, (bar, flagged) in enumerate(zip(bars2, flags_8x4)):
        if flagged:
            ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      "*", ha="center", va="bottom", fontsize=10, color="red")

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(x_labels_a)
    ax_a.set_ylabel("Execution time (s)", fontsize=9)
    ax_a.set_title("(a) Rank/thread split", fontsize=9)
    ax_a.legend(fontsize=7, framealpha=0.8)

    panel_b_configs: list[tuple[str, str]] = [
        ("hyb_1n_8x4", "1n (packed)"),
        ("hyb_2n_4x4", "2n (spread)"),
        ("hyb_4n_2x4", "4n (spread)"),
    ]

    x_labels_b: list[str] = []
    vals_b: list[float] = []
    flags_b: list[bool] = []
    colours_b: list[str] = []

    for i, (config, label) in enumerate(panel_b_configs):
        x_labels_b.append(label)
        cell: AggregatedCell = hyb_table[config][test_id]
        vals_b.append(cell.value)
        flags_b.append(cell.used_minimum)
        colours_b.append(PALETTE[i + 2])

    x_pos_b: np.ndarray = np.arange(len(x_labels_b))
    bars_b = ax_b.bar(x_pos_b, vals_b, 0.5, color=colours_b)

    for bar, flagged in zip(bars_b, flags_b):
        if flagged:
            ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      "*", ha="center", va="bottom", fontsize=10, color="red")

    ax_b.set_xticks(x_pos_b)
    ax_b.set_xticklabels(x_labels_b, fontsize=8)
    ax_b.set_title("(b) Packed vs spread at p=32", fontsize=9)

    fig.suptitle("", fontsize=0)
    fig.text(0.5, -0.02,
             "* = aggregation used minimum (contention detected)", fontsize=7,
             ha="center", color="red")
    fig.tight_layout()
    save_figure(fig, "fig4_config_study")


def figure5_cuda(
    cuda_table: dict[str, dict[str, dict[str, AggregatedCell]]],
    seq_baselines: dict[str, AggregatedCell],
) -> dict[str, object]:
    """Two standalone column-width figures with the original graphical style.

    The time plot includes only fully correct configurations with complete layer
    coverage. The speedup plot chooses the smallest eligible median, not a minimum
    repetition. Open squares separate corrected node122 reruns from node111.
    Missing bars distinguish failed/unverified from below-floor measurements.
    """
    print("Figure 5: CUDA (correctness-filtered medians)")
    fig_a, ax_a = plt.subplots(figsize=(4.6, 3.6))
    fig_b, ax_b = plt.subplots(figsize=(4.6, 3.6))
    setup_axes(ax_a)
    setup_axes(ax_b)
    block_sizes = [32, 64, 128, 256, 512, 1024]
    ref_test = "A_1M"
    audit: dict[str, object] = {"time_series": {}, "selected_speedups": {},
                              "sequential_baselines": {}}
    for branch in CUDA_BRANCHES:
        cells = [(bs, cuda_table.get(branch, {}).get(f"cuda_bs{bs}", {}).get(ref_test))
                 for bs in block_sizes]
        cells = [(bs, cell) for bs, cell in cells
                 if cell is not None and not cell.below_floor]
        if not cells:
            continue
        colour = BRANCH_COLOURS[branch]
        # The connecting line is a visual guide; source allocation is encoded
        # separately and disclosed in the figure note and machine-readable audit.
        ax_a.plot([bs for bs, cell in cells], [cell.value for bs, cell in cells],
                  "-", color=colour, linewidth=1.2, label=BRANCH_LABELS[branch], zorder=3)
        for bs, cell in cells:
            rerun = any(row.nodelist == "node122" for row in cell.source_rows)
            ax_a.errorbar(bs, cell.value,
                          yerr=[[cell.value-min(cell.raw_times)],
                                [max(cell.raw_times)-cell.value]],
                          fmt="s" if rerun else "o", color=colour,
                          markerfacecolor="white" if rerun else colour,
                          markersize=5, capsize=3, linewidth=1.0, zorder=4)
            audit["time_series"][f"{branch}|cuda_bs{bs}|{ref_test}"] = asdict(cell)
    ax_a.set_xscale("log", base=2)
    ax_a.set_yscale("log")
    ax_a.set_xticks(block_sizes)
    ax_a.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_a.set_xlabel("Thread-block size", fontsize=9)
    ax_a.set_ylabel(f"Execution time (s) [{ref_test}]", fontsize=9)
    ax_a.set_title("Time vs block size", fontsize=9)
    ax_a.legend(fontsize=7, framealpha=0.8)

    cuda_tests = [t for t in SERIES_TESTS
                  if t in seq_baselines and not seq_baselines[t].below_floor]
    n_branches = len(CUDA_BRANCHES)
    x_pos = np.arange(len(cuda_tests))
    bar_width = 0.8 / n_branches
    missing = []
    for i, branch in enumerate(CUDA_BRANCHES):
        colour = BRANCH_COLOURS[branch]
        offset = (i - n_branches / 2 + 0.5) * bar_width
        legend_added = False
        for x, test_id in zip(x_pos, cuda_tests):
            all_cells = [(bs, cuda_table.get(branch, {}).get(f"cuda_bs{bs}", {}).get(test_id))
                         for bs in block_sizes]
            eligible = [(bs, cell) for bs, cell in all_cells
                        if cell is not None and not cell.below_floor]
            if not eligible:
                reason = "<floor" if any(cell is not None for bs, cell in all_cells) else "invalid"
                missing.append((x + offset, reason, colour))
                audit["selected_speedups"][f"{branch}|{test_id}"] = {"omitted": reason}
                continue
            bs, cell = min(eligible, key=lambda pair: (pair[1].value, pair[0]))
            baseline = seq_baselines[test_id]
            speedup = baseline.value / cell.value
            low = min(baseline.raw_times) / max(cell.raw_times)
            high = max(baseline.raw_times) / min(cell.raw_times)
            ax_b.bar(x + offset, speedup, bar_width, color=colour,
                     label=BRANCH_LABELS[branch] if not legend_added else None, zorder=3)
            legend_added = True
            ax_b.errorbar(x + offset, speedup, yerr=[[speedup-low], [high-speedup]],
                          fmt="none", color="#333333", capsize=2, linewidth=.8, zorder=4)
            ax_b.annotate(f"b={bs}", (x+offset, high), xytext=(0, 4),
                          textcoords="offset points", ha="center", va="bottom",
                          rotation=90, fontsize=6.5, color=colour)
            audit["selected_speedups"][f"{branch}|{test_id}"] = {
                "block_size": bs, "speedup": speedup, "range_bounds": [low, high],
                "cuda_cell": asdict(cell), "sequential_cell": asdict(baseline)}
    for x, reason, colour in missing:
        ax_b.text(x, .015, reason, transform=ax_b.get_xaxis_transform(),
                  rotation=90, fontsize=6.5, color=colour, ha="center", va="bottom")
    for test_id in cuda_tests:
        audit["sequential_baselines"][test_id] = asdict(seq_baselines[test_id])
    ax_b.set_xticks(x_pos)
    ax_b.set_xlim(-.5, len(cuda_tests)-.5)
    ax_b.set_xticklabels(cuda_tests, fontsize=7, rotation=15)
    ax_b.set_ylabel("Speedup vs T_serial", fontsize=9)
    ax_b.set_title("Best eligible block-size speedup", fontsize=9)
    ax_b.set_ylim(bottom=0, top=ax_b.get_ylim()[1]*1.30)
    ax_b.legend(fontsize=7, framealpha=0.8, ncol=2)
    # Provenance and omission conventions live in the report captions and audit
    # instead of a wide shared footer that would shrink text in a single column.
    fig_a.tight_layout()
    fig_b.tight_layout()
    save_figure(fig_a, "fig5_cuda_time")
    save_figure(fig_b, "fig5_cuda_speedup")
    return audit


def parse_arguments() -> argparse.Namespace:
    """
    Declares the command-line interface: optional --figure N to render a single
    figure, otherwise all figures are produced.
    """
    parser = argparse.ArgumentParser(
        description="Generate figures for the PEM multicore report."
    )
    parser.add_argument(
        "--figure", type=int, default=None, choices=[1, 2, 3, 4, 5],
        help="Generate only the specified figure (1-5); omit for all.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Entry point: loads all data, builds tables, generates the requested
    figures, and prints derived numbers to stdout and JSON.
    """
    args: argparse.Namespace = parse_arguments()

    print("Loading raw data...")
    raw_rows = load_raw_rows()
    print(f"  {len(raw_rows)} raw rows loaded")

    hyb_table = build_hyb_table(raw_rows)
    seq_baselines = build_seq_baselines(raw_rows)
    par1_baselines = build_par1_baselines(hyb_table)
    cuda_table = build_cuda_table(raw_rows)

    print(f"  SEQ baselines: {sorted(seq_baselines.keys())}")
    print(f"  T_par(1) baselines: {sorted(par1_baselines.keys())}")
    print(f"  HYB configs: {len(hyb_table)}")
    print(f"  CUDA branches: {sorted(cuda_table.keys())}")

    weak_rows = load_weak_rows()
    weak_table = build_weak_table(weak_rows)
    print(f"  Weak scaling: {len(weak_rows)} rows, p = {sorted(weak_table.keys())}")

    figures: list[int] = [args.figure] if args.figure else [1, 2, 3, 4, 5]

    for fig_num in figures:
        print()
        if fig_num == 1:
            figure1_speedup(hyb_table, seq_baselines)
        elif fig_num == 2:
            figure2_efficiency(hyb_table, par1_baselines)
        elif fig_num == 3:
            figure3_weak(weak_table)
        elif fig_num == 4:
            figure4_config(hyb_table)
        elif fig_num == 5:
            audit = figure5_cuda(cuda_table, seq_baselines)
            audit["policy"] = {
                "statistic": "median", "variability": "observed min/max, not confidence intervals",
                "floor_seconds": 0.02, "cuda_cell": "all recorded repetitions valid, same allocation/build",
                "baseline": "largest valid cluster allocation sample; earliest job id breaks ties",
                "small_blocks": "only logged fixed cuda-1D/cuda-shared commits and matching binary digest",
                "main": "historical, not a repeated validation of final submission",
            }
            excluded = [row for row in raw_rows if row.section == "CUDA" and cuda_exclusion_reason(row)]
            audit["excluded_rows_by_reason"] = dict(Counter(cuda_exclusion_reason(row) for row in excluded))
            audit["excluded_rows"] = [dict(asdict(row), reason=cuda_exclusion_reason(row)) for row in excluded]
            with (OUTPUT_DIR / "fig5_cuda_audit.json").open("w") as handle:
                json.dump(audit, handle, indent=2)
            print("  -> fig5_cuda_audit.json")

    if args.figure == 5:
        return 0  # A CUDA-only rebuild must not overwrite unrelated summaries.

    print("\n--- Derived numbers ---")
    numbers = compute_derived_numbers(seq_baselines, par1_baselines, hyb_table)
    for test_id, entry in sorted(numbers.items()):
        print(f"\n  {test_id}:")
        for key, val in entry.items():
            print(f"    {key}: {val}")

    json_path: Path = OUTPUT_DIR / "derived_numbers.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(numbers, handle, indent=2, ensure_ascii=False)
    print(f"\n  -> {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
