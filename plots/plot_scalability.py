#!/usr/bin/env python3
"""
Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International

Parses the textual output produced by scalability.sbatch and draws the
problem-size scalability curve of the CUDA version of the high-energy particle
storms simulation, placing the layer size on the x axis and the mean wall clock
time on the y axis. Benchmarks of a single storm file and benchmarks of a whole
suite with its waves interleaved are drawn as two distinct series, and the
standard deviation collected over the repetitions is reported as the vertical
error bar of every point.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

BENCHMARK_HEADER: re.Pattern[str] = re.compile(
    r"^--- (?P<label>.+) \(size (?P<size>\d+), (?P<waves>\d+) storm file\(s\)\) ---$"
)
MEAN_LINE: re.Pattern[str] = re.compile(r"^Tempo medio:\s+(?P<mean>[0-9eE.+-]+) s$")
STDDEV_LINE: re.Pattern[str] = re.compile(r"^Std dev:\s+(?P<stddev>[0-9eE.+-]+) s$")

COLOUR_SINGLE: str = "#2a78d6"
COLOUR_COMBINED: str = "#eb6834"
COLOUR_SURFACE: str = "#fcfcfb"
COLOUR_INK: str = "#0b0b0b"
COLOUR_INK_SECONDARY: str = "#52514e"
COLOUR_MUTED: str = "#898781"
COLOUR_GRID: str = "#e1e0d9"
COLOUR_AXIS: str = "#c3c2b7"

LABEL_TITLE: str = "Scalabilita rispetto alla dimensione del problema (CUDA)"
LABEL_X: str = "Dimensione del layer (posizioni)"
LABEL_Y: str = "Tempo medio di esecuzione (s)"
LABEL_SINGLE: str = "File singolo"
LABEL_COMBINED: str = "Suite combinata"
LABEL_TREND: str = "Media per dimensione"

MISSING_MATPLOTLIB: str = (
    "matplotlib non e' installato.\n"
    "Su Arch Linux:  sudo pacman -S python-matplotlib\n"
    "In un venv:     python3 -m venv .venv && .venv/bin/pip install matplotlib"
)


@dataclass(frozen=True)
class Measurement:
    """
    A single benchmark entry recovered from the batch output, holding the name
    of the test it refers to, the layer size it was executed with, how many
    storm files were interleaved in the execution and the mean and standard
    deviation of the wall clock times collected over the repetitions.
    """

    label: str
    size: int
    waves: int
    mean: float
    stddev: float

    @property
    def is_combined(self) -> bool:
        """
        Tells whether the entry comes from a combined suite benchmark, which is
        the case whenever more than one storm file took part in the execution,
        since the single file benchmarks always run exactly one wave file.
        """
        return self.waves > 1


def parse_measurements(text: str) -> list[Measurement]:
    """
    Extracts every benchmark entry from the output of scalability.sbatch by
    reading each benchmark header together with the mean and standard deviation
    lines that follow it. Blocks left incomplete by an aborted or still running
    job are discarded, so a partial output still produces a usable plot.
    """
    measurements: list[Measurement] = []
    label: str | None = None
    size: int = 0
    waves: int = 0
    mean: float | None = None

    for raw_line in text.splitlines():
        line: str = raw_line.rstrip()

        header: re.Match[str] | None = BENCHMARK_HEADER.match(line)
        if header is not None:
            label = header.group("label")
            size = int(header.group("size"))
            waves = int(header.group("waves"))
            mean = None
            continue

        if label is None:
            continue

        mean_match: re.Match[str] | None = MEAN_LINE.match(line)
        if mean_match is not None:
            mean = float(mean_match.group("mean"))
            continue

        stddev_match: re.Match[str] | None = STDDEV_LINE.match(line)
        if stddev_match is not None and mean is not None:
            measurements.append(
                Measurement(label, size, waves, mean, float(stddev_match.group("stddev")))
            )
            label = None
            mean = None

    return measurements


def load_measurements(paths: list[Path]) -> list[Measurement]:
    """
    Reads and parses every batch output file given on the command line, merging
    all the entries into a single series sorted by layer size so that repeated
    executions of the sweep can be plotted together on the same axes.
    """
    measurements: list[Measurement] = []
    for path in paths:
        measurements.extend(parse_measurements(path.read_text(encoding="utf-8", errors="replace")))
    return sorted(measurements, key=lambda entry: (entry.size, entry.label))


def mean_per_size(measurements: list[Measurement]) -> tuple[list[int], list[float]]:
    """
    Collapses the entries sharing the same layer size into a single value by
    averaging their mean times, producing the trend line that shows how the
    execution time evolves with the problem size when several test files are
    benchmarked at the same size.
    """
    totals: dict[int, list[float]] = {}
    for entry in measurements:
        totals.setdefault(entry.size, []).append(entry.mean)

    sizes: list[int] = sorted(totals)
    return sizes, [sum(totals[size]) / len(totals[size]) for size in sizes]


def write_csv(measurements: list[Measurement], path: Path) -> None:
    """
    Dumps the parsed entries as a comma separated table, so that the same data
    feeding the chart can be checked by hand or reused in a report without
    parsing the batch output a second time.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test", "size", "waves", "mean_s", "stddev_s"])
        for entry in measurements:
            writer.writerow([entry.label, entry.size, entry.waves, entry.mean, entry.stddev])


def build_plot(
    measurements: list[Measurement],
    output_path: Path,
    use_log_scale: bool,
    show_trend: bool,
    annotate: bool,
) -> None:
    """
    Draws the scalability chart with the layer size on the x axis and the mean
    wall clock time on the y axis. Single file and combined suite benchmarks
    form two series told apart by both colour and marker shape, the standard
    deviation of every entry becomes a vertical error bar and, unless disabled,
    a dashed line joins the average time measured at each distinct layer size.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(MISSING_MATPLOTLIB)

    singles: list[Measurement] = [entry for entry in measurements if not entry.is_combined]
    combined: list[Measurement] = [entry for entry in measurements if entry.is_combined]

    figure, axes = plt.subplots(figsize=(9.0, 6.0), facecolor=COLOUR_SURFACE)
    axes.set_facecolor(COLOUR_SURFACE)

    handles: list[object] = []

    for entries, colour, marker, series_label in (
        (singles, COLOUR_SINGLE, "o", LABEL_SINGLE),
        (combined, COLOUR_COMBINED, "s", LABEL_COMBINED),
    ):
        if not entries:
            continue
        series = axes.errorbar(
            [entry.size for entry in entries],
            [entry.mean for entry in entries],
            yerr=[entry.stddev for entry in entries],
            fmt=marker,
            color=colour,
            markersize=8,
            markeredgecolor=COLOUR_SURFACE,
            markeredgewidth=1.0,
            elinewidth=1.4,
            capsize=3,
            linestyle="none",
            label=series_label,
        )
        handles.append(series)

    if show_trend and singles:
        trend_sizes, trend_times = mean_per_size(singles)
        trend = axes.plot(
            trend_sizes,
            trend_times,
            color=COLOUR_SINGLE,
            linewidth=2.0,
            linestyle="--",
            alpha=0.55,
            zorder=1,
            label=LABEL_TREND,
        )
        handles.append(trend[0])

    if annotate:
        for entry in measurements:
            axes.annotate(
                entry.label,
                (entry.size, entry.mean),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=7,
                color=COLOUR_INK_SECONDARY,
            )

    if use_log_scale:
        axes.set_xscale("log")
        axes.set_yscale("log")

    axes.set_title(LABEL_TITLE, color=COLOUR_INK, fontsize=13, pad=14)
    axes.set_xlabel(LABEL_X, color=COLOUR_INK_SECONDARY, fontsize=11)
    axes.set_ylabel(LABEL_Y, color=COLOUR_INK_SECONDARY, fontsize=11)
    axes.tick_params(colors=COLOUR_MUTED, labelsize=9)
    axes.grid(True, which="both", color=COLOUR_GRID, linewidth=0.6, linestyle=":", zorder=0)
    axes.set_axisbelow(True)

    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(COLOUR_AXIS)

    legend = axes.legend(handles=handles, frameon=False, fontsize=10, loc="upper left")
    for text in legend.get_texts():
        text.set_color(COLOUR_INK_SECONDARY)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, facecolor=COLOUR_SURFACE)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    """
    Declares and reads the command line interface of the tool, accepting one or
    more batch output files and the options controlling the appearance of the
    resulting chart and the optional dump of the parsed table.
    """
    parser = argparse.ArgumentParser(
        description="Disegna la curva di scalabilita a partire dall'output di scalability.sbatch."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="file .out prodotti da scalability.sbatch",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("scalability.png"),
        help="file dell'immagine da generare, l'estensione sceglie il formato (default: scalability.png)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="scrive anche i dati estratti in un file CSV",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="usa assi lineari invece della scala logaritmica su entrambi gli assi",
    )
    parser.add_argument(
        "--no-trend",
        action="store_true",
        help="non disegna la linea di tendenza sulle medie per dimensione",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="scrive il nome del test accanto a ogni punto",
    )
    return parser.parse_args()


def main() -> int:
    """
    Entry point of the tool: loads the measurements from the requested batch
    output files, optionally dumps them as CSV, renders the chart and reports
    on standard output how many entries were plotted. Returns a non zero status
    when no benchmark entry could be recovered, since in that case the chart
    would be empty and the batch output is most likely not the expected one.
    """
    arguments: argparse.Namespace = parse_arguments()

    missing: list[Path] = [path for path in arguments.inputs if not path.is_file()]
    if missing:
        print(f"File non trovati: {', '.join(str(path) for path in missing)}", file=sys.stderr)
        return 1

    measurements: list[Measurement] = load_measurements(arguments.inputs)
    if not measurements:
        print(
            "Nessuna misura trovata. Il file passato e' davvero l'output di scalability.sbatch?",
            file=sys.stderr,
        )
        return 1

    if arguments.csv is not None:
        write_csv(measurements, arguments.csv)
        print(f"Dati scritti in {arguments.csv}")

    build_plot(
        measurements,
        arguments.output,
        not arguments.linear,
        not arguments.no_trend,
        arguments.annotate,
    )

    sizes: set[int] = {entry.size for entry in measurements}
    print(f"Grafico scritto in {arguments.output}")
    print(f"{len(measurements)} misure su {len(sizes)} dimensioni distinte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
