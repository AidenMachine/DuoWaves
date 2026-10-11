"""
Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International

Shared data loading, parsing and aggregation layer for the PEM multicore
report plotting scripts. All CSV parsing uses csv.reader to handle the
quoted nodelist field that contains commas.
"""

from __future__ import annotations

import csv
import re
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from statistics import median

TIMING_RESOLUTION_FLOOR: float = 0.02

RAW_DIR: Path = Path(__file__).resolve().parent.parent / "results" / "raw"
WEAK_DIR: Path = Path(__file__).resolve().parent.parent / "results" / "weak"
OUTPUT_DIR: Path = Path(__file__).resolve().parent

HYB_CONFIG_RE: re.Pattern[str] = re.compile(
    r"hyb_(?P<nodes>\d+)n_(?P<ppn>\d+)x(?P<threads>\d+)"
)
CUDA_CONFIG_RE: re.Pattern[str] = re.compile(r"cuda_bs(?P<block_size>\d+)")


@dataclass(frozen=True)
class RawRow:
    """
    One measurement row from the strong-scaling CSV campaign, parsed from the
    pipe-delimited key and the remaining comma-separated fields.
    """

    section: str
    branch: str
    config: str
    test_id: str
    run: int
    time: float
    correctness: str
    job_id: str
    nodelist: str
    timestamp: str
    md5: str
    result: str


@dataclass(frozen=True)
class WeakRow:
    """
    One measurement row from the weak-scaling CSV files, with fields
    p, layer_size, ppn, threads, run, time, correctness, md5, nodelist.
    """

    p: int
    layer_size: int
    ppn: int
    threads: int
    run: int
    time: float
    correctness: str
    md5: str
    nodelist: str


@dataclass
class AggregatedCell:
    """
    Median timing and original samples. Below-floor cells are retained for
    auditing but must not be used in speedup rankings.
    """

    value: float
    used_minimum: bool = False
    below_floor: bool = False
    raw_times: list[float] = field(default_factory=list)
    source_rows: list[RawRow] = field(default_factory=list)


def load_raw_rows(directory: Path | None = None) -> list[RawRow]:
    """
    Reads every CSV file under the raw results directory and returns a list of
    RawRow objects. Uses csv.reader to handle the quoted nodelist field that may
    contain commas inside the quotes.
    """
    if directory is None:
        directory = RAW_DIR

    rows: list[RawRow] = []
    for csv_path in sorted(directory.glob("*.csv")):
        if csv_path.stat().st_size == 0:
            continue
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for fields in reader:
                if len(fields) < 6:
                    continue
                key_parts: list[str] = fields[0].split("|")
                if len(key_parts) < 5:
                    continue
                try:
                    rows.append(RawRow(
                        section=key_parts[0],
                        branch=key_parts[1],
                        config=key_parts[2],
                        test_id=key_parts[3],
                        run=int(key_parts[4]),
                        time=float(fields[1]),
                        correctness=fields[2],
                        job_id=fields[3],
                        nodelist=fields[4],
                        timestamp=fields[5] if len(fields) > 5 else "",
                        md5=fields[6] if len(fields) > 6 else "",
                        result=fields[7] if len(fields) > 7 else "",
                    ))
                except (ValueError, IndexError):
                    continue
    return rows


def load_weak_rows(directory: Path | None = None) -> list[WeakRow]:
    """
    Reads every weak_*.csv file under the weak results directory and returns
    a list of WeakRow objects. Uses csv.reader for safe parsing.
    """
    if directory is None:
        directory = WEAK_DIR

    rows: list[WeakRow] = []
    for csv_path in sorted(directory.glob("weak_*.csv")):
        if csv_path.stat().st_size == 0:
            continue
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for fields in reader:
                if len(fields) < 9:
                    continue
                try:
                    rows.append(WeakRow(
                        p=int(fields[0]),
                        layer_size=int(fields[1]),
                        ppn=int(fields[2]),
                        threads=int(fields[3]),
                        run=int(fields[4]),
                        time=float(fields[5]),
                        correctness=fields[6],
                        md5=fields[7],
                        nodelist=fields[8],
                    ))
                except (ValueError, IndexError):
                    continue
    return rows


def parse_hyb_config(config: str) -> tuple[int, int, int, int] | None:
    """
    Parses a hybrid config string like 'hyb_2n_4x8' and returns
    (nodes, ppn, threads, total_cores). Returns None if the string does
    not match the expected grammar.
    """
    match: re.Match[str] | None = HYB_CONFIG_RE.match(config)
    if match is None:
        return None
    nodes: int = int(match.group("nodes"))
    ppn: int = int(match.group("ppn"))
    threads: int = int(match.group("threads"))
    return nodes, ppn, threads, nodes * ppn * threads


def parse_cuda_config(config: str) -> int | None:
    """
    Parses a CUDA config string like 'cuda_bs256' and returns the block size
    as an integer. Returns None if the string does not match.
    """
    match: re.Match[str] | None = CUDA_CONFIG_RE.match(config)
    if match is None:
        return None
    return int(match.group("block_size"))


def aggregate(times: list[float]) -> AggregatedCell:
    """
    Always use the median, including noisy allocations; retain every sample.
    The documented 0.02 s floor is a reporting policy, not a measured timer
    resolution. used_minimum stays false for compatibility with other plots.
    """
    assert len(times) > 0, "aggregate() called with empty times list"

    med: float = median(times)
    below_floor: bool = med < TIMING_RESOLUTION_FLOOR
    return AggregatedCell(
        value=med,
        below_floor=below_floor,
        raw_times=sorted(times),
    )


def build_hyb_table(
    rows: list[RawRow],
    branch: str = "Add_SIMD_Directives",
) -> dict[str, dict[str, AggregatedCell]]:
    """
    Groups HYB rows by (config, test_id) and aggregates each cell. Returns a
    nested dict: config -> test_id -> AggregatedCell. Asserts that every cell
    has exactly 5 repetitions.
    """
    buckets: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.section != "HYB" or row.branch != branch or row.correctness != "PASS":
            continue
        key: tuple[str, str] = (row.config, row.test_id)
        buckets.setdefault(key, []).append(row.time)

    table: dict[str, dict[str, AggregatedCell]] = {}
    for (config, test_id), times in buckets.items():
        if len(times) != 5:
            print(f"WARNING: ({config}, {test_id}) has {len(times)} rows, expected 5")
        table.setdefault(config, {})[test_id] = aggregate(times)
    return table


def build_seq_baselines(rows: list[RawRow]) -> dict[str, AggregatedCell]:
    """
    Use valid cluster references, keeping independent allocations separate.
    Select the allocation with the most repetitions; ties use earliest job id,
    never fastest runtime. BASELINE denotes the supplied sequential reference.
    Retain source rows so the selected baseline is explicit in the audit JSON.
    """
    allocations: dict[str, dict[tuple[str, str, str], list[RawRow]]] = {}
    for row in rows:
        if (row.section != "SEQ" or row.correctness not in {"PASS", "BASELINE"}
                or row.job_id.lower() == "local" or row.nodelist.lower() == "local"
                or not row.job_id.isdigit()):
            continue
        key = (row.job_id, row.nodelist, row.md5)
        allocations.setdefault(row.test_id, {}).setdefault(key, []).append(row)
    baselines: dict[str, AggregatedCell] = {}
    for test_id, jobs in allocations.items():
        selected = min(jobs.values(), key=lambda group: (-len(group), int(group[0].job_id)))
        cell = aggregate([row.time for row in selected])
        cell.source_rows = selected
        baselines[test_id] = cell
    return baselines


def build_par1_baselines(
    hyb_table: dict[str, dict[str, AggregatedCell]],
) -> dict[str, AggregatedCell]:
    """
    Extracts the single-core parallel baseline T_par(1) from the hyb_1n_1x1
    config in the HYB table. Returns test_id -> AggregatedCell.
    """
    config_1x1: str = "hyb_1n_1x1"
    if config_1x1 not in hyb_table:
        return {}
    return dict(hyb_table[config_1x1])


def best_time_per_p(
    hyb_table: dict[str, dict[str, AggregatedCell]],
    test_id: str,
) -> dict[int, AggregatedCell]:
    """
    For a given test, finds the best (minimum time) configuration at each
    total core count p. Returns p -> AggregatedCell (from the winning config).
    """
    p_best: dict[int, AggregatedCell] = {}
    for config, tests in hyb_table.items():
        parsed = parse_hyb_config(config)
        if parsed is None:
            continue
        p: int = parsed[3]
        if test_id not in tests:
            continue
        cell: AggregatedCell = tests[test_id]
        if cell.below_floor:
            continue
        if p not in p_best or cell.value < p_best[p].value:
            p_best[p] = cell
    return p_best


def build_weak_table(rows: list[WeakRow]) -> dict[int, AggregatedCell]:
    """
    Groups weak-scaling rows by total core count p and aggregates each group.
    Returns p -> AggregatedCell.
    """
    buckets: dict[int, list[float]] = {}
    for row in rows:
        if row.correctness != "PASS":
            continue
        buckets.setdefault(row.p, []).append(row.time)

    return {p: aggregate(times) for p, times in buckets.items()}


@lru_cache(maxsize=None)
def corrected_cuda_build(branch: str, job_id: str, md5: str) -> bool:
    """Recognize the small-block reruns by their logged fixed commit and binary.

    The August main/cuda-first campaign has incomplete relaxation below 256,
    even for accidental PASS peaks. Only the fixed cuda-1D/cuda-shared builds
    are admitted there. New source revisions require an explicit provenance
    audit before extending this policy.
    """
    fixed_commits = {"cuda-1D": "7d86f1e", "cuda-shared": "6b05a7d"}
    if branch not in fixed_commits or not job_id.isdigit() or not md5:
        return False
    path = RAW_DIR.parent.parent / "sbatch_jobs" / job_id / f"{job_id}.out"
    if not path.is_file():
        return False
    log = path.read_text()
    return (re.search(rf"Branch:\s+{re.escape(branch)} \({fixed_commits[branch]}\)", log)
            is not None and re.search(rf"Binary md5:\s+{re.escape(md5)}\b", log) is not None)


def cuda_exclusion_reason(row: RawRow) -> str | None:
    if row.correctness != "PASS":
        return "failed_or_unverified"
    bs = parse_cuda_config(row.config)
    if bs is None:
        return "unknown_configuration"
    if bs < 256 and not corrected_cuda_build(row.branch, row.job_id, row.md5):
        return "incomplete_relaxation_or_unverified_build"
    return None


def build_cuda_table(
    rows: list[RawRow],
) -> dict[str, dict[str, dict[str, AggregatedCell]]]:
    """
    Aggregate correct, structurally valid CUDA runs. Configurations with any
    rejected repetition are omitted altogether to avoid ranking flaky builds.
    Retain input rows for executable/workload/allocation provenance.
    """
    buckets: dict[tuple[str, str, str], list[RawRow]] = {}
    for row in rows:
        if row.section != "CUDA":
            continue
        key: tuple[str, str, str] = (row.branch, row.config, row.test_id)
        buckets.setdefault(key, []).append(row)

    table: dict[str, dict[str, dict[str, AggregatedCell]]] = {}
    for (branch, config, test_id), group in buckets.items():
        if any(cuda_exclusion_reason(row) for row in group):
            continue
        # Do not silently pool independent allocations or executables.
        if len({(row.job_id, row.nodelist, row.md5) for row in group}) != 1:
            raise ValueError(f"Mixed CUDA provenance: {branch}/{config}/{test_id}")
        cell = aggregate([row.time for row in group])
        cell.source_rows = group
        table.setdefault(branch, {}).setdefault(config, {})[test_id] = cell
    return table


def compute_derived_numbers(
    seq_baselines: dict[str, AggregatedCell],
    par1_baselines: dict[str, AggregatedCell],
    hyb_table: dict[str, dict[str, AggregatedCell]],
) -> dict[str, object]:
    """
    Computes the derived numbers listed in the spec (serial optimisation gain,
    parallel speedup at p=128, total speedup, Amdahl serial fraction, parallel
    overhead at p=1) and returns them as a JSON-serialisable dictionary.
    """
    numbers: dict[str, object] = {}

    for test_id in sorted(par1_baselines):
        t_par1: float = par1_baselines[test_id].value
        t_serial: float = seq_baselines[test_id].value if test_id in seq_baselines else 0.0
        p_best: dict[int, AggregatedCell] = best_time_per_p(hyb_table, test_id)

        entry: dict[str, object] = {
            "T_serial": t_serial,
            "T_par1": t_par1,
        }

        if t_serial > 0:
            entry["serial_optimisation_gain"] = round(t_serial / t_par1, 2)

        if 128 in p_best:
            t_128: float = p_best[128].value
            entry["T_128"] = round(t_128, 4)
            entry["parallel_speedup_128"] = round(t_par1 / t_128, 1)
            if t_serial > 0:
                entry["total_speedup_128"] = round(t_serial / t_128, 1)

        if 2 in p_best and t_par1 > 0:
            t_2: float = p_best[2].value
            entry["parallel_overhead_p1"] = f"{t_par1:.2f} vs 2*T(2)={2*t_2:.2f}"
            entry["parallel_overhead_pct"] = round(
                100.0 * (t_par1 - 2 * t_2) / t_par1, 1
            )

        amdahl_f: list[float] = []
        for p, cell in sorted(p_best.items()):
            if p <= 1:
                continue
            s_p: float = t_par1 / cell.value
            f: float = (1.0 / s_p - 1.0 / p) / (1.0 - 1.0 / p)
            amdahl_f.append(f)
        if amdahl_f:
            entry["amdahl_serial_fraction"] = {
                "values": [round(f * 100, 3) for f in amdahl_f],
                "min_pct": round(min(amdahl_f) * 100, 3),
                "max_pct": round(max(amdahl_f) * 100, 3),
                "ceiling_x": [
                    round(1.0 / max(min(amdahl_f), 1e-9), 0),
                    round(1.0 / max(max(amdahl_f), 1e-9), 0),
                ],
            }

        numbers[test_id] = entry

    return numbers
