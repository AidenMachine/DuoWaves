"""
Realized by @luckignolo32 (GitHub) - Creative Commons Attribution-ShareAlike 4.0 International

Builder of the results spreadsheet. It merges every raw CSV file written by
the suite jobs and lays the measurements out as one sheet per section, each
sheet following the block structure of the reference example: Avg time and
Min time side by side, Max time and STD deviation below them, then
Correctness and Runs combined, with one column per branch and one row per
(configuration, test) pair. The spreadsheet is written to
results/test-results.ods. This script runs on the laptop inside the virtual
environment created by setup_venv.sh.

The tests named in EXCLUDED_TESTS, currently the 100M workload of section A,
are left out of every block: their raw measurements stay on disk but produce
no rows.
"""

import csv
import pathlib
import re
import statistics
import sys

from odf.opendocument import OpenDocumentSpreadsheet
from odf.table import Table, TableCell, TableRow
from odf.text import P

EXCLUDED_TESTS = {"A_100M"}

ALL_TESTS = ["A_20k", "A_100k", "A_1M", "A_100M",
             "B_1", "B_3", "B_5", "B_10",
             "C_3", "C_100", "C_1k", "C_5k"]

TESTS = [t for t in ALL_TESTS if t not in EXCLUDED_TESTS]

SHEET_NAMES = {"HYB": "Hybrid", "CUDA": "CUDA", "SEQ": "Sequential"}


def suite_paths():
    """
    TR-4
    Resolves the suite directories from the location of this file, so the
    builder can be invoked from any working directory.
    """
    suite = pathlib.Path(__file__).resolve().parent
    relation = suite.parent.parent
    return suite, relation / "results"


def load_conf(suite_dir):
    """
    TM-SHEETS
    Parses suite.conf, which is kept in plain KEY="value" form precisely so
    that this builder and the bash entry points can share one configuration.
    """
    conf = {}
    text = (suite_dir / "suite.conf").read_text()
    for match in re.finditer(r'^([A-Z_]+)="([^"]*)"$', text, re.MULTILINE):
        conf[match.group(1)] = match.group(2)
    return conf


def config_order(conf, section):
    """
    TM-HYB, TM-CUDA
    Reproduces, for one section, the configuration identifiers in the same
    order the launchers submit them, so the spreadsheet rows follow the
    campaign layout.
    """
    configs = []
    if section == "HYB":
        for nodes in conf["HYBRID_NODE_COUNTS"].split():
            for layout in conf["HYBRID_LAYOUTS"].split():
                configs.append(f"hyb_{nodes}n_{layout}")
    elif section == "CUDA":
        for bs in conf["CUDA_BLOCK_SIZES"].split():
            configs.append(f"cuda_bs{bs}")
    else:
        configs.append("seq")
    return configs


def branch_order(conf, section):
    """
    TM-SHEETS
    Returns the branch columns of one section in the chronological order
    fixed by suite.conf.
    """
    if section == "CUDA":
        return conf["CUDA_BRANCHES"].split()
    if section == "SEQ":
        return [conf["SEQ_BRANCH"]]
    return conf["MPI_BRANCHES"].split()


def run_count(conf, section):
    """
    TM-RUNS, TM-SEQ
    Returns how many repetitions one section is expected to record. The
    sequential section measures every workload once, since it only has to
    publish the deterministic correctness baselines, while the parallel
    sections keep the five repetitions the campaign asks for.
    """
    if section == "SEQ":
        return int(conf["SEQ_RUNS_PER_TEST"])
    return int(conf["RUNS_PER_TEST"])


def load_measurements(results_dir):
    """
    OUT-01
    Merges every raw CSV file into one dictionary mapping the run key
    (section, branch, config, test, run) to its time and correctness
    verdict, reporting rows that cannot be parsed instead of hiding them.
    Measurements of the tests listed in EXCLUDED_TESTS are dropped without a
    warning: they were recorded by the campaign but are deliberately kept out
    of the spreadsheet, so they are neither rows nor anomalies.
    """
    data = {}
    raw = results_dir / "raw"
    if not raw.is_dir():
        sys.exit(f"no raw results found at {raw}")
    for path in sorted(raw.glob("*.csv")):
        with path.open(newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 3 or row[0].count("|") != 4:
                    print(f"warning: skipping malformed row in {path.name}: {row}")
                    continue
                section, branch, config, test, run = row[0].split("|")
                if test in EXCLUDED_TESTS:
                    continue
                key = (section, branch, config, test, int(run))
                if key in data and data[key][0] != row[1]:
                    print(f"warning: duplicate measurement for {row[0]}, keeping the first")
                    continue
                data[key] = (row[1], row[2])
    return data


def cell(value=None, text=None):
    """
    OUT-02
    Builds one spreadsheet cell, numeric when a value is given and textual
    otherwise, so times stay usable in spreadsheet formulas.
    """
    if value is not None:
        c = TableCell(valuetype="float", value=value)
        c.addElement(P(text=str(value)))
        return c
    c = TableCell(valuetype="string")
    if text:
        c.addElement(P(text=text))
    return c


def runs_of(data, section, branch, config, test, runs_per_test):
    """
    TM-RUNS
    Collects the recorded repetitions of one (branch, configuration, test)
    cell as parallel lists of times and verdicts, in repetition order.
    """
    times, verdicts = [], []
    for run in range(1, runs_per_test + 1):
        entry = data.get((section, branch, config, test, run))
        if entry:
            times.append(float(entry[0]))
            verdicts.append(entry[1])
    return times, verdicts


def correctness_text(times, verdicts, runs_per_test):
    """
    TM-CORRECTNESS
    Summarizes the verdicts of one cell: PASS only when all expected
    repetitions were recorded and passed, FAIL as soon as one repetition
    failed, MISSING_BASELINE when no baseline was available, and an explicit
    coverage marker when repetitions are missing.
    """
    if not verdicts:
        return ""
    if "FAIL" in verdicts:
        return f"FAIL ({verdicts.count('FAIL')}/{len(verdicts)})"
    if "MISSING_BASELINE" in verdicts:
        return "MISSING_BASELINE"
    if len(verdicts) < runs_per_test:
        return f"PASS ({len(verdicts)}/{runs_per_test})"
    return "PASS" if all(v in ("PASS", "BASELINE") for v in verdicts) else "?"


def stat_block_rows(title, branches, rows, aggregate):
    """
    OUT-02
    Builds the rows of one statistics block: a header row with the block
    title followed by the branch names, then one row per (configuration,
    test) pair whose cells hold the aggregate of the recorded times.
    """
    header = [cell(text=title)] + [cell(text=b) for b in branches]
    body = []
    for label, per_branch in rows:
        cells = [cell(text=label)]
        for times in per_branch:
            cells.append(cell(value=round(aggregate(times), 6)) if times else cell())
        body.append(cells)
    return [header] + body


def resolution_text(times, floor):
    """
    MET-01
    Marks a cell whose median repetition falls under the timing resolution
    floor, where the number recorded is the process built around the program
    rather than the program. C_3 and C_100 reach it on the wider layouts,
    with repetitions of 0.00 and 0.01 seconds whose ratios mean nothing.
    """
    if not times:
        return ""
    return "below floor" if statistics.median(times) < floor else ""


def flag_block_rows(title, branches, rows, flag):
    """
    OUT-02
    Builds the rows of one textual block, laid out exactly like a statistics
    block but holding a marker per cell instead of an aggregate.
    """
    header = [cell(text=title)] + [cell(text=b) for b in branches]
    body = []
    for label, per_branch in rows:
        cells = [cell(text=label)]
        for times in per_branch:
            cells.append(cell(text=flag(times)))
        body.append(cells)
    return [header] + body


def paste_blocks(sheet, left_rows, right_rows, gap=1):
    """
    OUT-02
    Writes two blocks side by side on the sheet, separated by one empty
    column, mirroring the reference example layout.
    """
    width = max(len(r) for r in left_rows)
    for i in range(max(len(left_rows), len(right_rows))):
        row = TableRow()
        left = left_rows[i] if i < len(left_rows) else []
        right = right_rows[i] if i < len(right_rows) else []
        for j in range(width + gap):
            if j < len(left):
                row.addElement(left[j])
            else:
                row.addElement(cell())
        for c in right:
            row.addElement(c)
        sheet.addElement(row)
    sheet.addElement(TableRow())


def build_sheet(doc, conf, data, section):
    """
    TM-SHEETS
    Builds the sheet of one section: the title row, the five statistics
    blocks, the resolution block, the correctness block and the runs combined
    block listing every single repetition, with one column per branch of the
    section.

    The median block comes first because it, and not the average, is the
    figure representing a cell. The cluster is shared, and a repetition
    occasionally costs many times what its siblings cost: a cell of five
    repetitions around 10.3 s holding one of 310 s is reported by its average
    as a 70 s cell. The average keeps its block right next to the median, so
    the distance between the two stays readable wherever contention disturbed
    the measurement.
    """
    branches = branch_order(conf, section)
    configs = config_order(conf, section)
    runs_per_test = run_count(conf, section)
    sheet = Table(name=SHEET_NAMES[section])

    grid = []
    for config in configs:
        for test in TESTS:
            per_branch = [runs_of(data, section, b, config, test, runs_per_test)
                          for b in branches]
            grid.append((f"{config} / {test}", per_branch))

    title = TableRow()
    title.addElement(cell(text=f"SOLUTION: {SHEET_NAMES[section]}"))
    sheet.addElement(title)

    floor = float(conf["TIMING_RESOLUTION_FLOOR"])
    times_grid = [(label, [t for t, v in pb]) for label, pb in grid]
    paste_blocks(sheet,
                 stat_block_rows("Median time", branches, times_grid,
                                 statistics.median),
                 stat_block_rows("Avg time", branches, times_grid, statistics.mean))
    paste_blocks(sheet,
                 stat_block_rows("Min time", branches, times_grid, min),
                 stat_block_rows("Max time", branches, times_grid, max))
    paste_blocks(sheet,
                 stat_block_rows("STD deviation", branches, times_grid,
                                 lambda t: statistics.pstdev(t) if len(t) > 1 else 0.0),
                 flag_block_rows("Resolution", branches, times_grid,
                                 lambda t: resolution_text(t, floor)))

    correctness = [[cell(text="Correctness")] + [cell(text=b) for b in branches]]
    for label, per_branch in grid:
        cells = [cell(text=label)]
        for times, verdicts in per_branch:
            cells.append(cell(text=correctness_text(times, verdicts, runs_per_test)))
        correctness.append(cells)

    combined = [[cell(text="Runs combined")] + [cell(text=b) for b in branches]]
    for label, per_branch in grid:
        combined.append([cell(text=label)] + [cell() for _ in branches])
        for run in range(runs_per_test):
            cells = [cell(text=f"Run {run + 1}")]
            for times, verdicts in per_branch:
                cells.append(cell(value=times[run]) if run < len(times) else cell())
            combined.append(cells)

    paste_blocks(sheet, correctness, combined)
    doc.spreadsheet.addElement(sheet)


def main():
    """
    TR-6, TM-SHEETS
    Entry point: loads the configuration and the raw measurements, builds
    one sheet per section plus the sequential sheet and stores the document
    in the results directory as required.
    """
    suite_dir, results_dir = suite_paths()
    conf = load_conf(suite_dir)
    data = load_measurements(results_dir)

    known = {(s, b, c, t, r) for s in SHEET_NAMES
             for b in branch_order(conf, s)
             for c in config_order(conf, s)
             for t in TESTS
             for r in range(1, run_count(conf, s) + 1)}
    for key in sorted(set(data) - known):
        print(f"warning: measurement outside the campaign grid: {'|'.join(map(str, key))}")

    doc = OpenDocumentSpreadsheet()
    for section in SHEET_NAMES:
        build_sheet(doc, conf, data, section)

    target = results_dir / "test-results.ods"
    doc.save(str(target))
    print(f"wrote {target} with {len(data)} measurements")


if __name__ == "__main__":
    main()
