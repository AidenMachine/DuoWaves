"""Regression checks for measurement policies that change report conclusions."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import data


def row(**changes):
    sample = data.RawRow("CUDA", "main", "cuda_bs256", "A_1M", 1,
                         .1, "PASS", "100", "node111", "", "", "")
    return replace(sample, **changes)


class MeasurementPolicyTests(unittest.TestCase):
    def test_noisy_cell_keeps_median(self):
        cell = data.aggregate([1, 10, 11, 12, 100])
        self.assertEqual(cell.value, 11)
        self.assertFalse(cell.used_minimum)

    def test_floor_is_applied_to_median_not_fastest_run(self):
        self.assertFalse(data.aggregate([.001, .03, .04]).below_floor)
        self.assertTrue(data.aggregate([.01, .015, .1]).below_floor)

    def test_accidental_pass_with_incomplete_relaxation_is_excluded(self):
        self.assertEqual(data.build_cuda_table([row(config="cuda_bs32")]), {})

    def test_flaky_configuration_is_not_ranked_from_its_passing_subset(self):
        self.assertEqual(data.build_cuda_table([row(), row(run=2, correctness="FAIL")]), {})

    def test_separate_cuda_allocations_are_not_silently_pooled(self):
        with self.assertRaises(ValueError):
            data.build_cuda_table([row(), row(run=2, job_id="101")])

    def test_baseline_selection_uses_sample_count_not_fastest_allocation(self):
        rows = [row(section="SEQ", job_id="10", time=1),
                row(section="SEQ", job_id="20", time=10),
                row(section="SEQ", job_id="20", run=2, time=12),
                row(section="SEQ", job_id="local", time=.001),
                row(section="SEQ", job_id="30", correctness="FAIL", time=.001)]
        baseline = data.build_seq_baselines(rows)["A_1M"]
        self.assertEqual(baseline.value, 11)
        self.assertEqual({r.job_id for r in baseline.source_rows}, {"20"})

    def test_corrected_rerun_requires_fixed_commit_and_matching_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "sbatch_jobs" / "123"
            log_dir.mkdir(parents=True)
            (log_dir / "123.out").write_text(
                "Branch: cuda-1D (7d86f1e)\nBinary md5: abc123\n")
            with patch.object(data, "RAW_DIR", root / "results" / "raw"):
                data.corrected_cuda_build.cache_clear()
                self.assertTrue(data.corrected_cuda_build("cuda-1D", "123", "abc123"))
                self.assertFalse(data.corrected_cuda_build("cuda-1D", "123", "wrong"))
                self.assertFalse(data.corrected_cuda_build("main", "123", "abc123"))
            data.corrected_cuda_build.cache_clear()

    def test_missing_weak_baseline_is_excluded(self):
        missing = data.WeakRow(128, 8000000, 4, 8, 1, 3.7,
                              "MISSING_BASELINE", "", "node1")
        self.assertEqual(data.build_weak_table([missing]), {})


if __name__ == "__main__":
    unittest.main()
