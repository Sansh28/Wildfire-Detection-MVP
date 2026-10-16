import unittest

import numpy as np

from wildfire_mvp.metrics import (
    confusion_counts,
    expected_calibration_error,
    f1_score,
    pr_auc,
    reliability_bins,
    spearman_rank_correlation,
)


class MetricsTests(unittest.TestCase):
    def test_confusion_counts(self):
        counts = confusion_counts(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.2, 0.1]))
        self.assertEqual(counts, {"tp": 1, "fp": 1, "tn": 1, "fn": 1})

    def test_f1_is_bounded(self):
        score = f1_score(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_pr_auc_is_bounded(self):
        score = pr_auc(np.array([1, 0, 1, 0]), np.array([0.9, 0.2, 0.8, 0.1]))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_reliability_bins_count_matches(self):
        bins = reliability_bins(np.array([1, 0, 1]), np.array([0.9, 0.2, 0.6]), num_bins=3)
        self.assertEqual(sum(item.count for item in bins), 3)

    def test_ece_is_bounded(self):
        ece = expected_calibration_error(np.array([1, 0, 1]), np.array([0.9, 0.2, 0.6]), num_bins=3)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_spearman_detects_positive_correlation(self):
        corr = spearman_rank_correlation(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.0, 0.1, 0.7, 1.0]))
        self.assertGreater(corr, 0.0)


if __name__ == "__main__":
    unittest.main()
