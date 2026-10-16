import json
import tempfile
import unittest
from pathlib import Path

from wildfire_mvp.aggregate_results import (
    classify_band_profile,
    load_eval_rows,
    parse_channels_from_snapshot,
    write_experiment_markdown,
)


class AggregateResultsTests(unittest.TestCase):
    def test_band_profile_classifier(self):
        self.assertEqual(classify_band_profile(["B04", "B03", "B02"]), "RGB")
        self.assertEqual(classify_band_profile(["B04", "B03", "B02", "B08"]), "RGB+NIR")
        self.assertEqual(
            classify_band_profile(["B04", "B03", "B02", "B08", "B11", "B12"]),
            "RGB+NIR+SWIR",
        )

    def test_parse_channels_from_snapshot(self):
        snapshot = "[data]\nchannels = [\"B04\", \"B03\", \"B02\", \"B08\"]\n"
        channels = parse_channels_from_snapshot(snapshot)
        self.assertEqual(channels, ["B04", "B03", "B02", "B08"])

    def test_load_rows_and_write_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs_dir = root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "run_id": "eval_temporal_test_abc",
                "run_type": "evaluate",
                "model": "temporal",
                "split": "test",
                "started_at": "2026-04-18T10:00:00+00:00",
                "git_commit": "abc123",
                "config": {
                    "path": "/tmp/config.toml",
                    "snapshot_toml": "[data]\nchannels = [\"B04\", \"B03\", \"B02\", \"B08\"]\n",
                },
                "outputs": {
                    "report_json": "/tmp/report.json",
                    "metrics": {
                        "pr_auc": 0.9,
                        "f1": 0.8,
                        "recall": 0.85,
                        "ece": 0.1,
                        "variance_error_spearman": 0.5,
                    },
                },
            }
            (runs_dir / "eval_temporal_test_abc.json").write_text(json.dumps(manifest), encoding="utf-8")

            rows = load_eval_rows(runs_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].band_profile, "RGB+NIR")

            exp1, exp2 = write_experiment_markdown(rows, root / "reports")
            self.assertTrue(exp1.exists())
            self.assertTrue(exp2.exists())


if __name__ == "__main__":
    unittest.main()
