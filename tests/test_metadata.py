import tempfile
import unittest
from pathlib import Path

from wildfire_mvp.data_pipeline import (
    assign_event_splits,
    build_split_label_summary,
    build_sequence_records,
    generate_hard_negative_events,
    save_summary_json,
    save_events,
    save_sequences,
)
from wildfire_mvp.schemas import EventRecord


class MetadataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            EventRecord("a", 34.0, -118.0, "2025-08-01", "western_us", 1, "firms", "high", "train"),
            EventRecord("b", 35.0, -119.0, "2025-08-02", "western_us", 1, "firms", "high", "train"),
            EventRecord("c", 36.0, -120.0, "2025-08-03", "canada", 1, "firms", "high", "train"),
        ]

    def test_ood_region_goes_to_ood_split(self):
        assigned = assign_event_splits(self.events, 0.7, 0.15, seed=42, ood_regions={"canada"})
        canada = [item for item in assigned if item.region == "canada"]
        self.assertEqual(len(canada), 1)
        self.assertEqual(canada[0].split, "ood_test")

    def test_sequence_builder_creates_three_dates(self):
        assigned = assign_event_splits(self.events, 0.7, 0.15, seed=42, ood_regions={"canada"})
        sequences = build_sequence_records(assigned, [-10, -5, 0])
        self.assertEqual(len(sequences), len(assigned))
        for sequence in sequences:
            sequence.validate()

    def test_csv_writers_create_files(self):
        assigned = assign_event_splits(self.events, 0.7, 0.15, seed=42, ood_regions={"canada"})
        sequences = build_sequence_records(assigned, [-10, -5, 0])
        with tempfile.TemporaryDirectory() as tmpdir:
            events_path = Path(tmpdir) / "events.csv"
            sequences_path = Path(tmpdir) / "sequences.csv"
            save_events(events_path, assigned)
            save_sequences(sequences_path, sequences)
            self.assertTrue(events_path.exists())
            self.assertTrue(sequences_path.exists())

    def test_hard_negatives_are_label_zero_and_split_safe(self):
        assigned = assign_event_splits(self.events, 0.7, 0.15, seed=42, ood_regions={"canada"})
        negatives = generate_hard_negative_events(
            assigned,
            negatives_per_positive=1,
            min_distance_km=7.5,
            max_distance_km=15.0,
            day_jitter=1,
            seed=42,
        )
        self.assertEqual(len(negatives), len(assigned))
        for neg in negatives:
            self.assertEqual(neg.label, 0)
            self.assertEqual(neg.source, "hard_negative")
            self.assertIn(neg.split, {"train", "val", "test", "ood_test"})
            neg.validate()

    def test_hard_negative_generation_can_be_disabled(self):
        assigned = assign_event_splits(self.events, 0.7, 0.15, seed=42, ood_regions={"canada"})
        negatives = generate_hard_negative_events(
            assigned,
            negatives_per_positive=0,
            min_distance_km=7.5,
            max_distance_km=15.0,
            day_jitter=1,
            seed=42,
        )
        self.assertEqual(negatives, [])

    def test_split_summary_counts_are_consistent(self):
        assigned = assign_event_splits(self.events, 0.7, 0.15, seed=42, ood_regions={"canada"})
        negatives = generate_hard_negative_events(
            assigned,
            negatives_per_positive=1,
            min_distance_km=7.5,
            max_distance_km=15.0,
            day_jitter=1,
            seed=42,
        )
        all_events = assigned + negatives
        sequences = build_sequence_records(all_events, [-10, -5, 0])
        summary = build_split_label_summary(all_events, sequences)
        self.assertEqual(summary["totals"]["events"], len(all_events))
        self.assertEqual(summary["totals"]["sequences"], len(sequences))
        event_count_sum = sum(item["label_0"] + item["label_1"] for item in summary["events"].values())
        sequence_count_sum = sum(item["label_0"] + item["label_1"] for item in summary["sequences"].values())
        self.assertEqual(event_count_sum, len(all_events))
        self.assertEqual(sequence_count_sum, len(sequences))

    def test_save_summary_json_writes_file(self):
        summary = {
            "events": {"train": {"label_0": 1, "label_1": 2}},
            "sequences": {"train": {"label_0": 1, "label_1": 2}},
            "totals": {"events": 3, "sequences": 3},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "split_summary.json"
            save_summary_json(summary_path, summary)
            self.assertTrue(summary_path.exists())


if __name__ == "__main__":
    unittest.main()
