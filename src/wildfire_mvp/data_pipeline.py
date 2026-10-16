from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
import csv
import hashlib
import json
import math
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
import random

from .schemas import EventRecord, SequenceRecord


def ensure_layout(base_paths: list[Path]) -> None:
    for path in base_paths:
        path.mkdir(parents=True, exist_ok=True)


def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius = 6371.0
    dlat = radians(lat_b - lat_a)
    dlon = radians(lon_b - lon_a)
    a = sin(dlat / 2) ** 2 + cos(radians(lat_a)) * cos(radians(lat_b)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return radius * c


def _event_key(lat: float, lon: float, when: str, radius_km: float, window_days: int) -> str:
    day_bucket = datetime.fromisoformat(when).date().toordinal() // max(window_days, 1)
    lat_bucket = round(lat / max(radius_km / 111.0, 1e-6))
    lon_bucket = round(lon / max(radius_km / 111.0, 1e-6))
    raw = f"{lat_bucket}:{lon_bucket}:{day_bucket}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def cluster_firms_events(
    rows: list[dict],
    radius_km: float,
    window_days: int,
    region: str,
    allowed_confidence: set[str],
) -> list[EventRecord]:
    clusters: dict[str, EventRecord] = {}
    confidence_aliases = {
        "n": "nominal",
        "h": "high",
        "l": "low",
    }
    for row in rows:
        confidence = str(row.get("confidence", "")).strip().lower()
        confidence = confidence_aliases.get(confidence, confidence)
        if confidence not in allowed_confidence:
            continue

        lat = float(row["latitude"])
        lon = float(row["longitude"])
        when = str(row["acq_date"])
        event_id = _event_key(lat, lon, when, radius_km, window_days)

        if event_id not in clusters:
            clusters[event_id] = EventRecord(
                event_id=event_id,
                lat=lat,
                lon=lon,
                date=when,
                region=region,
                label=1,
                source="firms_viirs",
                confidence=confidence,
                split="train",
            )
            continue

        existing = clusters[event_id]
        existing_date = datetime.fromisoformat(existing.date).date()
        new_date = datetime.fromisoformat(when).date()
        if new_date < existing_date:
            existing.date = when
            existing.lat = lat
            existing.lon = lon
            existing.confidence = confidence

    return list(clusters.values())


def assign_event_splits(
    events: list[EventRecord],
    train_fraction: float,
    val_fraction: float,
    seed: int,
    ood_regions: set[str],
) -> list[EventRecord]:
    randomizer = random.Random(seed)
    grouped: dict[str, list[EventRecord]] = defaultdict(list)
    for event in events:
        grouped[event.region].append(event)

    assigned: list[EventRecord] = []
    for region, items in grouped.items():
        randomizer.shuffle(items)
        if region in ood_regions:
            for item in items:
                item.split = "ood_test"
                assigned.append(item)
            continue

        total = len(items)
        train_cut = int(total * train_fraction)
        val_cut = train_cut + int(total * val_fraction)
        for idx, item in enumerate(items):
            if idx < train_cut:
                item.split = "train"
            elif idx < val_cut:
                item.split = "val"
            else:
                item.split = "test"
            assigned.append(item)
    return assigned


def build_sequence_records(events: list[EventRecord], frame_offsets_days: list[int]) -> list[SequenceRecord]:
    sequences: list[SequenceRecord] = []
    sorted_offsets = sorted(frame_offsets_days)
    if len(sorted_offsets) != 3:
        raise ValueError("MVP sequence builder expects exactly 3 frame offsets")

    for event in events:
        event_date = datetime.fromisoformat(event.date).date()
        dates = [(event_date + timedelta(days=offset)).isoformat() for offset in sorted_offsets]
        sequence_id = f"seq_{event.event_id}"
        patch_id = f"patch_{event.event_id}"
        sequences.append(
            SequenceRecord(
                sequence_id=sequence_id,
                event_id=event.event_id,
                t0_date=dates[0],
                t1_date=dates[1],
                t2_date=dates[2],
                patch_id=patch_id,
                label=event.label,
                split=event.split,
            )
        )
    return sequences


def _normalize_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def _offset_lat_lon(lat: float, lon: float, distance_km: float, bearing_radians: float) -> tuple[float, float]:
    delta_lat = (distance_km / 111.0) * math.cos(bearing_radians)
    cos_lat = max(math.cos(math.radians(lat)), 1e-6)
    delta_lon = (distance_km / (111.0 * cos_lat)) * math.sin(bearing_radians)
    new_lat = max(min(lat + delta_lat, 89.9), -89.9)
    new_lon = _normalize_lon(lon + delta_lon)
    return new_lat, new_lon


def _negative_event_id(parent_event_id: str, index: int, seed: int) -> str:
    raw = f"neg:{parent_event_id}:{index}:{seed}"
    return f"neg_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def generate_hard_negative_events(
    events: list[EventRecord],
    negatives_per_positive: int,
    min_distance_km: float,
    max_distance_km: float,
    day_jitter: int,
    seed: int,
    max_attempts_per_negative: int = 12,
) -> list[EventRecord]:
    if negatives_per_positive <= 0:
        return []
    if min_distance_km <= 0 or max_distance_km < min_distance_km:
        raise ValueError("Invalid hard-negative distance bounds")
    if day_jitter < 0:
        raise ValueError("day_jitter must be non-negative")

    rng = random.Random(seed)
    positives = [event for event in events if event.label == 1]
    positives_by_region: dict[str, list[EventRecord]] = defaultdict(list)
    for event in positives:
        positives_by_region[event.region].append(event)

    negatives: list[EventRecord] = []
    for event in positives:
        region_positives = positives_by_region[event.region]
        base_date = datetime.fromisoformat(event.date).date()

        for index in range(negatives_per_positive):
            candidate: EventRecord | None = None
            for _ in range(max_attempts_per_negative):
                distance_km = rng.uniform(min_distance_km, max_distance_km)
                bearing = rng.uniform(0.0, 2.0 * math.pi)
                neg_lat, neg_lon = _offset_lat_lon(event.lat, event.lon, distance_km, bearing)

                # Keep negatives spatially separated from all known positives in the same region.
                nearest_km = min(
                    haversine_km(neg_lat, neg_lon, item.lat, item.lon) for item in region_positives
                )
                if nearest_km < min_distance_km:
                    continue

                date_shift = rng.randint(-day_jitter, day_jitter)
                neg_date = (base_date + timedelta(days=date_shift)).isoformat()
                candidate = EventRecord(
                    event_id=_negative_event_id(event.event_id, index, seed),
                    lat=neg_lat,
                    lon=neg_lon,
                    date=neg_date,
                    region=event.region,
                    label=0,
                    source="hard_negative",
                    confidence="synthetic",
                    split=event.split,
                )
                break

            if candidate is not None:
                negatives.append(candidate)
    return negatives


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_events(path: Path, events: list[EventRecord]) -> None:
    rows = [event.to_dict() for event in events]
    write_csv(path, rows, ["event_id", "lat", "lon", "date", "region", "label", "source", "confidence", "split"])


def save_sequences(path: Path, sequences: list[SequenceRecord]) -> None:
    rows = [asdict(item) for item in sequences]
    write_csv(path, rows, ["sequence_id", "event_id", "t0_date", "t1_date", "t2_date", "patch_id", "label", "split"])


def build_split_label_summary(events: list[EventRecord], sequences: list[SequenceRecord]) -> dict:
    def _counter(items: list[tuple[str, int]]) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"label_0": 0, "label_1": 0})
        for split, label in items:
            key = "label_1" if int(label) == 1 else "label_0"
            counts[split][key] += 1
        return {split: dict(values) for split, values in sorted(counts.items())}

    event_items = [(event.split, event.label) for event in events]
    sequence_items = [(seq.split, seq.label) for seq in sequences]
    summary = {
        "events": _counter(event_items),
        "sequences": _counter(sequence_items),
        "totals": {
            "events": len(events),
            "sequences": len(sequences),
        },
    }
    return summary


def save_summary_json(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
