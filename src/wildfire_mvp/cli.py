from __future__ import annotations

import argparse
from pathlib import Path
from datetime import date

from .data_pipeline import (
    assign_event_splits,
    build_split_label_summary,
    build_sequence_records,
    cluster_firms_events,
    ensure_layout,
    generate_hard_negative_events,
    read_csv,
    save_summary_json,
    save_events,
    save_sequences,
)
from .settings import load_config
from .schemas import EventRecord
from .download import download_firms_csv, extract_sentinel_patches


def cmd_init_layout(args) -> None:
    config = load_config(args.config)
    ensure_layout(
        [
            config.paths.raw_dir,
            config.paths.processed_dir,
            config.paths.artifacts_dir,
            config.paths.tensor_dir,
            config.paths.raw_dir / "firms",
            config.paths.raw_dir / "sentinel",
        ]
    )


def cmd_cluster_events(args) -> None:
    config = load_config(args.config)
    rows = read_csv(Path(args.input))
    events = cluster_firms_events(
        rows,
        radius_km=config.data.cluster_radius_km,
        window_days=config.data.cluster_days,
        region=args.region or config.firms["default_region"],
        allowed_confidence=set(config.firms["min_confidence"]),
    )
    save_events(Path(args.output), events)


def cmd_build_splits(args) -> None:
    config = load_config(args.config)
    rows = read_csv(Path(args.events))
    events = [EventRecord(**_convert_event_row(row)) for row in rows]
    assigned = assign_event_splits(
        events,
        train_fraction=config.data.train_fraction,
        val_fraction=config.data.val_fraction,
        seed=config.seed,
        ood_regions=set(config.data.ood_regions),
    )
    negatives = generate_hard_negative_events(
        assigned,
        negatives_per_positive=int(config.data.hard_negatives_per_positive),
        min_distance_km=float(config.data.hard_negative_min_distance_km),
        max_distance_km=float(config.data.hard_negative_max_distance_km),
        day_jitter=int(config.data.hard_negative_day_jitter),
        seed=config.seed,
    )
    all_events = assigned + negatives
    sequences = build_sequence_records(all_events, config.data.frame_offsets_days)
    output_dir = Path(args.output_dir)
    save_events(output_dir / "events.csv", all_events)
    save_sequences(output_dir / "sequences.csv", sequences)
    summary = build_split_label_summary(all_events, sequences)
    save_summary_json(output_dir / "split_summary.json", summary)
    print(summary)


def cmd_download_firms(args) -> None:
    config = load_config(args.config)
    output_csv = Path(args.output or (config.paths.raw_dir / "firms" / "firms_events.csv"))
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    download_firms_csv(
        output_csv=output_csv,
        region=args.region or config.firms["default_region"],
        start_date=start_date,
        end_date=end_date,
        api_key_env=config.firms.get("api_key_env", "FIRMS_API_KEY"),
    )


def cmd_extract_sentinel(args) -> None:
    config = load_config(args.config)
    events_csv = Path(args.events) if args.events else config.paths.events_csv
    extract_sentinel_patches(
        events_csv=events_csv,
        output_dir=Path(args.output_dir or config.paths.tensor_dir),
        provider=config.sentinel["provider"],
        collection=config.sentinel["collection"],
        service_account_json=config.sentinel.get("service_account_json", ""),
        project=config.sentinel.get("project", ""),
        bands=config.data.channels,
        frame_offsets_days=config.data.frame_offsets_days,
        image_size=config.data.image_size,
        max_cloud_cover=config.data.max_cloud_cover,
        scale_m=int(config.sentinel.get("scale_m", 20)),
        search_window_days=int(config.sentinel.get("search_window_days", 3)),
        service_account_email_env=config.sentinel.get("service_account_email_env", "GOOGLE_SERVICE_ACCOUNT_EMAIL"),
        min_clear_fraction=float(config.sentinel.get("min_clear_fraction", 0.7)),
        max_candidates=int(config.sentinel.get("max_candidates", 5)),
    )


def _convert_event_row(row: dict) -> dict:
    return {
        "event_id": row["event_id"],
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "date": row["date"],
        "region": row["region"],
        "label": int(row["label"]),
        "source": row["source"],
        "confidence": row["confidence"],
        "split": row["split"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wildfire detection MVP CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_layout = subparsers.add_parser("init-layout", help="Create data and artifact directories")
    init_layout.add_argument("--config", required=True)
    init_layout.set_defaults(func=cmd_init_layout)

    cluster = subparsers.add_parser("cluster-events", help="Cluster FIRMS detections into wildfire events")
    cluster.add_argument("--config", required=True)
    cluster.add_argument("--input", required=True)
    cluster.add_argument("--output", required=True)
    cluster.add_argument("--region", default="")
    cluster.set_defaults(func=cmd_cluster_events)

    build_splits = subparsers.add_parser("build-splits", help="Assign event-level splits and build sequences")
    build_splits.add_argument("--config", required=True)
    build_splits.add_argument("--events", required=True)
    build_splits.add_argument("--output-dir", required=True)
    build_splits.set_defaults(func=cmd_build_splits)

    download_firms = subparsers.add_parser("download-firms", help="Download FIRMS detections CSV")
    download_firms.add_argument("--config", required=True)
    download_firms.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    download_firms.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    download_firms.add_argument("--region", default="")
    download_firms.add_argument("--output", default="")
    download_firms.set_defaults(func=cmd_download_firms)

    extract_sentinel = subparsers.add_parser(
        "extract-sentinel", help="Extract Sentinel-2 sequence tensors from event metadata"
    )
    extract_sentinel.add_argument("--config", required=True)
    extract_sentinel.add_argument("--events", default="", help="Path to events.csv")
    extract_sentinel.add_argument("--output-dir", default="", help="Directory to write sequence .npy tensors")
    extract_sentinel.set_defaults(func=cmd_extract_sentinel)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
