from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import re
from pathlib import Path
import tomllib


@dataclass(slots=True)
class EvalRow:
    run_id: str
    started_at: str
    model: str
    split: str
    band_profile: str
    channel_count: int
    pr_auc: float
    f1: float
    recall: float
    ece: float
    variance_error_spearman: float
    git_commit: str
    report_json: str
    manifest_path: str


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify_band_profile(channels: list[str]) -> str:
    present = set(channels)
    rgb = {"B04", "B03", "B02"}
    if rgb.issubset(present):
        if {"B11", "B12"}.issubset(present):
            return "RGB+NIR+SWIR"
        if "B08" in present:
            return "RGB+NIR"
        return "RGB"
    return "custom"


def parse_channels_from_snapshot(snapshot_toml: str) -> list[str]:
    if not snapshot_toml.strip():
        return []
    try:
        parsed = tomllib.loads(snapshot_toml)
    except tomllib.TOMLDecodeError:
        return []
    data = parsed.get("data", {})
    channels = data.get("channels", [])
    if isinstance(channels, list):
        return [str(item) for item in channels]
    return []


def load_eval_rows(runs_dir: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    if not runs_dir.exists():
        return rows

    for path in sorted(runs_dir.glob("eval_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        outputs = payload.get("outputs", {})
        metrics = outputs.get("metrics", {})
        config = payload.get("config", {})
        channels = parse_channels_from_snapshot(str(config.get("snapshot_toml", "")))
        rows.append(
            EvalRow(
                run_id=str(payload.get("run_id", path.stem)),
                started_at=str(payload.get("started_at", "")),
                model=str(payload.get("model", "unknown")),
                split=str(payload.get("split", "unknown")),
                band_profile=classify_band_profile(channels),
                channel_count=len(channels),
                pr_auc=_safe_float(metrics.get("pr_auc")),
                f1=_safe_float(metrics.get("f1")),
                recall=_safe_float(metrics.get("recall")),
                ece=_safe_float(metrics.get("ece")),
                variance_error_spearman=_safe_float(metrics.get("variance_error_spearman")),
                git_commit=str(payload.get("git_commit", "unknown")),
                report_json=str(outputs.get("report_json", "")),
                manifest_path=str(path),
            )
        )
    return rows


def _timestamp_key(value: str) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def latest_by_key(rows: list[EvalRow], key_fields: tuple[str, ...]) -> list[EvalRow]:
    latest: dict[tuple, EvalRow] = {}
    for row in rows:
        key = tuple(getattr(row, field) for field in key_fields)
        existing = latest.get(key)
        if existing is None or _timestamp_key(row.started_at) >= _timestamp_key(existing.started_at):
            latest[key] = row
    return sorted(latest.values(), key=lambda item: tuple(getattr(item, field) for field in key_fields))


def write_all_runs_csv(rows: list[EvalRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "started_at",
                "model",
                "split",
                "band_profile",
                "channel_count",
                "pr_auc",
                "f1",
                "recall",
                "ece",
                "variance_error_spearman",
                "git_commit",
                "report_json",
                "manifest_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_id": row.run_id,
                    "started_at": row.started_at,
                    "model": row.model,
                    "split": row.split,
                    "band_profile": row.band_profile,
                    "channel_count": row.channel_count,
                    "pr_auc": row.pr_auc,
                    "f1": row.f1,
                    "recall": row.recall,
                    "ece": row.ece,
                    "variance_error_spearman": row.variance_error_spearman,
                    "git_commit": row.git_commit,
                    "report_json": row.report_json,
                    "manifest_path": row.manifest_path,
                }
            )


def _md_table(rows: list[EvalRow]) -> str:
    header = (
        "| model | split | band_profile | channels | PR-AUC | F1 | Recall | ECE | Var-Err Spearman |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|"
    )
    body = [
        "| "
        + " | ".join(
            [
                row.model,
                row.split,
                row.band_profile,
                str(row.channel_count),
                f"{row.pr_auc:.4f}",
                f"{row.f1:.4f}",
                f"{row.recall:.4f}",
                f"{row.ece:.4f}",
                f"{row.variance_error_spearman:.4f}",
            ]
        )
        + " |"
        for row in rows
    ]
    return "\n".join([header, *body])


def _increment_key(row: EvalRow) -> int:
    match = re.search(r"inc(\d+)", row.run_id)
    if match:
        return int(match.group(1))
    match = re.search(r"inc(\d+)", row.report_json)
    if match:
        return int(match.group(1))
    return -1


def _format_delta(current: float, previous: float | None) -> str:
    if previous is None:
        return "n/a"
    return f"{current - previous:+.4f}"


def write_incremental_comparison(rows: list[EvalRow], output_dir: Path) -> tuple[Path, Path]:
    inc_rows = [row for row in rows if _increment_key(row) >= 0]
    inc_rows = sorted(inc_rows, key=lambda item: (_increment_key(item), item.model))

    inc_csv = output_dir / "incremental_comparison.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with inc_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "increment",
                "model",
                "split",
                "band_profile",
                "pr_auc",
                "f1",
                "recall",
                "ece",
                "variance_error_spearman",
                "delta_pr_auc",
                "delta_f1",
                "delta_recall",
                "delta_ece",
                "delta_variance_error_spearman",
                "run_id",
                "manifest_path",
            ],
        )
        writer.writeheader()

        previous_by_model: dict[str, EvalRow] = {}
        for row in inc_rows:
            previous = previous_by_model.get(row.model)
            increment = _increment_key(row)
            writer.writerow(
                {
                    "increment": increment,
                    "model": row.model,
                    "split": row.split,
                    "band_profile": row.band_profile,
                    "pr_auc": row.pr_auc,
                    "f1": row.f1,
                    "recall": row.recall,
                    "ece": row.ece,
                    "variance_error_spearman": row.variance_error_spearman,
                    "delta_pr_auc": _format_delta(row.pr_auc, None if previous is None else previous.pr_auc),
                    "delta_f1": _format_delta(row.f1, None if previous is None else previous.f1),
                    "delta_recall": _format_delta(row.recall, None if previous is None else previous.recall),
                    "delta_ece": _format_delta(row.ece, None if previous is None else previous.ece),
                    "delta_variance_error_spearman": _format_delta(
                        row.variance_error_spearman,
                        None if previous is None else previous.variance_error_spearman,
                    ),
                    "run_id": row.run_id,
                    "manifest_path": row.manifest_path,
                }
            )
            previous_by_model[row.model] = row

    inc_md = output_dir / "incremental_comparison.md"
    md_lines = [
        "# Incremental Comparison Summary",
        "",
        "Inc-only comparison across the fast subset expansion runs.",
        "",
        "| increment | model | PR-AUC | F1 | Recall | ECE | Var-Err Spearman | delta PR-AUC | delta F1 | delta Recall | delta ECE | delta Var-Err Spearman |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    previous_by_model.clear()
    for row in inc_rows:
        previous = previous_by_model.get(row.model)
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(_increment_key(row)),
                    row.model,
                    f"{row.pr_auc:.4f}",
                    f"{row.f1:.4f}",
                    f"{row.recall:.4f}",
                    f"{row.ece:.4f}",
                    f"{row.variance_error_spearman:.4f}",
                    _format_delta(row.pr_auc, None if previous is None else previous.pr_auc),
                    _format_delta(row.f1, None if previous is None else previous.f1),
                    _format_delta(row.recall, None if previous is None else previous.recall),
                    _format_delta(row.ece, None if previous is None else previous.ece),
                    _format_delta(
                        row.variance_error_spearman,
                        None if previous is None else previous.variance_error_spearman,
                    ),
                ]
            )
            + " |"
        )
        previous_by_model[row.model] = row
    inc_md.write_text("\n".join(md_lines), encoding="utf-8")

    return inc_md, inc_csv


def write_experiment_markdown(rows: list[EvalRow], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    exp1_rows = latest_by_key(rows, ("model", "split", "band_profile"))
    exp1_md = output_dir / "experiment1_h1_h2_h3.md"
    exp1_content = "\n".join(
        [
            "# Experiment 1 Summary (H1/H2/H3)",
            "",
            "Latest evaluation run for each model/split/band profile.",
            "",
            _md_table(exp1_rows) if exp1_rows else "No evaluation runs found.",
        ]
    )
    exp1_md.write_text(exp1_content, encoding="utf-8")

    temporal_rows = [row for row in rows if row.model == "temporal"]
    exp2_rows = latest_by_key(temporal_rows, ("split", "band_profile"))
    exp2_md = output_dir / "experiment2_band_ablation.md"
    exp2_content = "\n".join(
        [
            "# Experiment 2 Band Ablation Summary",
            "",
            "Latest temporal-model run for each split and band profile.",
            "",
            _md_table(exp2_rows) if exp2_rows else "No temporal evaluation runs found.",
        ]
    )
    exp2_md.write_text(exp2_content, encoding="utf-8")

    return exp1_md, exp2_md


def write_overview_json(rows: list[EvalRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_eval_runs": len(rows),
        "models": sorted({row.model for row in rows}),
        "splits": sorted({row.split for row in rows}),
        "band_profiles": sorted({row.band_profile for row in rows}),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_uncertainty_summary(rows: list[EvalRow], output_dir: Path) -> Path:
    output_path = output_dir / "uncertainty_summary.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_groups: dict[str, list[EvalRow]] = {}
    for row in rows:
        model_groups.setdefault(row.model, []).append(row)

    lines = [
        "# Uncertainty Summary",
        "",
        "Run-wide view of calibration and variance-error behavior across all evaluation artifacts.",
        "",
        "| model | runs | mean ECE | mean Var-Err Spearman | positive Var-Err runs |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in sorted(model_groups):
        model_rows = model_groups[model]
        ece_values = [row.ece for row in model_rows]
        var_values = [row.variance_error_spearman for row in model_rows]
        positive_count = sum(1 for value in var_values if value > 0)
        lines.append(
            f"| {model} | {len(model_rows)} | {sum(ece_values) / len(ece_values):.4f} | {sum(var_values) / len(var_values):.4f} | {positive_count} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_dominance_summary(rows: list[EvalRow], output_dir: Path) -> Path:
    output_path = output_dir / "increment_dominance_summary.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    by_increment: dict[int, dict[str, EvalRow]] = {}
    for row in rows:
        increment = _increment_key(row)
        if increment < 0:
            continue
        by_increment.setdefault(increment, {})[row.model] = row

    lines = [
        "# Increment Dominance Summary",
        "",
        "Per-increment winner for PR-AUC and recall across static vs temporal runs.",
        "",
        "| increment | PR-AUC winner | recall winner |",
        "|---:|---|---|",
    ]
    for increment in sorted(by_increment):
        pair = by_increment[increment]
        if "static" not in pair or "temporal" not in pair:
            continue
        static = pair["static"]
        temporal = pair["temporal"]
        pr_winner = "temporal" if temporal.pr_auc > static.pr_auc else ("static" if static.pr_auc > temporal.pr_auc else "tie")
        recall_winner = "temporal" if temporal.recall > static.recall else ("static" if static.recall > temporal.recall else "tie")
        lines.append(f"| {increment} | {pr_winner} | {recall_winner} |")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate wildfire MVP evaluation artifacts")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory")
    parser.add_argument("--output-dir", default="artifacts/reports", help="Output directory for summary reports")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    rows = load_eval_rows(artifacts_dir / "runs")
    write_all_runs_csv(rows, output_dir / "all_eval_runs.csv")
    exp1_path, exp2_path = write_experiment_markdown(rows, output_dir)
    inc_md_path, inc_csv_path = write_incremental_comparison(rows, output_dir)
    uncertainty_path = write_uncertainty_summary(rows, output_dir)
    dominance_path = write_dominance_summary(rows, output_dir)
    write_overview_json(rows, output_dir / "overview.json")

    print(
        {
            "total_eval_runs": len(rows),
            "all_eval_runs_csv": str(output_dir / "all_eval_runs.csv"),
            "experiment1_markdown": str(exp1_path),
            "experiment2_markdown": str(exp2_path),
            "incremental_markdown": str(inc_md_path),
            "incremental_csv": str(inc_csv_path),
            "uncertainty_markdown": str(uncertainty_path),
            "dominance_markdown": str(dominance_path),
            "overview_json": str(output_dir / "overview.json"),
        }
    )


if __name__ == "__main__":
    main()
