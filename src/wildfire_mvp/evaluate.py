from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .metrics import (
    confusion_counts,
    expected_calibration_error,
    f1_score,
    precision_score,
    pr_auc,
    recall_score,
    reliability_bins,
    specificity_score,
    spearman_rank_correlation,
)
from .settings import load_config
from .reproducibility import (
    collect_environment_info,
    command_log,
    detect_git_commit,
    load_config_snapshot,
    save_manifest,
    select_device,
    utc_timestamp_slug,
)


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("torch is required for evaluation") from exc
    return torch, DataLoader


def _save_reliability_plot(report: dict, output_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save reliability diagrams") from exc

    bins = report["reliability_bins"]
    x = [0.5 * (item["lower"] + item["upper"]) for item in bins]
    y = [item["accuracy"] for item in bins]

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.plot(x, y, marker="o", linewidth=1.5)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def _predict_split(model, loader, model_name: str, mc_passes: int, device: str):
    torch, _ = _require_torch()
    from .models import predict_with_mc_dropout

    y_true = []
    y_prob = []
    y_var = []

    for features, labels in loader:
        features = features.to(device)
        if model_name == "temporal":
            mean, variance = predict_with_mc_dropout(model, features, mc_passes)
            probs = mean.detach().cpu().numpy()
            variances = variance.detach().cpu().numpy()
        else:
            model.eval()
            with torch.no_grad():
                logits = model(features).squeeze(-1)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
            variances = np.zeros_like(probs)
        y_prob.extend(probs.tolist())
        y_var.extend(variances.tolist())
        y_true.extend(labels.numpy().tolist())

    return (
        np.asarray(y_true, dtype=int),
        np.asarray(y_prob, dtype=float),
        np.asarray(y_var, dtype=float),
    )


def _best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold_min: float, threshold_max: float) -> float:
    if len(y_true) == 0:
        return float(np.clip(0.5, threshold_min, threshold_max))

    lo = float(max(0.0, min(1.0, threshold_min)))
    hi = float(max(0.0, min(1.0, threshold_max)))
    if lo > hi:
        lo, hi = hi, lo

    unique = np.unique(np.clip(y_prob, 0.0, 1.0))
    defaults = np.array([0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)
    candidates = np.unique(np.concatenate([defaults, unique]))
    candidates = candidates[(candidates >= lo) & (candidates <= hi)]
    if len(candidates) == 0:
        return float(np.clip(0.5, lo, hi))

    best_t = float(np.clip(0.5, lo, hi))
    best_f1 = -1.0
    for threshold in candidates:
        score = f1_score(y_true, y_prob, float(threshold))
        if score > best_f1:
            best_f1 = score
            best_t = float(threshold)
    return float(np.clip(best_t, lo, hi))


def _best_threshold_by_recall_floor_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    target_recall: float,
) -> float:
    if len(y_true) == 0:
        return float(np.clip(0.5, threshold_min, threshold_max))

    lo = float(max(0.0, min(1.0, threshold_min)))
    hi = float(max(0.0, min(1.0, threshold_max)))
    if lo > hi:
        lo, hi = hi, lo

    target = float(max(0.0, min(1.0, target_recall)))
    unique = np.unique(np.clip(y_prob, 0.0, 1.0))
    defaults = np.array([0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)
    candidates = np.unique(np.concatenate([defaults, unique]))
    candidates = candidates[(candidates >= lo) & (candidates <= hi)]
    if len(candidates) == 0:
        return float(np.clip(0.5, lo, hi))

    feasible = []
    for threshold in candidates:
        threshold = float(threshold)
        rec = recall_score(y_true, y_prob, threshold)
        spec = specificity_score(y_true, y_prob, threshold)
        prec = precision_score(y_true, y_prob, threshold)
        f1 = f1_score(y_true, y_prob, threshold)
        if rec >= target:
            feasible.append((spec, prec, f1, threshold))

    if feasible:
        feasible.sort(reverse=True)
        return feasible[0][3]

    # Fallback when recall floor cannot be satisfied in bounds: keep highest recall,
    # then prefer better specificity/precision/F1.
    ranked = []
    for threshold in candidates:
        threshold = float(threshold)
        rec = recall_score(y_true, y_prob, threshold)
        spec = specificity_score(y_true, y_prob, threshold)
        prec = precision_score(y_true, y_prob, threshold)
        f1 = f1_score(y_true, y_prob, threshold)
        ranked.append((rec, spec, prec, f1, threshold))
    ranked.sort(reverse=True)
    return ranked[0][4]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate wildfire detection MVP")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["static", "temporal"], required=True)
    parser.add_argument("--split", default="ood_test")
    parser.add_argument("--output", default="", help="Optional JSON output path for evaluation report")
    args = parser.parse_args()
    run_started_at = datetime.now(timezone.utc).isoformat()

    config = load_config(args.config)
    torch, DataLoader = _require_torch()
    from .datasets import SequenceTensorDataset
    from .models import StaticResNet18, TemporalAttentionClassifier, predict_with_mc_dropout

    dataset = SequenceTensorDataset(
        metadata_csv=config.paths.sequences_csv,
        tensor_dir=config.paths.tensor_dir,
        split=args.split,
        static_mode=args.model == "static",
    )
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)
    device = select_device(torch)
    in_channels = len(config.data.channels)
    if args.model == "static":
        model = StaticResNet18(in_channels=in_channels, dropout=config.training.dropout)
    else:
        model = TemporalAttentionClassifier(in_channels=in_channels, dropout=config.training.dropout)

    checkpoint = config.paths.artifacts_dir / "models" / f"{args.model}_best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    y_true_arr, y_prob_arr, y_var_arr = _predict_split(
        model=model,
        loader=loader,
        model_name=args.model,
        mc_passes=config.evaluation.mc_passes,
        device=device,
    )

    threshold = 0.5
    threshold_min = config.evaluation.threshold_min
    threshold_max = config.evaluation.threshold_max
    if threshold_min > threshold_max:
        threshold_min, threshold_max = threshold_max, threshold_min
    threshold_selection = config.evaluation.threshold_selection.strip().lower()
    target_recall = config.evaluation.target_recall
    if args.split != "val":
        val_dataset = SequenceTensorDataset(
            metadata_csv=config.paths.sequences_csv,
            tensor_dir=config.paths.tensor_dir,
            split="val",
            static_mode=args.model == "static",
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
            val_true, val_prob, _ = _predict_split(
                model=model,
                loader=val_loader,
                model_name=args.model,
                mc_passes=config.evaluation.mc_passes,
                device=device,
            )
            if len(np.unique(val_true)) > 1:
                if threshold_selection == "recall_floor_specificity":
                    threshold = _best_threshold_by_recall_floor_specificity(
                        val_true,
                        val_prob,
                        threshold_min=threshold_min,
                        threshold_max=threshold_max,
                        target_recall=target_recall,
                    )
                else:
                    threshold = _best_threshold_by_f1(
                        val_true,
                        val_prob,
                        threshold_min=threshold_min,
                        threshold_max=threshold_max,
                    )

    threshold = float(np.clip(threshold, threshold_min, threshold_max))

    errors = np.abs((y_prob_arr >= threshold).astype(int) - y_true_arr)

    report = {
        "pr_auc": pr_auc(y_true_arr, y_prob_arr),
        "f1": f1_score(y_true_arr, y_prob_arr, threshold),
        "precision": precision_score(y_true_arr, y_prob_arr, threshold),
        "recall": recall_score(y_true_arr, y_prob_arr, threshold),
        "specificity": specificity_score(y_true_arr, y_prob_arr, threshold),
        "threshold": threshold,
        "threshold_selection": threshold_selection,
        "target_recall": target_recall,
        "confusion": confusion_counts(y_true_arr, y_prob_arr, threshold),
        "ece": expected_calibration_error(y_true_arr, y_prob_arr, config.evaluation.ece_bins),
        "variance_error_spearman": spearman_rank_correlation(y_var_arr, errors),
        "reliability_bins": [
            {
                "lower": item.lower,
                "upper": item.upper,
                "count": item.count,
                "confidence": item.confidence,
                "accuracy": item.accuracy,
            }
            for item in reliability_bins(y_true_arr, y_prob_arr, config.evaluation.ece_bins)
        ],
    }

    output_json = Path(args.output) if args.output else config.paths.artifacts_dir / "eval" / f"{args.model}_{args.split}.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    reliability_png = output_json.with_suffix(".reliability.png")
    _save_reliability_plot(report, reliability_png)

    run_ended_at = datetime.now(timezone.utc).isoformat()
    run_id = f"eval_{args.model}_{args.split}_{utc_timestamp_slug()}"
    manifest_path = config.paths.artifacts_dir / "runs" / f"{run_id}.json"
    manifest = {
        "run_id": run_id,
        "run_type": "evaluate",
        "model": args.model,
        "split": args.split,
        "started_at": run_started_at,
        "ended_at": run_ended_at,
        "git_commit": detect_git_commit(cwd=config.paths.artifacts_dir.parent),
        "command": command_log(),
        "config": {
            "path": str(Path(args.config).resolve()),
            "snapshot_toml": load_config_snapshot(args.config),
        },
        "environment": collect_environment_info(),
        "device": device,
        "outputs": {
            "report_json": str(output_json),
            "reliability_png": str(reliability_png),
            "metrics": report,
        },
    }
    save_manifest(manifest_path, manifest)

    print(report)
    print({"manifest": str(manifest_path)})


if __name__ == "__main__":
    main()
