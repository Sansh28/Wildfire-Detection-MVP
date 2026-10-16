from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .settings import load_config
from .metrics import f1_score, pr_auc, recall_score
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
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("torch is required for training") from exc
    return torch, nn, DataLoader


def train_one_epoch(model, loader, optimizer, criterion, device):
    torch, _, _ = _require_torch()
    model.train()
    losses = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device).view(-1)
        optimizer.zero_grad()
        logits = model(features).view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate_epoch(model, loader, device):
    torch, _, _ = _require_torch()
    model.eval()
    y_true = []
    y_prob = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob.extend(np.asarray(probs).reshape(-1).tolist())
            y_true.extend(np.asarray(labels.numpy()).reshape(-1).tolist())
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    return {
        "pr_auc": pr_auc(y_true_arr, y_prob_arr),
        "f1": f1_score(y_true_arr, y_prob_arr),
        "recall": recall_score(y_true_arr, y_prob_arr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train wildfire detection MVP")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["static", "temporal"], required=True)
    args = parser.parse_args()
    run_started_at = datetime.now(timezone.utc).isoformat()

    config = load_config(args.config)
    torch, nn, DataLoader = _require_torch()

    from .datasets import SequenceTensorDataset
    from .models import StaticResNet18, TemporalAttentionClassifier

    train_dataset = SequenceTensorDataset(
        metadata_csv=config.paths.sequences_csv,
        tensor_dir=config.paths.tensor_dir,
        split="train",
        static_mode=args.model == "static",
    )
    val_dataset = SequenceTensorDataset(
        metadata_csv=config.paths.sequences_csv,
        tensor_dir=config.paths.tensor_dir,
        split="val",
        static_mode=args.model == "static",
    )

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    device = select_device(torch)
    in_channels = len(config.data.channels)
    if args.model == "static":
        model = StaticResNet18(in_channels=in_channels, dropout=config.training.dropout)
    else:
        model = TemporalAttentionClassifier(in_channels=in_channels, dropout=config.training.dropout)
    model.to(device)

    pos_weight = torch.tensor([config.training.positive_class_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    best_pr_auc = -1.0
    patience = 0
    history = []
    best_checkpoint: Path | None = None
    for epoch in range(config.training.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate_epoch(model, val_loader, device)
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        print(metrics)
        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            patience = 0
            output_dir = config.paths.artifacts_dir / "models"
            output_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint = output_dir / f"{args.model}_best.pt"
            torch.save(model.state_dict(), best_checkpoint)
        else:
            patience += 1
            if patience >= config.training.early_stopping_patience:
                break

    run_ended_at = datetime.now(timezone.utc).isoformat()
    run_id = f"train_{args.model}_{utc_timestamp_slug()}"
    manifest_path = config.paths.artifacts_dir / "runs" / f"{run_id}.json"
    manifest = {
        "run_id": run_id,
        "run_type": "train",
        "model": args.model,
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
            "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
            "history": history,
        },
    }
    save_manifest(manifest_path, manifest)
    print({"manifest": str(manifest_path), "best_checkpoint": str(best_checkpoint) if best_checkpoint else ""})


if __name__ == "__main__":
    main()
