from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MIN_EVAL_NEGATIVES_PER_SPLIT = 15


def _assign_splits(df: pd.DataFrame, seed: int, ensure_eval_negatives: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    assigned = df.sample(frac=1.0, random_state=seed).reset_index(drop=True).copy()
    n = len(assigned)

    if ensure_eval_negatives and n >= 2:
        assigned["split"] = "train"
        assigned.loc[n - 2, "split"] = "val"
        assigned.loc[n - 1, "split"] = "test"
        return assigned

    train_cut = max(1, int(n * 0.7))
    val_cut = max(train_cut + 1, int(n * 0.85))
    assigned.loc[: train_cut - 1, "split"] = "train"
    assigned.loc[train_cut: val_cut - 1, "split"] = "val"
    assigned.loc[val_cut:, "split"] = "test"
    return assigned


def _assign_negative_splits(df: pd.DataFrame, seed: int, min_eval_per_split: int) -> pd.DataFrame:
    if df.empty:
        return df

    assigned = df.sample(frac=1.0, random_state=seed).reset_index(drop=True).copy()
    n = len(assigned)
    required = min_eval_per_split * 2
    if n < required + 1:
        raise RuntimeError(
            "Need at least "
            f"{required + 1} natural negatives to guarantee "
            f"{min_eval_per_split} negatives in each of val/test and keep train negatives. "
            f"Found {n}."
        )

    assigned["split"] = "train"
    assigned.loc[: min_eval_per_split - 1, "split"] = "val"
    assigned.loc[min_eval_per_split: required - 1, "split"] = "test"
    return assigned


def main() -> None:
    base = Path("data/processed_fast")
    tensors_dir = base / "tensors"
    in_csv = base / "sequences_available.csv"
    out_csv = base / "sequences_balanced.csv"

    seq = pd.read_csv(in_csv)
    positives = seq[seq["label"] == 1].copy().reset_index(drop=True)
    natural_negatives = seq[seq["label"] == 0].copy().reset_index(drop=True)
    if len(positives) < 6:
        raise RuntimeError("Need at least 6 positive sequences to create train/val/test subset")

    # Deterministic split for quick fast-run iteration.
    positives = _assign_splits(positives, seed=42)
    natural_negatives = _assign_negative_splits(
        natural_negatives,
        seed=4242,
        min_eval_per_split=MIN_EVAL_NEGATIVES_PER_SPLIT,
    )

    negatives = []
    rng = np.random.default_rng(42)
    train_pos = positives[positives["split"] == "train"].copy().reset_index(drop=True)
    train_nat_neg = natural_negatives[natural_negatives["split"] == "train"].copy().reset_index(drop=True)
    synth_needed = max(0, len(train_pos) - len(train_nat_neg))
    for _, row in train_pos.head(synth_needed).iterrows():
        source_id = row["sequence_id"]
        source_path = tensors_dir / f"{source_id}.npy"
        arr = np.load(source_path).astype(np.float32)

        # Synthetic hard-negative from real patch context: perturb and dim signal.
        neg = np.clip(arr * 0.65 + rng.normal(0, 0.03, size=arr.shape).astype(np.float32), 0.0, 1.0)
        neg_id = f"{source_id}_neg"
        np.save(tensors_dir / f"{neg_id}.npy", neg)

        neg_row = row.copy()
        neg_row["sequence_id"] = neg_id
        neg_row["event_id"] = f"{row['event_id']}_neg"
        neg_row["patch_id"] = f"{row['patch_id']}_neg"
        neg_row["label"] = 0
        neg_row["split"] = "train"
        negatives.append(neg_row)

    all_rows = pd.concat([positives, natural_negatives, pd.DataFrame(negatives)], ignore_index=True)
    all_rows = all_rows.sample(frac=1.0, random_state=42).reset_index(drop=True)
    all_rows.to_csv(out_csv, index=False)

    print(
        {
            "positives": int((all_rows["label"] == 1).sum()),
            "negatives": int((all_rows["label"] == 0).sum()),
            "natural_negatives": int(len(natural_negatives)),
            "synthetic_negatives": int(len(negatives)),
            "split_counts": all_rows["split"].value_counts().to_dict(),
            "out_csv": str(out_csv),
        }
    )


if __name__ == "__main__":
    main()
