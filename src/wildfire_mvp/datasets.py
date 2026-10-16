from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _require_torch():
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError as exc:
        raise RuntimeError("torch is required for dataset loading and training") from exc
    return torch, Dataset


class SequenceTensorDataset:
    def __init__(self, metadata_csv: str | Path, tensor_dir: str | Path, split: str, static_mode: bool = False):
        torch, dataset_base = _require_torch()
        self._torch = torch
        self._dataset_base = dataset_base
        self.metadata = pd.read_csv(metadata_csv)
        self.metadata = self.metadata[self.metadata["split"] == split].reset_index(drop=True)
        self.tensor_dir = Path(tensor_dir)
        self.static_mode = static_mode

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        tensor_path = self.tensor_dir / f"{row['sequence_id']}.npy"
        label = int(row.get("label", 1))
        sequence = np.load(tensor_path)
        sequence = self._torch.tensor(sequence, dtype=self._torch.float32)
        if self.static_mode:
            sequence = sequence[-1]
        return sequence, self._torch.tensor(label, dtype=self._torch.float32)
