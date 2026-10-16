from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(slots=True)
class ProjectPaths:
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    events_csv: Path
    sequences_csv: Path
    tensor_dir: Path


@dataclass(slots=True)
class DataConfig:
    sequence_length: int
    image_size: int
    channels: list[str]
    full_channels: list[str]
    frame_offsets_days: list[int]
    max_cloud_cover: float
    train_regions: list[str]
    ood_regions: list[str]
    train_fraction: float
    val_fraction: float
    test_fraction: float
    cluster_radius_km: float
    cluster_days: int
    hard_negatives_per_positive: int
    hard_negative_min_distance_km: float
    hard_negative_max_distance_km: float
    hard_negative_day_jitter: int


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    dropout: float
    early_stopping_patience: int
    positive_class_weight: float


@dataclass(slots=True)
class EvaluationConfig:
    mc_passes: int
    ece_bins: int
    threshold_min: float = 0.2
    threshold_max: float = 0.8
    threshold_selection: str = "f1"
    target_recall: float = 0.85


@dataclass(slots=True)
class AppConfig:
    name: str
    seed: int
    paths: ProjectPaths
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    firms: dict
    sentinel: dict


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(config_path: str | Path) -> AppConfig:
    config_file = Path(config_path).resolve()
    with config_file.open("rb") as handle:
        raw = tomllib.load(handle)

    base = config_file.parent.parent if config_file.parent.name == "configs" else config_file.parent
    paths = raw["paths"]
    project = raw["project"]
    data = raw["data"]
    training = raw["training"]
    evaluation = raw["evaluation"]

    return AppConfig(
        name=project["name"],
        seed=int(project["seed"]),
        paths=ProjectPaths(
            raw_dir=_resolve(base, paths["raw_dir"]),
            processed_dir=_resolve(base, paths["processed_dir"]),
            artifacts_dir=_resolve(base, paths["artifacts_dir"]),
            events_csv=_resolve(base, paths["events_csv"]),
            sequences_csv=_resolve(base, paths["sequences_csv"]),
            tensor_dir=_resolve(base, paths["tensor_dir"]),
        ),
        data=DataConfig(**data),
        training=TrainingConfig(**training),
        evaluation=EvaluationConfig(**evaluation),
        firms=raw["firms"],
        sentinel=raw["sentinel"],
    )
