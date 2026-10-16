from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date


VALID_SPLITS = {"train", "val", "test", "ood_test"}


@dataclass(slots=True)
class EventRecord:
    event_id: str
    lat: float
    lon: float
    date: str
    region: str
    label: int
    source: str
    confidence: str
    split: str

    def validate(self) -> None:
        if not -90 <= self.lat <= 90:
            raise ValueError(f"Invalid latitude: {self.lat}")
        if not -180 <= self.lon <= 180:
            raise ValueError(f"Invalid longitude: {self.lon}")
        date.fromisoformat(self.date)
        if self.label not in (0, 1):
            raise ValueError(f"Invalid label: {self.label}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Invalid split: {self.split}")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class SequenceRecord:
    sequence_id: str
    event_id: str
    t0_date: str
    t1_date: str
    t2_date: str
    patch_id: str
    label: int
    split: str

    def validate(self) -> None:
        for item in (self.t0_date, self.t1_date, self.t2_date):
            date.fromisoformat(item)
        if self.label not in (0, 1):
            raise ValueError(f"Invalid label: {self.label}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Invalid split: {self.split}")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)
