from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"
    UNAVAILABLE = "INSPECTION UNAVAILABLE"
    ERROR = "SYSTEM ERROR"


@dataclass(slots=True)
class Defect:
    defect_id: str
    bbox: tuple[int, int, int, int]
    contour: list[tuple[int, int]]
    area_px: float
    width_px: int
    height_px: int
    center: tuple[float, float]
    anomaly_score: float
    colour_score: float = 0.0
    texture_score: float = 0.0
    glare_fraction: float = 0.0
    severity: str = "MINOR"
    width_mm: float | None = None
    height_mm: float | None = None
    area_mm2: float | None = None


@dataclass(slots=True)
class InspectionResult:
    verdict: Verdict
    timestamp: str
    product: str
    anomaly_score: float = 0.0
    colour_score: float = 0.0
    texture_score: float = 0.0
    glare_percent: float = 0.0
    alignment_confidence: float = 1.0
    inspection_ms: float = 0.0
    defects: list[Defect] = field(default_factory=list)
    image_path: str = ""
    roi_path: str = ""
    model_version: str = ""
    yolo_model: str = ""
    anomaly_model: str = ""
    thresholds: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    debug_images: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        data.pop("debug_images", None)
        return data

