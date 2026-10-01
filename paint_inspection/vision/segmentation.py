from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SegmentationPrediction:
    mask: np.ndarray
    classes: dict[int, str]
    confidence: float


class BaseSegmenter(ABC):
    @abstractmethod
    def load(self, path: str) -> None: ...
    @abstractmethod
    def predict(self, image: np.ndarray) -> SegmentationPrediction: ...


class SegFormerAdapter(BaseSegmenter):
    def load(self, path: str) -> None:
        raise RuntimeError("Install and configure a trained SegFormer checkpoint before enabling this adapter")
    def predict(self, image: np.ndarray) -> SegmentationPrediction:
        raise RuntimeError("SegFormer is not configured")


class YoloSegAdapter(BaseSegmenter):
    def load(self, path: str) -> None:
        from ultralytics import YOLO
        self.model = YOLO(path)
    def predict(self, image: np.ndarray) -> SegmentationPrediction:
        result = self.model.predict(image, verbose=False)[0]
        if result.masks is None:
            return SegmentationPrediction(np.zeros(image.shape[:2], np.uint8), {}, 0.0)
        mask = result.masks.data.max(0).values.cpu().numpy()
        return SegmentationPrediction(mask, result.names, float(result.boxes.conf.max().item()))
