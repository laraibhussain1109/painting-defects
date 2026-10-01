from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int


class YoloRoiDetector:
    def __init__(self) -> None:
        self.model = None
        self.path = ""

    def load(self, path: str) -> None:
        if not Path(path).is_file():
            raise FileNotFoundError(f"YOLO model not found: {path}")
        from ultralytics import YOLO

        self.model = YOLO(path)
        self.path = path

    def detect(self, image: np.ndarray, confidence: float, iou: float, image_size: int,
               expected_class: int | None = None) -> Detection | None:
        if self.model is None:
            raise RuntimeError("YOLO ROI model is not loaded")
        result = self.model.predict(image, conf=confidence, iou=iou, imgsz=image_size, verbose=False)[0]
        candidates: list[Detection] = []
        for box in result.boxes:
            cls = int(box.cls.item())
            if expected_class is not None and cls != expected_class:
                continue
            xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
            candidates.append(Detection(xyxy, float(box.conf.item()), cls))
        return max(candidates, key=lambda d: (d.confidence, area(d.bbox)), default=None)


def area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def crop_roi(image: np.ndarray, bbox: tuple[int, int, int, int], padding: int = 0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    clipped = (max(0, x1 - padding), max(0, y1 - padding), min(w, x2 + padding), min(h, y2 + padding))
    x1, y1, x2, y2 = clipped
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI is empty or outside the image")
    return image[y1:y2, x1:x2].copy(), clipped


def build_mask(shape: tuple[int, int], inspection_polygon: list[list[int]] | None = None,
               excluded_polygons: list[list[list[int]]] | None = None) -> np.ndarray:
    mask = np.zeros(shape, np.uint8) if inspection_polygon else np.full(shape, 255, np.uint8)
    if inspection_polygon:
        cv2.fillPoly(mask, [np.asarray(inspection_polygon, np.int32)], 255)
    for polygon in excluded_polygons or []:
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 0)
    return mask

