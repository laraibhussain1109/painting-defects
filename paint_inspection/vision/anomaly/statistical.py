from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .base import AnomalyPrediction, BaseAnomalyDetector


class StatisticalFeatureDetector(BaseAnomalyDetector):
    """Reliable CPU fallback using multi-scale colour/gradient normal statistics.

    This keeps inspection operational when optional Anomalib backends are absent. It is
    deliberately explicit in metadata and never pretends to be a neural checkpoint.
    """
    name = "statistical-fallback"

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.metadata: dict = {}

    @property
    def loaded(self) -> bool:
        return self.mean is not None

    @staticmethod
    def features(image: np.ndarray, size: int = 256) -> np.ndarray:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        local = np.sqrt(np.maximum(cv2.blur(gray * gray, (7, 7)) - cv2.blur(gray, (7, 7)) ** 2, 0))
        return np.dstack((lab, grad, local))

    def train(self, good_directory: Path, output: Path, progress: Callable[[str], None] | None = None) -> dict:
        paths = [p for p in good_directory.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
        if not paths:
            raise ValueError(f"No GOOD images found in {good_directory}")
        start = time.perf_counter(); features = []
        for index, path in enumerate(paths, 1):
            image = cv2.imread(str(path))
            if image is not None:
                features.append(self.features(image))
            if progress:
                progress(f"Feature modelling {index}/{len(paths)}")
        stack = np.stack(features)
        self.mean = stack.mean(axis=0); self.std = np.maximum(stack.std(axis=0), .025)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, mean=self.mean, std=self.std)
        self.metadata = {"model_name": output.stem, "model_type": self.name, "version": "1.0",
                         "training_date": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset_path": str(good_directory),
                         "number_of_images": len(features), "input_resolution": 256,
                         "training_seconds": round(time.perf_counter() - start, 3)}
        output.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        return self.metadata

    def load(self, path: Path) -> None:
        data = np.load(path)
        self.mean, self.std = data["mean"], data["std"]
        meta = path.with_suffix(".json")
        self.metadata = json.loads(meta.read_text()) if meta.exists() else {"version": "unknown"}

    def predict(self, images: list[np.ndarray]) -> AnomalyPrediction:
        if not self.loaded:
            raise RuntimeError("Anomaly model is not loaded")
        maps, scores = [], []
        for image in images:
            feature = self.features(image, self.mean.shape[0])
            distance = np.sqrt(np.mean(((feature - self.mean) / self.std) ** 2, axis=2))
            anomaly = np.clip(distance / 6, 0, 1).astype(np.float32)
            anomaly = cv2.resize(anomaly, image.shape[1::-1], interpolation=cv2.INTER_CUBIC)
            maps.append(anomaly); scores.append(float(np.percentile(anomaly, 99)))
        return AnomalyPrediction(maps, scores)

