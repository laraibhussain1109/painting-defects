from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from .vision.anomaly import BaseAnomalyDetector


def calibrate_threshold(good_scores: list[float], bad_scores: list[float] | None = None) -> dict:
    if not good_scores: raise ValueError("At least one GOOD validation score is required")
    good = np.asarray(good_scores); report = {"mean": float(good.mean()), "std": float(good.std()),
        "p95": float(np.percentile(good, 95)), "p99": float(np.percentile(good, 99)), "maximum": float(good.max())}
    report["recommended_threshold"] = float(min(1, max(np.percentile(good, 99), good.mean() + 3 * good.std())))
    if bad_scores:
        scores = np.r_[good, bad_scores]; labels = np.r_[np.zeros(len(good)), np.ones(len(bad_scores))]
        candidates = np.unique(scores); best = max(candidates, key=lambda t: f1_score(labels, scores >= t))
        predicted = scores >= best; tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
        report.update(recommended_threshold=float(best), tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
                      precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)),
                      f1=float(f1_score(labels, predicted)))
    return report


def validate_directory(detector: BaseAnomalyDetector, directory: Path) -> list[float]:
    scores = []
    for path in directory.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image = cv2.imread(str(path))
            if image is not None: scores.extend(detector.predict([image]).scores)
    return scores


def write_training_report(path: Path, metadata: dict, calibration: dict) -> None:
    path.write_text(json.dumps({**metadata, "normal_anomaly_distribution": calibration}, indent=2), encoding="utf-8")

