from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000


@dataclass(slots=True)
class BranchResult:
    score: float
    mask: np.ndarray
    map: np.ndarray
    metrics: dict[str, float]


def colour_inspection(image: np.ndarray, references: list[np.ndarray], mask: np.ndarray,
                      threshold: float = 8.0) -> BranchResult:
    if not references:
        zeros = np.zeros(image.shape[:2], np.float32)
        return BranchResult(0.0, zeros.astype(np.uint8), zeros, {"unavailable": 1.0})
    target = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_labs = []
    for ref in references:
        resized = cv2.resize(ref, image.shape[1::-1])
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] += target[:, :, 0].mean() - lab[:, :, 0].mean()
        ref_labs.append(lab)
    reference = np.median(ref_labs, axis=0).astype(np.float32)
    # OpenCV's encoded LAB ranges must be converted to standard CIELAB.
    convert = lambda x: np.dstack((x[:, :, 0] * 100 / 255, x[:, :, 1] - 128, x[:, :, 2] - 128))
    delta = deltaE_ciede2000(convert(target), convert(reference)).astype(np.float32)
    valid = mask > 0
    abnormal = ((delta >= threshold) & valid).astype(np.uint8) * 255
    values = delta[valid]
    area = float(np.mean(abnormal[valid] > 0) * 100) if np.any(valid) else 0.0
    score = float(np.clip(np.percentile(values, 95) / max(threshold * 2, 1), 0, 1)) if values.size else 0.0
    return BranchResult(score, abnormal, delta, {"mean_delta_e": float(values.mean()) if values.size else 0.0,
                         "max_local_delta_e": float(values.max()) if values.size else 0.0,
                         "abnormal_colour_area_percent": area})


def texture_inspection(image: np.ndarray, references: list[np.ndarray], mask: np.ndarray,
                       threshold: float = 0.55) -> BranchResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    gradient = cv2.magnitude(cv2.Scharr(gray, cv2.CV_32F, 1, 0), cv2.Scharr(gray, cv2.CV_32F, 0, 1))
    mean = cv2.blur(gray, (9, 9)); variance = np.maximum(cv2.blur(gray * gray, (9, 9)) - mean * mean, 0)
    feature = cv2.GaussianBlur(0.55 * gradient + 0.45 * np.sqrt(variance), (5, 5), 0)
    if references:
        ref_features = []
        for ref in references:
            r = cv2.resize(ref, image.shape[1::-1]); rg = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            gm = cv2.magnitude(cv2.Scharr(rg, cv2.CV_32F, 1, 0), cv2.Scharr(rg, cv2.CV_32F, 0, 1))
            rm = cv2.blur(rg, (9, 9)); rv = np.maximum(cv2.blur(rg * rg, (9, 9)) - rm * rm, 0)
            ref_features.append(cv2.GaussianBlur(.55 * gm + .45 * np.sqrt(rv), (5, 5), 0))
        abnormality = np.abs(feature - np.median(ref_features, axis=0))
    else:
        abnormality = np.abs(feature - cv2.GaussianBlur(feature, (0, 0), 15))
    scale = np.percentile(abnormality[mask > 0], 99) if np.any(mask) else 1
    normalized = np.clip(abnormality / max(scale, 1e-6), 0, 1).astype(np.float32)
    score = float(np.percentile(normalized[mask > 0], 95)) if np.any(mask) else 0.0
    return BranchResult(score, ((normalized >= threshold) & (mask > 0)).astype(np.uint8) * 255,
                        normalized, {"texture_abnormality": score})


def glare_mask(image: np.ndarray, mask: np.ndarray, value_threshold: int = 235,
               saturation_threshold: int = 45) -> BranchResult:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright = (hsv[:, :, 2] >= value_threshold) & (hsv[:, :, 1] <= saturation_threshold)
    local = cv2.Laplacian(hsv[:, :, 2], cv2.CV_32F)
    glare = bright & (np.abs(local) > 2) & (mask > 0)
    out = cv2.morphologyEx(glare.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    percent = float(np.mean(out[mask > 0] > 0) * 100) if np.any(mask) else 0.0
    return BranchResult(min(percent / 20, 1), out, out.astype(np.float32) / 255, {"glare_percent": percent})

