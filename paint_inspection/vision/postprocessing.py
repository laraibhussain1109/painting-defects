from __future__ import annotations

import cv2
import numpy as np

from paint_inspection.core.types import Defect


def extract_defects(anomaly: np.ndarray, roi_mask: np.ndarray, threshold: float,
                    minimum_area: int, critical_area: int, kernel_size: int = 3,
                    colour_map: np.ndarray | None = None, texture_map: np.ndarray | None = None,
                    glare: np.ndarray | None = None, pixels_per_mm: float | None = None) -> tuple[np.ndarray, list[Defect]]:
    binary = ((anomaly >= threshold) & (roi_mask > 0)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    defects = []
    clean = np.zeros_like(binary)
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < minimum_area:
            continue
        region = labels == label; clean[region] = 255
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).tolist()
        mean = lambda value: float(value[region].mean()) if value is not None else 0.0
        defect = Defect(f"D{len(defects)+1}", (x, y, w, h), [tuple(p) for p in contour], float(area), w, h,
                        tuple(float(v) for v in centroids[label]), mean(anomaly), mean(colour_map),
                        mean(texture_map), mean(glare) / 255, "CRITICAL" if area >= critical_area else "MINOR")
        if pixels_per_mm and pixels_per_mm > 0:
            defect.width_mm, defect.height_mm = w / pixels_per_mm, h / pixels_per_mm
            defect.area_mm2 = area / (pixels_per_mm ** 2)
        defects.append(defect)
    return clean, defects


def draw_defects(image: np.ndarray, defects: list[Defect]) -> np.ndarray:
    output = image.copy()
    for defect in defects:
        points = np.asarray(defect.contour, np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [points], True, (0, 0, 255), 3, cv2.LINE_AA)
        x, y, _, _ = defect.bbox
        cv2.putText(output, defect.defect_id, (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
    return output

