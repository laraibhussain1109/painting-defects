from __future__ import annotations

import importlib.util

from .statistical import StatisticalFeatureDetector


class DinomalyDetector(StatisticalFeatureDetector):
    """Dinomaly adapter boundary with a functional fallback for optional installs."""
    name = "dinomaly"

    @property
    def anomalib_available(self) -> bool:
        return importlib.util.find_spec("anomalib") is not None


class EfficientADDetector(StatisticalFeatureDetector):
    name = "efficientad"


class PatchCoreDetector(StatisticalFeatureDetector):
    name = "patchcore"


def create_detector(name: str):
    choices = {"dinomaly": DinomalyDetector, "efficientad": EfficientADDetector, "patchcore": PatchCoreDetector}
    if name.lower() not in choices:
        raise ValueError(f"Unknown anomaly backend: {name}")
    return choices[name.lower()]()

