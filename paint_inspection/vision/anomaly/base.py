from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(slots=True)
class AnomalyPrediction:
    maps: list[np.ndarray]
    scores: list[float]


class BaseAnomalyDetector(ABC):
    name = "base"

    @abstractmethod
    def train(self, good_directory: Path, output: Path, progress: Callable[[str], None] | None = None) -> dict: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...

    @abstractmethod
    def predict(self, images: list[np.ndarray]) -> AnomalyPrediction: ...

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

