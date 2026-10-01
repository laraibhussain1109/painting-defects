from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class BaseCamera(ABC):
    @abstractmethod
    def open(self) -> None: ...
    @abstractmethod
    def read(self) -> np.ndarray | None: ...
    @abstractmethod
    def close(self) -> None: ...


class OpenCVCamera(BaseCamera):
    def __init__(self, index: int, width: int, height: int, fps: float = 15, exposure: float | None = None):
        self.settings = index, width, height, fps, exposure
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        index, width, height, fps, exposure = self.settings
        self.capture = cv2.VideoCapture(index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width); self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        if exposure is not None: self.capture.set(cv2.CAP_PROP_EXPOSURE, exposure)
        if not self.capture.isOpened():
            self.close(); raise RuntimeError(f"Unable to open camera {index}")

    def read(self) -> np.ndarray | None:
        if self.capture is None: return None
        ok, frame = self.capture.read()
        return frame if ok else None

    def close(self) -> None:
        if self.capture is not None: self.capture.release()
        self.capture = None

