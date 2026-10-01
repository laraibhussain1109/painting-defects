from abc import ABC, abstractmethod
from enum import Enum, auto

from .core.types import Verdict


class BasePLC(ABC):
    @abstractmethod
    def send(self, verdict: Verdict) -> None: ...
    def send_pass(self): self.send(Verdict.PASS)
    def send_fail(self): self.send(Verdict.FAIL)
    def send_uncertain(self): self.send(Verdict.UNCERTAIN)


class ConsolePLC(BasePLC):
    def send(self, verdict: Verdict) -> None: print(f"PLC OUTPUT: {verdict.value}")


class AutoState(Enum):
    WAITING_FOR_PART = auto(); PART_DETECTED = auto(); PART_STABLE = auto(); INSPECTING = auto(); RESULT_DISPLAY = auto(); WAITING_FOR_PART_REMOVAL = auto()


class PartStabilityMachine:
    def __init__(self, stable_frames: int = 5, removal_frames: int = 5):
        self.stable_frames, self.removal_frames = stable_frames, removal_frames; self.state = AutoState.WAITING_FOR_PART; self.count = 0
    def update(self, detected: bool) -> bool:
        if self.state in (AutoState.WAITING_FOR_PART, AutoState.PART_DETECTED):
            self.count = self.count + 1 if detected else 0; self.state = AutoState.PART_DETECTED if detected else AutoState.WAITING_FOR_PART
            if self.count >= self.stable_frames: self.state = AutoState.INSPECTING; self.count = 0; return True
        if self.state in (AutoState.RESULT_DISPLAY, AutoState.WAITING_FOR_PART_REMOVAL):
            self.count = self.count + 1 if not detected else 0; self.state = AutoState.WAITING_FOR_PART_REMOVAL
            if self.count >= self.removal_frames: self.state = AutoState.WAITING_FOR_PART; self.count = 0
        return False
    def completed(self): self.state = AutoState.RESULT_DISPLAY
