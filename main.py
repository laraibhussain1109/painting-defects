from __future__ import annotations

import logging
from pathlib import Path

from paint_inspection.gui import run_gui


def configure_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        handlers=[logging.FileHandler("logs/paint_inspection.log", encoding="utf-8"), logging.StreamHandler()])


if __name__ == "__main__":
    configure_logging()
    raise SystemExit(run_gui())

