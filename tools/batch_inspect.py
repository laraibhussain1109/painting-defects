"""Headless folder inspection utility."""
import argparse
from pathlib import Path

import cv2

from paint_inspection.config import ProfileManager
from paint_inspection.pipeline import InspectionPipeline


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("folder", type=Path); parser.add_argument("--profile", default="default")
    args = parser.parse_args(); pipeline = InspectionPipeline(ProfileManager().load(args.profile))
    for path in sorted(args.folder.iterdir()):
        image = cv2.imread(str(path))
        if image is not None:
            result, _, _ = pipeline.inspect(image, str(path)); print(f"{path.name},{result.verdict.value},{result.anomaly_score:.4f},{len(result.defects)}")


if __name__ == "__main__": main()
