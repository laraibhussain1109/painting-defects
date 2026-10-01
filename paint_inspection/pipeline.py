from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import ProductProfile, resolve_path
from .core.types import InspectionResult, Verdict
from .vision.alignment import align
from .vision.anomaly import create_detector
from .vision.classical import colour_inspection, glare_mask, texture_inspection
from .vision.fusion import fuse
from .vision.postprocessing import draw_defects, extract_defects
from .vision.preprocessing import normalize_illumination
from .vision.roi import YoloRoiDetector, build_mask, crop_roi
from .vision.tiling import reconstruct, tile_image

LOG = logging.getLogger(__name__)


class InspectionPipeline:
    def __init__(self, profile: ProductProfile):
        self.profile = profile
        self.yolo = YoloRoiDetector()
        self.anomaly = create_detector(profile.section("anomaly").get("backend", "dinomaly"))
        self.references: list[np.ndarray] = []
        self.load_configured_models()

    def load_configured_models(self) -> None:
        yolo = self.profile.section("yolo").get("model_path", "")
        if yolo and resolve_path(yolo).is_file():
            self.yolo.load(str(resolve_path(yolo)))
        model = self.profile.section("anomaly").get("model_path", "")
        if model and resolve_path(model).is_file():
            self.anomaly.load(resolve_path(model))
        self.references = [img for p in self.profile.data.get("references", [])
                           if (img := cv2.imread(str(resolve_path(p)))) is not None]

    def inspect(self, frame: np.ndarray, image_path: str = "") -> tuple[InspectionResult, np.ndarray, np.ndarray | None]:
        now = datetime.now(timezone.utc).isoformat(); start = time.perf_counter()
        base = InspectionResult(Verdict.UNAVAILABLE, now, self.profile.name, image_path=image_path)
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Camera/image frame is invalid")
            if self.yolo.model is None:
                base.message = "YOLO ROI model is not loaded; inspection cannot safely proceed"
                return base, frame, None
            yc = self.profile.section("yolo")
            detection = self.yolo.detect(frame, float(yc.get("confidence", .5)), float(yc.get("iou", .45)),
                                         int(yc.get("image_size", 1280)), yc.get("expected_class"))
            if detection is None:
                base.message = "Required component ROI was not detected"
                return base, frame, None
            roi, bbox = crop_roi(frame, detection.bbox, int(yc.get("padding", 0)))
            frame_overlay = frame.copy(); cv2.rectangle(frame_overlay, bbox[:2], bbox[2:], (0, 200, 80), 3)
            if not self.anomaly.loaded:
                base.message = "Anomaly model is not loaded; no quality verdict was issued"
                return base, frame_overlay, roi
            aligned, alignment_confidence = roi, 1.0
            ac = self.profile.section("alignment")
            if ac.get("enabled"):
                if not self.references:
                    base.message = "Alignment enabled but no master reference is configured"
                    return base, frame_overlay, roi
                ar = align(roi, self.references[0]); aligned, alignment_confidence = ar.image, ar.confidence
                if alignment_confidence < float(ac.get("minimum_confidence", .35)):
                    base.alignment_confidence = alignment_confidence
                    base.message = "Alignment confidence is below the safe inspection limit"
                    return base, frame_overlay, roi
            rc = self.profile.section("roi")
            mask = build_mask(aligned.shape[:2], rc.get("inspection_polygon"), rc.get("excluded_polygons"))
            processed = normalize_illumination(aligned, self.profile.section("preprocessing"), self.references[0] if self.references else None)
            tc = self.profile.section("tiling"); tiles = tile_image(processed, int(tc.get("tile_size", 512)), float(tc.get("overlap", .2)))
            predictions = []
            batch = int(tc.get("batch_size", 4))
            for start_index in range(0, len(tiles), batch):
                predictions.extend(self.anomaly.predict([t.image for t in tiles[start_index:start_index+batch]]).maps)
            anomaly_map = reconstruct(predictions, tiles, aligned.shape[:2])
            glare = glare_mask(processed, mask, **{k: int(self.profile.section("glare").get(k, v)) for k, v in
                                (("value_threshold", 235), ("saturation_threshold", 45))})
            gc = self.profile.section("glare"); strategy = gc.get("strategy", "REDUCE_ANOMALY_WEIGHT")
            if strategy == "IGNORE": anomaly_map[glare.mask > 0] = 0
            elif strategy == "REDUCE_ANOMALY_WEIGHT": anomaly_map[glare.mask > 0] *= float(gc.get("weight", .25))
            colour = colour_inspection(processed, self.references, mask, float(self.profile.section("colour").get("threshold_delta_e", 8)))
            texture = texture_inspection(processed, self.references, mask, float(self.profile.section("texture").get("threshold", .55)))
            pc = self.profile.section("postprocessing")
            clean, defects = extract_defects(anomaly_map, mask, float(pc.get("pixel_threshold", .62)), int(pc.get("minimum_area_px", 12)),
                                              int(pc.get("critical_area_px", 500)), int(pc.get("morphology_kernel", 3)), colour.map,
                                              texture.map, glare.mask, self.profile.section("calibration").get("pixels_per_mm"))
            area_percent = float(np.mean(clean[mask > 0] > 0) * 100) if np.any(mask) else 0
            anomaly_score = max((d.anomaly_score for d in defects), default=0.0)
            decision = fuse(anomaly_score, area_percent, colour.score, texture.score, glare.metrics["glare_percent"],
                            len(defects), self.profile.section("fusion"), strategy)
            base.verdict, base.anomaly_score, base.colour_score, base.texture_score = decision.verdict, decision.score, colour.score, texture.score
            base.glare_percent, base.alignment_confidence, base.defects = glare.metrics["glare_percent"], alignment_confidence, defects
            base.message = "; ".join(decision.reasons); base.model_version = self.profile.section("anomaly").get("version", "")
            base.yolo_model, base.anomaly_model = self.yolo.path, self.profile.section("anomaly").get("model_path", "")
            base.thresholds = {"anomaly": pc.get("pixel_threshold"), "colour": self.profile.section("colour").get("threshold_delta_e"), "texture": self.profile.section("texture").get("threshold")}
            base.debug_images = {"anomaly": anomaly_map, "mask": mask, "glare": glare.mask, "colour": colour.map, "texture": texture.map}
            return base, frame_overlay, draw_defects(aligned, defects)
        except Exception as exc:
            LOG.exception("Inspection failed")
            base.verdict, base.message = Verdict.ERROR, str(exc)
            return base, frame, None
        finally:
            base.inspection_ms = (time.perf_counter() - start) * 1000
