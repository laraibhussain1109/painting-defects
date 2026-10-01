from datetime import datetime
import json

import numpy as np
from skimage.color import deltaE_ciede2000

from paint_inspection.config import ProfileManager
from paint_inspection.core.types import InspectionResult, Verdict
from paint_inspection.vision.fusion import fuse
from paint_inspection.vision.postprocessing import extract_defects
from paint_inspection.vision.roi import build_mask, crop_roi
from paint_inspection.vision.tiling import reconstruct, tile_image


def test_roi_crop_clips_and_pads():
    image = np.zeros((50, 100, 3), np.uint8)
    crop, box = crop_roi(image, (10, 10, 30, 40), 15)
    assert box == (0, 0, 45, 50) and crop.shape == (50, 45, 3)


def test_mask_polygon_and_exclusion():
    mask = build_mask((20, 20), [[1,1],[18,1],[18,18],[1,18]], [[[5,5],[10,5],[10,10],[5,10]]])
    assert mask[2,2] == 255 and mask[7,7] == 0 and mask[0,0] == 0


def test_tiling_reconstructs_constant_image():
    image = np.zeros((713, 901, 3), np.uint8); tiles = tile_image(image, 256, .2)
    output = reconstruct([np.full((256, 256), .37, np.float32) for _ in tiles], tiles, image.shape[:2])
    assert output.shape == image.shape[:2] and np.allclose(output, .37, atol=1e-5)


def test_delta_e_identity_and_known_difference():
    a = np.array([[[50., 0., 0.]]]); b = np.array([[[50., 0., 0.]]]); c = np.array([[[50., 20., 0.]]])
    assert deltaE_ciede2000(a, b).item() == 0 and deltaE_ciede2000(a, c).item() > 10


def test_connected_components_and_size_filter():
    anomaly = np.zeros((50,50), np.float32); anomaly[2:4,2:4] = 1; anomaly[20:30,20:30] = .9
    clean, defects = extract_defects(anomaly, np.full((50,50),255,np.uint8), .5, 10, 80, 1)
    assert len(defects) == 1 and defects[0].area_px == 100 and defects[0].severity == "CRITICAL" and clean.sum() == 25500


def test_fusion_requires_corroboration_for_fail():
    config = {"pass_threshold": .3, "fail_threshold": .67, "anomaly_weight": .5, "colour_weight": .25, "texture_weight": .25}
    assert fuse(.99, 10, 0, 0, 0, 1, config).verdict is Verdict.UNCERTAIN
    assert fuse(.99, 10, .99, .8, 0, 1, config).verdict is Verdict.FAIL
    assert fuse(0, 0, 0, 0, 0, 0, config).verdict is Verdict.PASS


def test_profile_loading(tmp_path):
    data = {"name":"part", "fusion": {"fail_threshold": .8}}; (tmp_path/"part.json").write_text(json.dumps(data))
    profile = ProfileManager(tmp_path).load("part")
    assert profile.name == "part" and profile.section("fusion")["fail_threshold"] == .8


def test_result_serialization():
    result = InspectionResult(Verdict.PASS, datetime.now().isoformat(), "part")
    encoded = json.dumps(result.to_dict())
    assert '"verdict": "PASS"' in encoded and "debug_images" not in encoded

