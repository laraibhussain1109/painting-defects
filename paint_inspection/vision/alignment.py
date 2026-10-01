from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class AlignmentResult:
    image: np.ndarray
    confidence: float
    method: str


def align(image: np.ndarray, reference: np.ndarray) -> AlignmentResult:
    reference = cv2.resize(reference, image.shape[1::-1])
    gray, ref = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1500)
    k1, d1 = orb.detectAndCompute(gray, None); k2, d2 = orb.detectAndCompute(ref, None)
    if d1 is not None and d2 is not None:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
        good = [m for m, n in matches if m.distance < .72 * n.distance]
        if len(good) >= 8:
            src = np.float32([k1[m.queryIdx].pt for m in good]); dst = np.float32([k2[m.trainIdx].pt for m in good])
            matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3)
            confidence = float(inliers.mean()) if inliers is not None else 0.0
            if matrix is not None and confidence >= .35:
                return AlignmentResult(cv2.warpPerspective(image, matrix, image.shape[1::-1]), confidence, "ORB")
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        score, warp = cv2.findTransformECC(ref, gray, warp, cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 75, 1e-5))
        return AlignmentResult(cv2.warpAffine(image, warp, image.shape[1::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP), float(score), "ECC")
    except cv2.error:
        return AlignmentResult(image, 0.0, "FAILED")
