import cv2
import numpy as np


def normalize_illumination(image: np.ndarray, config: dict, reference: np.ndarray | None = None) -> np.ndarray:
    output = image.copy()
    if config.get("white_balance"):
        means = output.reshape(-1, 3).mean(axis=0)
        output = np.clip(output * (means.mean() / np.maximum(means, 1))[None, None, :], 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
    if config.get("lab_luminance") and reference is not None:
        ref_l = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)[:, :, 0]
        delta = float(ref_l.mean() - lab[:, :, 0].mean())
        lab[:, :, 0] = np.clip(lab[:, :, 0].astype(float) + delta, 0, 255)
    if config.get("clahe"):
        lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
    output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gamma = float(config.get("gamma", 1.0))
    if gamma != 1.0:
        lut = np.array([((i / 255) ** (1 / gamma)) * 255 for i in range(256)], np.uint8)
        output = cv2.LUT(output, lut)
    return output
