from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class Tile:
    image: np.ndarray
    x: int
    y: int
    valid_width: int
    valid_height: int


def tile_image(image: np.ndarray, size: int, overlap: float) -> list[Tile]:
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("Tile size must be positive and overlap in [0, 1)")
    h, w = image.shape[:2]
    step = max(1, int(round(size * (1 - overlap))))
    xs = _positions(w, size, step)
    ys = _positions(h, size, step)
    tiles = []
    for y in ys:
        for x in xs:
            crop = image[y:min(y + size, h), x:min(x + size, w)]
            vh, vw = crop.shape[:2]
            if vh != size or vw != size:
                crop = cv2.copyMakeBorder(crop, 0, size - vh, 0, size - vw, cv2.BORDER_REFLECT_101)
            tiles.append(Tile(crop, x, y, vw, vh))
    return tiles


def _positions(length: int, size: int, step: int) -> list[int]:
    if length <= size:
        return [0]
    values = list(range(0, length - size + 1, step))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def reconstruct(maps: list[np.ndarray], tiles: list[Tile], shape: tuple[int, int]) -> np.ndarray:
    if len(maps) != len(tiles) or not maps:
        raise ValueError("Each tile requires one anomaly map")
    total = np.zeros(shape, np.float32)
    weights = np.zeros(shape, np.float32)
    for anomaly, tile in zip(maps, tiles, strict=True):
        anomaly = cv2.resize(anomaly.astype(np.float32), tile.image.shape[1::-1])
        h, w = tile.valid_height, tile.valid_width
        window = _blend_window(tile.image.shape[0], tile.image.shape[1])[:h, :w]
        total[tile.y:tile.y+h, tile.x:tile.x+w] += anomaly[:h, :w] * window
        weights[tile.y:tile.y+h, tile.x:tile.x+w] += window
    return total / np.maximum(weights, 1e-6)


def _blend_window(h: int, w: int) -> np.ndarray:
    wy = np.maximum(np.hanning(h), 0.05) if h > 2 else np.ones(h)
    wx = np.maximum(np.hanning(w), 0.05) if w > 2 else np.ones(w)
    return np.outer(wy, wx).astype(np.float32)

