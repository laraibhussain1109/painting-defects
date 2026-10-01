from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from .core.types import InspectionResult


class InspectionDatabase:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True); self.path = path
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY, timestamp TEXT, product TEXT, image_path TEXT, roi_path TEXT,
                verdict TEXT, anomaly_score REAL, defect_count INTEGER, maximum_defect_size REAL,
                colour_score REAL, texture_score REAL, thresholds TEXT, model_version TEXT,
                yolo_model TEXT, anomaly_model TEXT, inspection_ms REAL, payload TEXT)""")

    def connect(self): return sqlite3.connect(self.path)

    def add(self, result: InspectionResult) -> int:
        payload = result.to_dict(); maximum = max((d.area_px for d in result.defects), default=0)
        values = (result.timestamp, result.product, result.image_path, result.roi_path, result.verdict.value,
                  result.anomaly_score, len(result.defects), maximum, result.colour_score, result.texture_score,
                  json.dumps(result.thresholds), result.model_version, result.yolo_model, result.anomaly_model,
                  result.inspection_ms, json.dumps(payload))
        with self.connect() as db:
            cursor = db.execute("INSERT INTO inspections (timestamp,product,image_path,roi_path,verdict,anomaly_score,defect_count,maximum_defect_size,colour_score,texture_score,thresholds,model_version,yolo_model,anomaly_model,inspection_ms,payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
            return int(cursor.lastrowid)

    def export_csv(self, destination: Path) -> None:
        with self.connect() as db, destination.open("w", newline="", encoding="utf-8") as handle:
            cursor = db.execute("SELECT * FROM inspections ORDER BY id")
            writer = csv.writer(handle); writer.writerow([d[0] for d in cursor.description]); writer.writerows(cursor)

