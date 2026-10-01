from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton, QSplitter,
    QStatusBar, QTableWidget, QTableWidgetItem, QTextEdit, QToolBar, QVBoxLayout, QWidget)

from .camera import OpenCVCamera
from .config import ProfileManager, ROOT, load_yaml, resolve_path
from .core.types import InspectionResult, Verdict
from .database import InspectionDatabase
from .pipeline import InspectionPipeline
from .training import calibrate_threshold, validate_directory, write_training_report

LOG = logging.getLogger(__name__)


def pixmap(image: np.ndarray | None, size) -> QPixmap:
    if image is None: return QPixmap()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); h, w, c = rgb.shape
    qimage = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage).scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


class PolygonCanvas(QLabel):
    def __init__(self, image: np.ndarray, inspection: list, exclusions: list):
        super().__init__(); self.image = image; self.inspection = [list(p) for p in inspection]; self.exclusions = [[list(p) for p in poly] for poly in exclusions]
        self.mode = "inspection"; self.current: list[list[int]] = []; self.setMinimumSize(800, 500); self.setMouseTracking(True)
    def mousePressEvent(self, event: QMouseEvent):
        shown = pixmap(self.image, self.size()); ox, oy = (self.width()-shown.width())//2, (self.height()-shown.height())//2
        if ox <= event.position().x() < ox+shown.width() and oy <= event.position().y() < oy+shown.height():
            x = int((event.position().x()-ox) * self.image.shape[1]/shown.width()); y = int((event.position().y()-oy) * self.image.shape[0]/shown.height())
            self.current.append([x, y]); self.update()
    def paintEvent(self, event):
        super().paintEvent(event); shown = pixmap(self.image, self.size()); ox, oy = (self.width()-shown.width())//2, (self.height()-shown.height())//2
        painter = QPainter(self); painter.drawPixmap(ox, oy, shown); sx, sy = shown.width()/self.image.shape[1], shown.height()/self.image.shape[0]
        for points, color in [(self.inspection, "#45d879"), *[(p, "#ff5058") for p in self.exclusions], (self.current, "#ffb83e")]:
            if len(points) > 1:
                painter.setPen(QPen(QColor(color), 3)); mapped = [(int(ox+x*sx), int(oy+y*sy)) for x,y in points]
                for a,b in zip(mapped, mapped[1:]): painter.drawLine(*a, *b)
                if points is not self.current and len(mapped)>2: painter.drawLine(*mapped[-1], *mapped[0])
    def finish_polygon(self):
        if len(self.current) < 3: return
        if self.mode == "inspection": self.inspection = self.current
        else: self.exclusions.append(self.current)
        self.current = []; self.update()


class RoiEditorDialog(QDialog):
    def __init__(self, image, roi_config, parent=None):
        super().__init__(parent); self.setWindowTitle("ROI Mask & Exclusion Editor"); self.resize(1000, 700)
        layout = QVBoxLayout(self); self.canvas = PolygonCanvas(image, roi_config.get("inspection_polygon", []), roi_config.get("excluded_polygons", [])); layout.addWidget(self.canvas)
        row = QHBoxLayout(); mode = QComboBox(); mode.addItems(["INSPECTION POLYGON", "EXCLUSION REGION"]); mode.currentIndexChanged.connect(lambda i: setattr(self.canvas, "mode", "inspection" if i == 0 else "exclusion")); row.addWidget(mode)
        finish = QPushButton("FINISH POLYGON"); finish.clicked.connect(self.canvas.finish_polygon); row.addWidget(finish)
        undo = QPushButton("UNDO POINT"); undo.clicked.connect(lambda: (self.canvas.current.pop() if self.canvas.current else None, self.canvas.update())); row.addWidget(undo)
        clear = QPushButton("CLEAR EXCLUSIONS"); clear.clicked.connect(lambda: (self.canvas.exclusions.clear(), self.canvas.update())); row.addWidget(clear)
        save = QPushButton("SAVE"); save.clicked.connect(self.accept); row.addWidget(save); cancel = QPushButton("CANCEL"); cancel.clicked.connect(self.reject); row.addWidget(cancel); layout.addLayout(row)


class InspectionWorker(QThread):
    completed = pyqtSignal(object, object, object); failed = pyqtSignal(str)
    def __init__(self, pipeline, frame, path=""):
        super().__init__(); self.pipeline, self.frame, self.path = pipeline, frame.copy(), path
    def run(self):
        try: self.completed.emit(*self.pipeline.inspect(self.frame, self.path))
        except Exception as exc: self.failed.emit(str(exc))


class TrainingWorker(QThread):
    progress = pyqtSignal(str); completed = pyqtSignal(str, object); failed = pyqtSignal(str)
    def __init__(self, pipeline, dataset: Path, output: Path):
        super().__init__(); self.pipeline, self.dataset, self.output = pipeline, dataset, output
    def run(self):
        try:
            metadata = self.pipeline.anomaly.train(self.dataset / "good", self.output, self.progress.emit)
            validation_good = self.dataset / "validation_good"
            good = validate_directory(self.pipeline.anomaly, validation_good if validation_good.exists() else self.dataset / "good")
            bad = validate_directory(self.pipeline.anomaly, self.dataset / "validation_bad") if (self.dataset / "validation_bad").exists() else []
            calibration = calibrate_threshold(good, bad); report = self.output.with_name(self.output.stem + "_report.json")
            write_training_report(report, metadata, calibration); self.completed.emit(str(self.output), calibration)
        except Exception as exc: self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Paint Inspection System"); self.resize(1500, 900)
        self.manager = ProfileManager(); app = load_yaml("config/app.yaml"); self.profile = self.manager.load(app.get("active_product", "default"))
        self.pipeline = InspectionPipeline(self.profile); self.database = InspectionDatabase(resolve_path(app["database"]))
        self.camera = None; self.frame = None; self.last_roi = None; self.current_path = ""; self.files: list[Path] = []; self.file_index = -1
        self.worker = None; self.training_worker = None; self.developer = False
        self.timer = QTimer(self); self.timer.timeout.connect(self._camera_tick)
        self._build(); self._style(); self._device_status()

    def _build(self):
        toolbar = QToolBar("Operator controls"); toolbar.setMovable(False); self.addToolBar(toolbar)
        actions = [("START CAMERA", self.start_camera), ("STOP CAMERA", self.stop_camera), ("OPEN IMAGE", self.open_image),
                   ("OPEN FOLDER", self.open_folder), ("PREVIOUS", lambda: self.navigate(-1)), ("NEXT", lambda: self.navigate(1)),
                   ("INSPECT", self.inspect), ("TRAIN ANOMALY MODEL", self.train), ("LOAD MODEL", self.load_anomaly),
                   ("SELECT YOLO MODEL", self.select_yolo), ("PRODUCT PROFILE", self.profile_dialog),
                   ("CALIBRATION", self.calibration), ("ROI EDITOR", self.roi_editor),
                   ("IMPORT GOOD IMAGES", lambda: self.import_images("good")), ("IMPORT BAD TEST IMAGES", lambda: self.import_images("validation_bad")),
                   ("SETTINGS", self.settings), ("SAVE DEBUG OUTPUT", self.save_debug), ("VIEW LOGS", self.view_logs), ("EXIT", self.close)]
        for text, slot in actions:
            action = QAction(text, self); action.triggered.connect(slot); toolbar.addAction(action)
        self.auto = QCheckBox("AUTO INSPECTION"); toolbar.addWidget(self.auto)
        dev = QCheckBox("DEVELOPER MODE"); dev.toggled.connect(self._developer); toolbar.addWidget(dev)
        root = QWidget(); layout = QVBoxLayout(root); self.setCentralWidget(root)
        title = QLabel("HYBRID AI PAINT DEFECT INSPECTION"); title.setObjectName("title"); title.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(title)
        images = QSplitter(); self.live = self._image_panel("CAMERA / SOURCE"); self.roi = self._image_panel("INSPECTION ROI"); images.addWidget(self.live.parentWidget()); images.addWidget(self.roi.parentWidget()); layout.addWidget(images, 4)
        lower = QHBoxLayout(); self.verdict = QLabel("SYSTEM READY"); self.verdict.setObjectName("verdict"); self.verdict.setAlignment(Qt.AlignmentFlag.AlignCenter); lower.addWidget(self.verdict, 2)
        self.details = QTableWidget(0, 2); self.details.setHorizontalHeaderLabels(["MEASUREMENT", "VALUE"]); self.details.horizontalHeader().setStretchLastSection(True); lower.addWidget(self.details, 2)
        self.debug = QTextEdit(); self.debug.setReadOnly(True); self.debug.hide(); lower.addWidget(self.debug, 2); layout.addLayout(lower, 2)
        self.setStatusBar(QStatusBar()); self.statusBar().showMessage(f"Product: {self.profile.name} | Camera stopped")

    def _image_panel(self, name):
        box = QWidget(); layout = QVBoxLayout(box); caption = QLabel(name); caption.setObjectName("caption"); view = QLabel("No image"); view.setAlignment(Qt.AlignmentFlag.AlignCenter); view.setMinimumSize(500, 420); view.setObjectName("viewer"); layout.addWidget(caption); layout.addWidget(view); return view

    def _style(self):
        self.setStyleSheet("""QMainWindow,QWidget{background:#111418;color:#e8edf2;font:13px 'Segoe UI'} QToolBar{background:#20262c;border-bottom:2px solid #38b76b;padding:7px;spacing:5px} QToolButton,QPushButton{background:#293138;border:1px solid #46515b;border-radius:4px;padding:8px;color:white} QToolButton:hover,QPushButton:hover{background:#34764e} #title{font-size:25px;font-weight:700;color:#62d58a;padding:12px} #caption{font-size:15px;font-weight:600;color:#b9c5cf;padding:6px} #viewer{background:#090b0d;border:1px solid #39434c} #verdict{font-size:42px;font-weight:800;background:#252b31;border-radius:8px;margin:8px} QTableWidget,QTextEdit{background:#181d22;border:1px solid #39434c;gridline-color:#303940} QHeaderView::section{background:#293138;padding:6px} QStatusBar{background:#20262c} QCheckBox{padding:8px} """)

    def _device_status(self):
        try:
            import torch
            device = f"CUDA: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU fallback"
        except ImportError: device = "CPU fallback (PyTorch unavailable)"
        self.statusBar().showMessage(f"Product: {self.profile.name} | {device} | Camera stopped")

    def start_camera(self):
        try:
            c = load_yaml("config/camera.yaml"); self.camera = OpenCVCamera(c["index"], c["width"], c["height"], c["fps"], c.get("exposure")); self.camera.open(); self.timer.start(max(1, int(1000/c["fps"])))
        except Exception as exc: self.error(str(exc))
    def stop_camera(self):
        self.timer.stop()
        if self.camera: self.camera.close()
        self.camera = None
    def _camera_tick(self):
        frame = self.camera.read() if self.camera else None
        if frame is not None: self.frame = frame; self.live.setPixmap(pixmap(frame, self.live.size()))
        if self.auto.isChecked() and self.worker is None: self.inspect()
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open inspection image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path: self._load_path(Path(path))
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open image folder")
        if folder:
            self.files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}); self.file_index = 0
            if self.files: self._load_path(self.files[0])
    def navigate(self, delta):
        if self.files: self.file_index = (self.file_index + delta) % len(self.files); self._load_path(self.files[self.file_index])
    def _load_path(self, path):
        frame = cv2.imread(str(path))
        if frame is None: return self.error(f"Could not read {path}")
        self.frame, self.current_path = frame, str(path); self.live.setPixmap(pixmap(frame, self.live.size()))
    def inspect(self):
        if self.frame is None: return self.error("Start the camera or open an image first")
        if self.worker and self.worker.isRunning(): return
        self.verdict.setText("INSPECTING"); self.verdict.setStyleSheet("color:#5bbcff"); self.worker = InspectionWorker(self.pipeline, self.frame, self.current_path)
        self.worker.completed.connect(self._result); self.worker.failed.connect(self.error); self.worker.finished.connect(lambda: setattr(self, "worker", None)); self.worker.start()
    def _result(self, result: InspectionResult, frame, roi):
        self.live.setPixmap(pixmap(frame, self.live.size())); self.roi.setPixmap(pixmap(roi, self.roi.size())); self.last_roi = roi.copy() if roi is not None else None
        colors = {Verdict.PASS:"#48d178", Verdict.FAIL:"#ff4f57", Verdict.UNCERTAIN:"#ffb83e", Verdict.ERROR:"#ff4f57", Verdict.UNAVAILABLE:"#ffb83e"}
        self.verdict.setText(result.verdict.value); self.verdict.setStyleSheet(f"color:{colors[result.verdict]};background:#252b31")
        rows = [("Product", result.product), ("Defects", len(result.defects)), ("Evidence score", f"{result.anomaly_score:.3f}"), ("Colour", f"{result.colour_score:.3f}"), ("Texture", f"{result.texture_score:.3f}"), ("Time", f"{result.inspection_ms:.1f} ms"), ("Message", result.message)]
        self.details.setRowCount(len(rows))
        for i, pair in enumerate(rows):
            for j, value in enumerate(pair): self.details.setItem(i, j, QTableWidgetItem(str(value)))
        self.last_result = result
        if load_yaml("config/app.yaml").get("save_results", True) and roi is not None:
            category = result.verdict.value.lower().replace("inspection ", "").replace(" ", "_")
            folder = ROOT / "inspection_results" / result.timestamp[:10] / category; folder.mkdir(parents=True, exist_ok=True)
            stem = Path(result.image_path).stem if result.image_path else result.timestamp[11:19].replace(":", "-")
            output = folder / f"{stem}.jpg"; cv2.imwrite(str(output), roi); result.roi_path = str(output)
        self.database.add(result)
        if self.developer: self.debug.setText("\n".join(f"{k}: shape={v.shape}, range={v.min():.3f}..{v.max():.3f}" for k,v in result.debug_images.items()))
    def train(self):
        dataset = resolve_path(self.profile.data.get("dataset_path", f"dataset/{self.profile.name}")); output = ROOT/"models"/"anomaly"/f"{self.profile.name}_v1.npz"
        self.training_worker = TrainingWorker(self.pipeline, dataset, output); self.training_worker.progress.connect(lambda m: self.statusBar().showMessage(m)); self.training_worker.failed.connect(self.error); self.training_worker.completed.connect(self._trained); self.training_worker.start()
    def _trained(self, path, calibration):
        a = self.profile.section("anomaly"); a["model_path"] = str(Path(path).relative_to(ROOT)); a["version"] = "1.0"
        self.profile.section("postprocessing")["pixel_threshold"] = calibration["recommended_threshold"]; self.profile.save(); self.pipeline.anomaly.load(Path(path)); QMessageBox.information(self, "Training complete", f"Saved {path}\nRecommended threshold: {calibration['recommended_threshold']:.3f}")
    def load_anomaly(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load anomaly model", str(ROOT/"models"/"anomaly"), "Models (*.npz)")
        if path: self.pipeline.anomaly.load(Path(path)); self.profile.section("anomaly")["model_path"] = str(Path(path).relative_to(ROOT) if Path(path).is_relative_to(ROOT) else path); self.profile.save()
    def select_yolo(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO ROI model", str(ROOT/"models"/"yolo"), "PyTorch models (*.pt)")
        if path:
            try: self.pipeline.yolo.load(path); self.profile.section("yolo")["model_path"] = str(Path(path).relative_to(ROOT) if Path(path).is_relative_to(ROOT) else path); self.profile.save()
            except Exception as exc: self.error(str(exc))
    def profile_dialog(self):
        choices = self.manager.names() + ["<CREATE>", "<COPY CURRENT>", "<DELETE>"]
        name, ok = QInputDialog.getItem(self, "Product profile", "Load or manage product", choices, editable=False)
        if not ok: return
        try:
            if name in ("<CREATE>", "<COPY CURRENT>"):
                new, yes = QInputDialog.getText(self, "Product profile", "New profile name")
                if not yes: return
                self.profile = self.manager.create(new) if name == "<CREATE>" else self.manager.copy(self.profile.name, new)
            elif name == "<DELETE>":
                target, yes = QInputDialog.getItem(self, "Delete product", "Profile", [n for n in self.manager.names() if n != "default"], editable=False)
                if yes: self.manager.delete(target)
                return
            else: self.profile = self.manager.load(name)
            self.pipeline = InspectionPipeline(self.profile); self.statusBar().showMessage(f"Product: {self.profile.name}")
        except Exception as exc: self.error(str(exc))
    def calibration(self):
        value, ok = QInputDialog.getDouble(self, "Calibration", "Pixels per millimetre", self.profile.section("calibration").get("pixels_per_mm") or 1, .001, 100000, 4)
        if ok: self.profile.section("calibration")["pixels_per_mm"] = value; self.profile.save()
    def roi_editor(self):
        if self.last_roi is None: return self.error("Run an inspection to detect and display the ROI before editing its mask")
        dialog = RoiEditorDialog(self.last_roi, self.profile.section("roi"), self)
        if dialog.exec():
            self.profile.data["roi"] = {"inspection_polygon": dialog.canvas.inspection, "excluded_polygons": dialog.canvas.exclusions}; self.profile.save()
    def view_logs(self):
        destination, _ = QFileDialog.getSaveFileName(self, "Export inspection log", "inspections.csv", "CSV (*.csv)")
        if destination: self.database.export_csv(Path(destination))
    def import_images(self, subset):
        folder = QFileDialog.getExistingDirectory(self, f"Import {subset} images")
        if not folder: return
        destination = resolve_path(self.profile.data.get("dataset_path", f"dataset/{self.profile.name}")) / subset; destination.mkdir(parents=True, exist_ok=True)
        count = 0
        for source in Path(folder).iterdir():
            if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}: shutil.copy2(source, destination/source.name); count += 1
        QMessageBox.information(self, "Dataset import", f"Imported {count} images")
    def settings(self):
        QMessageBox.information(self, "Settings", "Portable camera and application settings are under config/. Product-specific inspection settings are saved in config/products/.")
    def save_debug(self):
        result = getattr(self, "last_result", None)
        if not result or not result.debug_images: return self.error("No debug output is available")
        folder = QFileDialog.getExistingDirectory(self, "Save debug output")
        if folder:
            for name, value in result.debug_images.items():
                data = value if value.dtype == np.uint8 else np.clip(value * (255 if value.max() <= 1 else 1), 0, 255).astype(np.uint8)
                cv2.imwrite(str(Path(folder)/f"{name}.png"), data)
    def _developer(self, enabled): self.developer = enabled; self.debug.setVisible(enabled)
    def error(self, message): self.verdict.setText("SYSTEM ERROR"); self.verdict.setStyleSheet("color:#ff4f57"); QMessageBox.critical(self, "Paint inspection", message)
    def closeEvent(self, event): self.stop_camera(); event.accept()


def run_gui() -> int:
    app = QApplication.instance() or QApplication([]); app.setApplicationName("Paint Inspection System")
    window = MainWindow(); window.show(); return app.exec()
