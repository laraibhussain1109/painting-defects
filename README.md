# Hybrid AI Paint Defect Inspection System

A production-oriented Python/PyQt6 prototype for high-resolution inspection of painted industrial components. A customer-supplied **YOLO model locates the component only**; quality is decided by corroborated anomaly, CIELAB colour, texture, glare, ROI-mask, and physical-size evidence. Missing models, missing ROIs, bad frames, and failed alignment never produce PASS.

## Architecture

```text
camera/image -> YOLO component ROI -> polygon/exclusion mask -> ORB/ECC alignment
 -> conservative illumination normalization -> overlapping tiled anomaly inference
 -> CIEDE2000 colour + multi-scale texture + glare mask
 -> morphology/components/size filtering -> evidence fusion -> PASS/FAIL/UNCERTAIN
```

The GUI exposes camera and image/folder modes, ROI overlays, graphical polygon masks, asynchronous inspection/training, product profiles, calibration, logs, and a developer panel. Production operators see red final contours—not raw heatmaps. `BaseAnomalyDetector` isolates backend churn. Dinomaly is the configured default; the included statistical multi-scale feature implementation is a fully working offline fallback and model format. EfficientAD and PatchCore adapters share the contract. Optional Anomalib integration can therefore evolve without coupling the application to a particular release. Supervised SegFormer and YOLO-Seg boundaries are prepared separately.

> **Model qualification:** This prototype must be validated with the real camera, optics, paint, fixtures, defects, and acceptance standard before production decisions. The fallback model is useful for commissioning but is not a substitute for validating a trained Dinomaly deployment.

## Install (Python 3.11+)

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the CUDA build of PyTorch recommended by the official [PyTorch selector](https://pytorch.org/get-started/locally/) first. For example, use the selector's current Windows/Pip/CUDA command rather than copying an old CUDA wheel command. Then:

```bash
pip install -r requirements.txt
```

The supported surfaces used by this repository are Ultralytics `YOLO(...).predict`, PyTorch inference, and Transformers/timm feature ecosystems. Anomalib is intentionally optional because its Engine/model/export APIs and Dinomaly availability differ by release; install and qualify a release separately (`pip install anomalib`) before replacing the adapter implementation. Failure to import an optional backend does not break classical inspection or the GUI.

## Quick start

1. Place the ROI detector at `models/yolo/best.pt`, or use **SELECT YOLO MODEL**.
2. Put normal images in `dataset/default/good/`. Optional validation sets are `validation_good/` and `validation_bad/`.
3. Run `python main.py`.
4. Select a product, then click **TRAIN ANOMALY MODEL**. Training runs on a `QThread`, saves `models/anomaly/default_v1.npz` plus JSON metadata/report, calibrates a threshold, updates the profile, and loads the result.
5. Use **OPEN IMAGE**, **OPEN FOLDER**, or **START CAMERA**, then **INSPECT**.

The camera starts only on request. Without both YOLO ROI and anomaly models the result is **INSPECTION UNAVAILABLE**, never PASS.

## Product configuration

Profiles live under `config/products/*.json`; `default.json` documents every setting. Copy it to create another product and select it with **PRODUCT PROFILE**. Paths may be project-relative or absolute. Profiles contain:

* YOLO path, confidence, IoU, expected class, input size, and ROI padding;
* inspection polygon plus any number of exclusion polygons;
* alignment/preprocessing switches, reference images, model/backend/version;
* tile size/overlap/batch size and colour/texture/glare thresholds;
* minor/critical pixel areas, fusion weights/limits, dataset, and pixels/mm.

After an ROI has been detected, **ROI EDITOR** supports point-drawn inspection and exclusion polygons. Finish each polygon explicitly and save it. Reference image paths should be normal examples captured with the same optics. References drive alignment, robust median colour statistics, and texture baselines; the system never performs naïve RGB subtraction.

For calibration, enter pixels/mm using **CALIBRATION**. Defect width, height and area are then computed in mm/mm². Without calibration, pixel measurements remain authoritative.

## Training and thresholds

Only GOOD images are required. The default model learns per-location multi-scale LAB, gradient and local-variance distributions. Optional good validation scores yield mean, standard deviation, percentiles, maximum, and a conservative initial threshold. When bad validation images exist, the threshold is selected by F1 and the report adds TP/TN/FP/FN, precision, recall, and F1. Always review false positives and false negatives against the plant's cost and safety requirements; edit the stored profile threshold to apply a qualified manual override.

Colour uses illumination-compensated CIELAB and CIEDE2000. Texture combines Scharr energy and local variance. Glare uses high-value/low-saturation pixels and local edges. Its default `REDUCE_ANOMALY_WEIGHT` mode attenuates rather than silently discards evidence. Fusion requires defect area and multi-branch agreement for FAIL; ambiguous evidence stays UNCERTAIN.

## High-resolution tiling

ROIs are never globally crushed to 224×224. The tiler covers the ROI at the configured 512/640-style size, overlaps neighboring tiles, batches model inference, resizes each output only to its source tile, and reconstructs the original resolution with a tapered blend window. Small ROIs use one padded tile. Adjust tile size, overlap and batch size in the profile based on defect scale and GPU memory.

## Camera and image operation

`config/camera.yaml` configures OpenCV index, resolution, FPS, and optional exposure. `BaseCamera` is the extension point for Basler, FLIR, DALSA, Hikrobot, or other SDKs. Camera acquisition uses a GUI timer and inspection uses a worker thread. Image mode includes open file/folder and previous/next navigation. Headless batch use:

```bash
python tools/batch_inspect.py path/to/images --profile default
```

AUTO mode is available in the GUI. For PLC-triggered deployment, connect the provided stability state machine and `BasePLC`; `ConsolePLC` is the hardware-free implementation.

## Results, developer mode, and data

Every completed attempt is serialized to SQLite at `inspection_results/inspections.db`. **VIEW LOGS** exports CSV. Developer mode exposes branch/map range diagnostics; raw maps remain hidden in normal operation. The pipeline retains anomaly, colour, texture, glare, and ROI mask images for qualified debug-output integrations. Runtime logs are written to `logs/paint_inspection.log`.

For deployments that retain result images, organize them below `inspection_results/YYYY-MM-DD/{pass,fail,uncertain}` and apply the site's retention/privacy policy. The directories are gitignored.

## Layout

* `paint_inspection/gui.py` — industrial GUI and non-blocking workers.
* `paint_inspection/pipeline.py` — fail-safe orchestration.
* `paint_inspection/vision/` — ROI, alignment, tiling, anomaly abstraction, classical branches, fusion and postprocessing.
* `paint_inspection/{camera,database,training,integration}.py` — hardware boundary, persistence, calibration/reporting, PLC/state machine.
* `config/` — portable app, camera, and per-product configuration.
* `models/`, `dataset/`, `inspection_results/`, `logs/` — runtime artifacts (large/private contents ignored).
* `tests/` — deterministic non-GUI component tests.

## Troubleshooting

* **CUDA not shown:** run `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`; confirm NVIDIA driver and PyTorch wheel CUDA versions. Set `cpu_only` in app configuration for a deliberate CPU deployment.
* **YOLO unavailable:** verify the selected `.pt` belongs to a compatible Ultralytics release and its expected class matches the profile. A missing file intentionally blocks inspection.
* **No ROI:** lower confidence only after reviewing detections, class selection, optics and exposure. YOLO confidence never contributes to paint quality.
* **Out of memory:** reduce anomaly batch size first, then tile size. Do not downscale the entire high-resolution ROI.
* **Alignment unavailable:** add representative masters, improve part features/lighting, or disable alignment only for a mechanically repeatable fixture.
* **Too many glare indications:** qualify value/saturation thresholds using glossy GOOD parts; do not merely raise all defect thresholds.
* **Headless Linux:** tests and batch mode work without displaying Qt. The GUI requires an X/Wayland display or Windows desktop.

## Testing

```bash
pytest -q
python -m compileall -q main.py paint_inspection tools
```

Tests cover ROI extraction, polygon/exclusion masks, seamless reconstruction, Delta-E behavior, connected components and size filters, fusion safety rules, profiles, threshold calibration, and result serialization.
