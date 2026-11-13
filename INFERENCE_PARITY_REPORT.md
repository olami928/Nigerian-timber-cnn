# Inference parity testing report

**Date:** 2026-11-13  
**Status:** NOT EXECUTED in this environment (tooling/dependencies unavailable)

## Required comparison (≥20 existing test photographs)

Compare across: (1) original Python inference, (2) existing web app inference, (3) Android APK inference.

For each image record:
- Predicted class
- Per-patch predictions (48, 6×8 raster order)
- Aggregated prediction
- Patch vote counts
- Patch agreement (% and `k/48`)

Investigate and document any differences (do not hide them).

## Available references in repo

- Python reference: `src/infer.py`, `src/aggregate.py` (majority vote, `counts + 1e-3*meanProb` tie-break, same 6×8/112/ImageNet norm/sharpness handling as exported config).
- Web/Android inference core: `app/public/inference-core.mjs`, `app/public/app.js` (same grid/resize/normalization/aggregation logic; ORT Web WASM).
- Existing parity test: `tests/test_infer.py` (`test_predict_matches_notebook_reference_on_20_held_out_images`) — compares predictor output to notebook-reference steps on 20 held-out independent-specimen images (requires torch/cv2/PIL/numpy/pandas, model exports, splits/manifest/mapping and image files).
- Model/config artifacts: `app/model/{model.onnx,class_names.json,preprocess_config.json}` and `artifacts/p2_training/final/shufflenet_v2/`.

## Environment blockers

- Python ML/runtime deps not installed here: `torch`, `cv2`, `numpy`, `pandas`, `PIL`, `onnxruntime` reported missing.
- `patch_ready_dataset/` image files not present in workspace at expected paths for manifest-driven runs (sampled paths showed no files).
- Android SDK/JDK/adb/device unavailable; no APK built — APK inference comparison cannot be run.

## How to run (on a prepared machine)

1. Install Python deps and ensure dataset/model/split artifacts and images are accessible at paths expected by `data/manifest.csv` and `data/splits.csv`.
2. Run: `python -m pytest tests/test_infer.py -q` (or run the parity test) and export per-image results (class, 48 patch preds, votes, agreement) for ≥20 images.
3. Capture identical outputs from web build and installed APK (same images, same config/model).
4. Produce a table/diff of all compared fields and explain any mismatch (floating-point/WASM/runtime/resize/EXIF/orientation differences).

**Result:** No parity numbers claimed. Populate this report with actual results after execution.
