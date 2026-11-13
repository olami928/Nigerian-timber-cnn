# Timber ID — modifications and remaining limitations

**Date:** 2026-11-13

## Modifications made

### Web UI/UX (`app/public/`)
- Reworked layout and styling (`index.html`, `styles.css`, `app.js`/`app.bundle.js`): warm ivory/cream surfaces, forest green/wood/charcoal palette, restrained scientific/educational presentation, mobile-first responsive layout.
- Homepage/Identify: branding, intro, prominent camera + gallery capture, preview with retake/replace, drag-drop, transverse end-grain guidance, suitable vs unsuitable photo guidance, 29-species scope, process explanation (6×8/48 patches, on-device ShuffleNetV2), and research-prototype disclaimer.
- Results UI: uploaded photo, predicted species (local/botanical/English when available), top-3 matches, patch vote/agreement %, mean probability (secondary), limited-reference notice, documented end-uses from local data only, and limitations; states patch agreement is not calibrated probability.
- Species directory: searchable list with local/botanical/English names, wood type, training specimen count, documented uses (or none recorded).
- About/Guide: methodology, dataset/scope, preprocessing/aggregation summary, limitations, privacy/appropriate-use guidance.
- Removed "Skip to content" link. Converted asset/model/runtime fetches to relative paths for packaged/local serving.

### Build & packaging (`app/`)
- `build.mjs`/assets: bundles and copies local ONNX model, class names, preprocess config, species data, and ONNX Runtime Web WASM (`ort/`) into `public/` (no runtime CDN dependency in app source).
- Added Capacitor 6 project (`@capacitor/core/cli/android`, `@capacitor/camera`), `capacitor.config.json` (`com.timberid.nigeriantimberid`, `Timber ID`, `webDir: public`), and generated Android project in `app/android/`.
- Synced web assets/model/WASM into Android assets.

### Android project (`app/android/`)
- Manifest updated for local/offline use: removed `INTERNET`, added camera/media-read permissions and non-required camera feature, FileProvider retained for captures, `usesCleartextTraffic=false`.
- Launcher foreground/background and splash resources updated to Timber ID branding/colors; app strings set.

### Documentation/reports
- Updated `app/README.md` with web build, Capacitor sync, debug build, and signed-release instructions.
- Added `INFERENCE_PARITY_REPORT.md`, `OFFLINE_TESTING_REPORT.md`, `MODIFICATIONS_AND_LIMITATIONS.md` (this file).

## Not changed (preserved)

- Model weights/architecture: ShuffleNetV2 ONNX export unchanged; ResNet18 not used as app model; MobileNetV2 not introduced.
- Class mapping/order: `class_names.json` (29) preserved.
- Inference pipeline logic: 6×8 grid (48 patches), 112×112 resize, ImageNet mean/std normalization, per-patch classification, majority vote with mean-probability tie-break, patch agreement/count reporting, blur/threshold config semantics, species lookup from local documented data.
- No retraining; no fabricated species/uses/results.

## Remaining limitations / blockers

- **No APK built or run:** environment lacks JDK 17, Android SDK/cmdline-tools/build-tools/platform 34, `adb`, and device/emulator (`./gradlew assembleDebug` fails with `JAVA_HOME`/`java` missing). Signed release not produced.
- **Offline/airplane-mode tests not performed:** cannot be claimed until executed on a real device with the built APK.
- **Inference parity (≥20 images) not executed:** Python ML deps (torch/cv2/numpy/pandas/PIL/onnxruntime) and dataset image availability for scripted comparison were not present; web-vs-Python-vs-APK parity unverified here.
- **ORT/WASM in Android WebView:** single-threaded WASM assets bundled and paths made relative, but on-device WebView compatibility/performance must be validated.
- **Camera/gallery on Android:** Web file inputs and Capacitor Camera plugin are present, but native capture flow, URI decoding, EXIF/orientation, large-image memory behavior, and permissions UX require device testing and possible native integration refinements.
- **UI inspiration images:** user-provided `image.png` screenshots could not be read in this environment; redesign follows written requirements but may not match intended visual references.
- **Model performance:** research prototype with low reported independent-specimen accuracy; results must not be treated as validated identification.
