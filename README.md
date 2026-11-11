# Nigerian Timber ID

A research prototype for identifying 29 Nigerian timber species from photographs of **transverse (cut end-grain)** wood surfaces, built as a Master's thesis project at the University of Ibadan.

- **Primary model**: ShuffleNetV2 (~1.5M parameters, ~6 MB ONNX export)
- **Comparison model**: ResNet18 (~11.3M parameters)
- **Inference method**: Patch-based (TDLI-PIV style) — each image is split into a 6×8 grid (48 patches), each patch is classified, and the image-level result is a majority vote with mean-probability tie-breaking
- **Website**: Static browser-inference site deployed on Vercel; model runs locally in the visitor's browser via ONNX Runtime Web (WASM). Images are never uploaded to a server.
- **Wood type & documented uses**: Looked up from `species_enduse.json` — never predicted from the image

---

## ⚠️ Important Limitations (read before deploying or using)

**Independent-specimen test accuracy is low:**

| Model | Accuracy | Test set |
|-------|----------|----------|
| ShuffleNetV2 | 12.1% | 611 images, 14 species |
| ResNet18 | 11.3% | 611 images, 14 species |

- Fifteen single-specimen species had **no independent-specimen test result**
- Most confused pair: Ita vs Oporoporo
- No confidence or sharpness threshold is approved for production use
- This is a **research prototype**, not a validated timber identification service

**Do not use predictions for purchasing, legal, import/export, or structural-grading decisions.**

The website prominently displays this limitation on the Identify screen.

---

## Project Structure

```
/home/dimeji/Nigerian-timber-cnn/
├── README.md                          # This file
├── vercel.json                        # Vercel deployment config
├── requirements.txt                   # Python dependencies (training/evaluation)
├── AGENT_BRIEF_timber_id.md           # Full project brief & rules
├── AGENTS.md                          # Agent instructions summary
├── PROJECT_RECORD.md                  # Decisions, process, results record
├── progress.md                        # Phase progress handoff
├── CHAPTER_4_INPUTS.md                # Thesis Chapter 4 tables & figures index
├── ARCHITECTURE_OPTIMIZATION_REPORT.md # Model optimization diagnosis
├── configs/
│   ├── species.csv                    # 29 species: id, slug, local name, botanical, wood type
│   ├── species_enduse.json            # Wood type + documented uses lookup (29 species)
│   ├── default.yaml                   # Training configuration (regularized recipe)
│   ├── legacy.yaml                    # Original notebook-matched training config
│   ├── cpu.yaml                       # CPU training config
│   ├── CPU_TRAINING.md                # CPU training instructions
│   └── english_names_list.md          # Template for English names (to be filled by owner)
├── data/
│   ├── manifest.csv                   # Image manifest: filenames, species, specimen codes
│   └── splits.csv                     # Specimen-grouped 3-fold train/val/test splits
├── src/                               # Python training/evaluation/inference source
│   ├── model.py                       # ShuffleNetV2 + ResNet18 model definitions
│   ├── train.py                       # Training pipeline
│   ├── data.py                        # Dataset loading & patch extraction
│   ├── split.py                       # Specimen-grouped split creation
│   ├── evaluate.py                    # Evaluation & metrics
│   ├── infer.py                       # Shared inference predictor
│   └── aggregate.py                   # Patch aggregation (majority vote)
├── tests/                             # Python unit tests
├── app/                               # Website (static, Vercel-deployed)
│   ├── package.json                   # npm dependencies (esbuild, onnxruntime-web)
│   ├── build.mjs                      # Build script: copies model, generates species.json, bundles JS
│   ├── build-species-data.mjs         # Generates public/species.json from configs + manifest
│   ├── model/
│   │   ├── model.onnx                 # ShuffleNetV2 ONNX export (~6 MB)
│   │   ├── class_names.json           # 29 class slugs in canonical order
│   │   └── preprocess_config.json     # Preprocessing configuration
│   ├── public/                        # Deployed static site (built output)
│   │   ├── index.html                 # Identify / Species / Photograph / About screens
│   │   ├── app.js                     # Application logic (model loading, inference, UI)
│   │   ├── inference-core.mjs         # Sharpness, resize, tensor preprocessing
│   │   ├── styles.css                 # Stylesheet
│   │   ├── species.json               # Generated species data (built at build time)
│   │   ├── model.onnx                 # Copied from app/model/ at build time
│   │   ├── preprocess_config.json     # Copied from app/model/ at build time
│   │   ├── class_names.json           # Copied from app/model/ at build time
│   │   ├── app.bundle.js              # Bundled app.js (built at build time)
│   │   └── ort/                       # ONNX Runtime Web WASM assets
├── notebooks/                         # Colab training notebooks
├── artifacts/                         # Training & evaluation outputs (gitignored)
│   ├── model.pt                       # ShuffleNetV2 PyTorch checkpoint (~6 MB)
│   ├── resnet18_fold1.pt              # ResNet18 checkpoint (~45 MB)
│   ├── leak_audit.csv                 # Specimen leak audit (PASSED for all 3 folds)
│   └── p2_training/, p3_evaluation/   # Training & evaluation outputs
├── patch_ready_dataset_raw_backup/    # Original image dataset (gitignored — 900 images)
└── .gitignore                         # Git ignore rules
```

---

## How the Model Works

### Architecture
- **ShuffleNetV2** (primary) and **ResNet18** (comparison), both with 29 output classes (species only)
- No multi-task head, no uncertainty-weighted loss — cross-entropy with class weights
- Backbone uses adaptive average pooling; classifier head includes LeakyReLU(0.1) and Dropout(0.5)

### Inference Pipeline (patch-based majority vote)
1. Input image is split into a **6×8 grid = 48 patches**
2. Each patch is resized to **112×112** with ImageNet normalization
   - mean: `[0.485, 0.456, 0.406]`
   - std: `[0.229, 0.224, 0.225]`
3. Each patch is classified by the model → softmax probabilities over 29 species
4. Image-level prediction = **majority vote** across the 48 patches
5. Ties broken by **mean probability**
6. Confidence = **patch agreement** (winning votes / 48), not a probability guarantee

### Dataset
- 900 images, 29 species, 45 specimens
- All hardwoods; transverse/end-grain surface only
- Splits are **specimen-grouped 3-fold cross-validation** (no specimen appears in more than one split role)
- Leak audit: PASSED for all 3 folds

---

## Website (Phase 5)

### Features
- **Identify**: Upload or photograph a cut-end wood surface; get species prediction with confidence, top-3 alternatives, documented uses, and scope note
- **Species**: Browse all 29 covered species with local names, botanical names, wood type, and documented uses
- **Photograph**: Guide on how to photograph wood end-grain for best results
- **About**: Model limitations, accuracy disclosure, privacy information, appropriate use

### Privacy
- Images are decoded and processed **locally in the browser**
- Images are **not uploaded or stored** by the site
- Only anonymous counts (time, status, confidence, model version) are logged — never images

### Mobile Behavior
- The file input shows the **system file picker** — users can choose to take a photo with the camera OR select an existing image from their gallery
- Sharpness threshold is set to 100 (feasible for phone cameras)

---

## Deployment on Vercel

### Prerequisites
- GitHub account
- Vercel account

### Steps

1. **Push to GitHub** (already done — repo is at `olami928/Nigerian-timber-cnn`):
   ```bash
   git push -u origin main
   ```

2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com) and log in
   - Click **"Add New Project"** → **"Import Git Repository"**
   - Select: `olami928/Nigerian-timber-cnn`
   - **Framework preset**: Select **Static Site** (or **Other** / **None** — do NOT select FastAPI)
   - Root Directory: `./` (default)
   - Click **Deploy**

3. **Vercel will automatically use** the settings in `vercel.json`:
   - Install Command: `npm --prefix app ci`
   - Build Command: `npm --prefix app run build`
   - Output Directory: `app/public`

4. **Redeploy after changes**:
   - Push changes to GitHub main branch
   - Vercel auto-deploys on push, or click **Redeploy** in the dashboard

### What Gets Deployed
The build script (`app/build.mjs`) does the following:
1. Reads `configs/species.csv` and `configs/species_enduse.json` to generate `public/species.json` (validates all 29 species match)
2. Copies model files from `app/model/` to `public/`: `model.onnx`, `class_names.json`, `preprocess_config.json`
3. Copies ONNX Runtime Web WASM assets to `public/ort/`
4. Bundles `public/app.js` → `public/app.bundle.js` with esbuild
5. Validates class order matches between model config and species list

### Build Output (gitignored, regenerated on deploy)
- `app/public/app.bundle.js`
- `app/public/class_names.json`
- `app/public/model.onnx`
- `app/public/preprocess_config.json`
- `app/public/species.json`
- `app/public/ort/`

---

## Local Development

### Build the website locally
```bash
# From the repository root
npm --prefix app ci
npm --prefix app run build
```

Serve `app/public/` with any static HTTP server:
```bash
# Example using Python
cd app/public && python -m http.server 8000
```

### Run Python tests
```bash
# From the repository root
python -m pytest tests/ -v
# or
python -m unittest discover tests/ -v
```

### Training (requires GPU — Colab recommended)
The local machine has CPU-only PyTorch. Full training runs on Colab:
- `notebooks/colab_training.ipynb` — resume-safe training workflow
- `configs/default.yaml` — regularized training recipe
- `configs/legacy.yaml` — original notebook-matched recipe
```

---

## Configuration Files

### `configs/species_enduse.json`
Contains wood type and documented uses for all 29 species. The `english_name` field is currently `null` for all species — fill this from your photographed names list using `configs/english_names_list.md` as a template.

**Rules** (from the brief):
- Uses come ONLY from `species_enduse.json` — never predicted from the image
- If `uses` is empty, hide the Documented uses section
- Never invent a use or source
- English names are blank until filled from the user's names list

### `app/public/preprocess_config.json`
Preprocessing configuration used by both the build script and the browser inference:
- `min_sharpness`: 100 (Laplacian variance threshold; below this returns a retake message)
- `min_confidence_pct`: null (no retake threshold approved)
- Grid: 6×8 = 48 patches, 112×112 resize, ImageNet normalization

### `vercel.json`
Vercel deployment configuration — tells Vercel to build the static npm site, not auto-detect FastAPI.

---

## Testing

### Website tests (manual)
- [ ] Normal image upload and prediction
- [ ] Blurry image → retake message (sharpness < 100)
- [ ] Wrong file type (non-image) → error
- [ ] Huge file (> 8 MB) → error
- [ ] Tiny image → error or graceful handling
- [ ] Mobile: camera option and gallery option both available
- [ ] Mobile: sharpness threshold works with phone camera photos
- [ ] Parity test: browser predictions match Python notebook on 20 held-out images

### Python unit tests
All tests pass:
```
tests/test_infer.py    - Inference parity (3 tests)
tests/test_data.py     - Data loading
tests/test_model.py    - Model architecture
tests/test_split.py    - Split creation & leak audit (6 tests)
tests/test_evaluate.py - Evaluation metrics (4 tests)
tests/test_train.py    - Training pipeline (13 tests)
```

---

## Thesis Context

**Thesis**: *Evaluation of Machine Learning Adaptability as an Image-Based Tool for Identifying Nigerian Timber Species and Probable End-Uses*

**Owner**: Dimeji (Alabi Oladimeji Taofeek), B.Sc. Wood Products Engineering, University of Ibadan

**Species names follow**: Keay (1989), *Trees of Nigeria*

**Dataset source**: Nigerian timber specimens from Ibadan wood markets (transverse/end-grain only)

**Prohibited**: MobileNetV2, GOIMAI-Phase-I dataset for training, SMOTE, MixUp, multi-task heads, "43-58% trader accuracy" claims, FRIN citations

---

## Contributing

This is a thesis research project. The primary model architecture, species list, and evaluation methodology are locked per the project brief. Changes to these require owner approval.

### Before modifying
1. Read `AGENT_BRIEF_timber_id.md` (full project rules)
2. Read `PROJECT_RECORD.md` (decisions and results)
3. Read `AGENTS.md` (agent instructions)
4. Never modify or delete image data — keep a backup before changing anything
5. Never invent citations, species names, uses, or results

---

## License

Research prototype for academic thesis work. Not for commercial use.