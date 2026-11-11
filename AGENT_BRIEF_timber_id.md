# AGENT BRIEF: Timber Species Identification (Model + Website)

Version 2.0, 10 November 2026
Owner: Dimeji (Alabi Oladimeji Taofeek), B.Sc. Wood Products Engineering, University of Ibadan
Thesis: Evaluation of Machine Learning Adaptability as an Image-Based Tool for Identifying Nigerian Timber Species and Probable End-Uses

Read this whole document, then read the existing Colab notebook, then report back (Phase 0). Time is short, so this brief contains only what is needed now. Extra ideas are parked in Section 16 ("Later") and must NOT be built unless I ask.

---

## 0. How to work with me

1. Work in phases (Section 6). **Stop at the end of every phase** and report. Do not start the next one until I say "go".
2. Stop and ask at every **DECISION GATE**. Never fill a gap with an assumption or an invented value.
3. Never invent a citation, statistic, species name, use, or result. If something is missing, say so.
4. Never delete or overwrite my images. Keep a backup copy before changing anything.
5. Everything must be reproducible: fixed seeds, saved split file, saved config, saved class order.
6. Explain results in plain language first, technical detail second. I must be able to explain every choice at my viva.
7. If this brief conflicts with the existing notebook, report the conflict. Do not silently pick one.

### Inputs you should have

- The existing Colab notebook (main reference for the pipeline)
- My image folders (one folder per specimen)
- This brief and `species_enduse.json`
- My names list (Yoruba, English, botanical names), if I have uploaded it

If any is missing, say so in Phase 0.

---

## 1. Project in one paragraph

Build a CNN that identifies 29 timber species traded in Ibadan wood markets from a macroscopic photo of the **transverse (end-grain) surface**, and a website that serves it. The model predicts **species only**. Hardwood/softwood and documented uses are looked up from the predicted species. Each prediction also returns a confidence percentage. Primary architecture: ShuffleNetV2. Comparison architecture: ResNet18. Method: patch-based (TDLI-PIV style). Each image is cut into a 6x8 grid (48 patches), each patch is classified, and the image result is a majority vote.

---

## 2. Locked decisions (do not change without asking)

1. **ShuffleNetV2 is primary, ResNet18 is the comparison.** MobileNetV2 must never appear anywhere.
2. **One model per architecture, species output only.** No second trained head, no multi-task learning, no uncertainty-weighted loss. Loss is cross-entropy.
3. **Wood type and uses come from a lookup table** keyed by the predicted species (`species_enduse.json`).
4. **Pipeline follows the existing notebook:** TDLI-PIV patches (Herrera-Poyatos et al., 2024, arXiv:2406.11772), 6x8 grid, 48 patches, majority vote.
5. **Dataset is transverse-only.** Never use the word "longitudinal" as a surface name.
6. **Splits are by specimen, not by image** (with the documented exception for single-specimen species, Section 8). Augmentation is for the training split only.
7. **Do not use the GOIMAI-Phase-I dataset for training.**
8. **No SMOTE and no MixUp.** Use class weights for imbalance.
9. **Species names follow Keay (1989), *Trees of Nigeria*.**
10. **Banned claims:** never state or cite a "43-58% trader accuracy" figure. Never cite FRIN as a verification body.
11. **Quality over quantity.** No page target. Do not pad.

---

## 3. Open decisions (do not decide these for me)

| # | Question | Status |
|---|---|---|
| D2 | Does the notebook use global max pooling or average pooling at the end of ShuffleNetV2? (LeakyReLU is confirmed.) | Read the notebook and report. |
| D3 | Split protocol in Section 8 (specimen-grouped 3-fold cross-validation) replaces a single 70:15:15 split. | Proposed. Needs my supervisor's approval. Build it. If rejected, fall back to fold 1 only. |
| D8 | Hosting for the website (Section 13). | Ask me. |
| D9 | Submission deadline. | Ask me. |
| D10 | Botanical name of species #12 (Epin). | Resolved: *Ficus exasperata* Vahl (confirmed by Dimeji from Keay, 1989). |

---

## 4. Species list (29)

Source: my corrected names list (October 2026). Epin replaces Iha. Obobo = *Ficus mucuso* is confirmed by me.

| # | Local / trade name | Botanical name |
|---|---|---|
| 1 | Afara | *Terminalia superba* |
| 2 | Agboin | *Piptadeniastrum africanum* |
| 3 | Apa | *Afzelia africana* |
| 4 | Araba / Ogungun | *Ceiba pentandra* |
| 5 | Arere | *Triplochiton scleroxylon* |
| 6 | Awun | *Alstonia boonei* |
| 7 | Ayin | *Anogeissus leiocarpa* |
| 8 | Ayunre | *Albizia zygia* |
| 9 | Emido | *Manilkara obovata* |
| 10 | Epu | *Ricinodendron heudelotii* |
| 11 | Gmelina | *Gmelina arborea* |
| 12 | Epin | *Ficus exasperata* |
| 13 | Ire | *Funtumia elastica* |
| 14 | Iroko | *Milicia excelsa* |
| 15 | Irugba | *Parkia biglobosa* |
| 16 | Ita | *Celtis zenkeri* |
| 17 | Iya | *Daniellia oliveri* |
| 18 | Iyeye | *Spondias mombin* |
| 19 | Laoro | *Boscia angustifolia* |
| 20 | Mahogany | *Khaya senegalensis* |
| 21 | Mango | *Mangifera indica* |
| 22 | Obi | *Cola nitida* |
| 23 | Obobo | *Ficus mucuso* |
| 24 | Okobo | *Mitragyna inermis* |
| 25 | Oporoporo | *Pterygota macrocarpa* |
| 26 | Oro | *Antiaris toxicaria* |
| 27 | Osun | *Pterocarpus osun* |
| 28 | Parapuru | *Klainedoxa* spp. (genus level) |
| 29 | Teak | *Tectona grandis* |

Rules:
- If my image folder for species #12 is still named `iha`, report it and ask before renaming.
- English names are blank in the JSON. Fill them only from my names list, never from memory.
- When I give you my names list, compare it with this table and report any mismatch. Do not edit automatically.

---

## 5. Repository layout

```
timber_project/
  README.md                    exact commands to reproduce everything
  requirements.txt             pinned versions
  data/                        my images (already prepared, see 7.2)
  data/manifest.csv            one row per image
  data/splits.csv              generated once, never edited by hand
  configs/species.csv          the 29 species
  configs/species_enduse.json  lookup table (provided)
  configs/default.yaml         seeds, patch size, thresholds, paths
  src/                         data.py split.py model.py train.py evaluate.py infer.py
  notebooks/                   original Colab notebook (unchanged copy)
  artifacts/                   models, reports, figures
  app/                         API + front end
  tests/                       leak audit, parity test, API tests
```

---

## 6. Phases and stop gates

- **P0 Read and report.** Summarise the notebook (framework, patch size P, normalisation, pooling, number of images). List conflicts with this brief and any questions. **STOP.**
- **P1 Data.** Manifest, checks, region groups, split plan, leak audit (Sections 7 and 8). **STOP.**
- **P2 Train.** ShuffleNetV2 and ResNet18 per fold, then final models (Section 9). **STOP.**
- **P3 Evaluate.** Metrics, confusion matrix, confidence threshold (Sections 10 and 11). **STOP.**
- **P4 Inference module.** One shared `predict()` with a parity test (Section 13.2). **STOP.**
- **P5 Website.** API and front end (Section 13). **STOP.**
- **P6 Deploy and test.** Hosting, phone test, local backup demo. **STOP.**

---

## 7. Data (P1)

### 7.1 Naming scheme

Each image folder is **one specimen**. Folders are named by the wood plus a number: `ayunre` = specimen 1, `ayunre2` = specimen 2, `ayunre3` = specimen 3. Maximum 3 specimens per species. Some species have 2, some only 1. File names inside folders (for example "image 1") carry no information.

### 7.2 Images are already grid-ready (do not redo)

I already edited all images with another agent so each one divides into the 6x8 grid. **You do not need to process them. Only confirm it:**

1. For every image, check width is divisible by 8 and height is divisible by 6 (8 columns x 6 rows).
2. Report the patch size each image gives (width/8 by height/6) and whether patches are square.
3. Report the distribution of image sizes, and how many images fail the check.
4. Report whether this patch size matches the patch size P in the notebook.
5. **Do not resize, crop or modify any image.** If some fail, list them and ask me.
6. Ask me whether untouched originals exist somewhere else. Recommend a backup of the current folder before anything else.

Normalisation: find out what the notebook does (divide by 255 only, or also ImageNet mean/std). Use exactly the same in training and in the website, and tell me so my Chapter Three matches.

### 7.3 Build the manifest from folder names

```python
import re, hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path("data")          # data/ayunre/image 1.jpg, data/ayunre2/image 1.jpg ...

def parse_folder(name: str):
    m = re.fullmatch(r"\s*([A-Za-z_\- ]+?)\s*(\d*)\s*", name)
    slug = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
    idx = int(m.group(2)) if m.group(2) else 1          # no number = specimen 1
    return slug, idx

rows = []
for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    slug, idx = parse_folder(folder.name)
    for img in sorted(folder.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(img) as im:
            w, h = im.size
        rows.append(dict(path=str(img), orig_name=img.name, folder=folder.name,
                         species=slug, specimen_idx=idx, specimen_code=f"{slug}_{idx}",
                         width=w, height=h,
                         sha256=hashlib.sha256(img.read_bytes()).hexdigest()))
df = pd.DataFrame(rows)
df.to_csv("data/manifest.csv", index=False)
```

Manifest columns: `path, orig_name, folder, species, specimen_idx, specimen_code, width, height, sha256, group_id, split_role`. Do not add other columns.

### 7.4 Checks (report before going further)

1. Every `species` matches `configs/species.csv`. Flag near-misses and suggest a match. Do not auto-correct.
2. Each species has specimen indexes 1, 2, 3 in order with no gaps and no more than 3.
3. Exact duplicate images (same hash): list them. Remove only with my approval.
4. Unreadable or corrupt images: list them.
5. Print images per specimen and specimens per species.

---

## 8. Splits (P1)

### 8.1 Why not a simple random split

With at most 3 specimens per species, a specimen-level 70:15:15 split is impossible. A random image split would put near-identical photos of the same wood face in both training and test (Same-Specimen-Picture Bias), which inflates accuracy. Use the protocol below. **My supervisor must approve it (D3).**

### 8.2 Protocol: specimen-grouped 3-fold cross-validation

In fold k, **specimen k of every species is the test set.**

| Species has | What happens |
|---|---|
| 3 specimens | Each fold tests one specimen and trains on the other two. |
| 2 specimens | Fold 1 tests specimen 1, fold 2 tests specimen 2, fold 3 has no test specimen for it. |
| 1 specimen | No independent test is possible. Use a within-specimen group split (8.4). Report separately. |

Validation (early stopping only): about 15% of region groups (8.3) from the training specimens.

**Final model for the website:** retrain on all specimens using the epoch count found in the folds. No test number belongs to this final model. All thesis numbers come from the fold models.

### 8.3 Region groups (stops neighbouring photos leaking)

Photos of the same cut face overlap. Cluster each specimen's images and always send a whole group to one side.

```python
import numpy as np, torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from torchvision.models import resnet18, ResNet18_Weights

def embed(paths, device="cuda"):
    w = ResNet18_Weights.IMAGENET1K_V1
    net = resnet18(weights=w); net.fc = torch.nn.Identity(); net.eval().to(device)
    tf, out = w.transforms(), []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            b = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i+64]]).to(device)
            out.append(net(b).cpu())
    X = torch.cat(out).numpy()
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def assign_groups(df, thr=0.25):
    df["group_id"] = ""
    for code, g in df.groupby("specimen_code"):
        if len(g) < 3:
            df.loc[g.index, "group_id"] = f"{code}_g0"; continue
        X = embed(g.path.tolist())
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=thr,
                                      metric="cosine", linkage="average").fit_predict(X)
        df.loc[g.index, "group_id"] = [f"{code}_g{l}" for l in lab]
    return df
```

Show me a contact sheet (one row per cluster) before accepting `thr`. If a specimen collapses into one cluster, report it.

### 8.4 Species with one specimen

Hold out region groups as validation and test inside that single specimen. Report these species in a **separate** table labelled "within-specimen (optimistic)", never in the headline metrics. Tell me how many such species exist. The best fix is a second specimen from a different plank or trader, so say so.

### 8.5 Split planner

```python
import zlib, numpy as np, pandas as pd

def roles_for_fold(df, fold, val_frac=0.15, seed=42):
    role = pd.Series("train", index=df.index)
    evalg = pd.Series("", index=df.index)
    for sp, g in df.groupby("species"):
        rng = np.random.default_rng(seed + zlib.crc32(sp.encode()))
        n = g.specimen_idx.nunique()
        if n >= 2:
            t = g.index[g.specimen_idx == fold]
            role[t] = "test"; evalg[t] = "independent_specimen" if len(t) else ""
            pool = g.drop(t)
        else:
            pool = g
        groups = np.array(sorted(pool.group_id.unique())); rng.shuffle(groups)
        n_val = max(1, round(val_frac * len(groups))) if len(groups) >= 3 else 0
        role[pool.index[pool.group_id.isin(groups[:n_val])]] = "val"
        if n == 1 and len(groups) >= 4:
            test_g = groups[n_val:n_val + max(1, round(val_frac * len(groups)))]
            ti = pool.index[pool.group_id.isin(test_g)]
            role[ti] = "test"; evalg[ti] = "within_specimen"
    return role, evalg
```

- Use `zlib.crc32`, not Python `hash()` (it changes between runs).
- Write `data/splits.csv` once. Training code only reads it. Never re-split silently.
- Patches are cut AFTER the split. All 48 patches of an image share that image's role.

### 8.6 Leak audit (must pass for every fold; save the output)

```python
def audit(df, role, fold):
    d = df.assign(role=role)
    for sp, g in d.groupby("species"):
        if g.specimen_idx.nunique() >= 2:
            tr = set(g[g.role == "train"].specimen_code)
            te = set(g[g.role == "test"].specimen_code)
            assert not tr & te, f"{sp}: specimen in train and test (fold {fold})"
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(d[d.role == a].group_id) & set(d[d.role == b].group_id), f"group leak {a}/{b}"
        assert not set(d[d.role == a].sha256) & set(d[d.role == b].sha256), f"duplicate image {a}/{b}"
```

---

## 9. Model and training (P2)

1. ShuffleNetV2 (ImageNet-pretrained), classification layer for the number of species in `configs/species.csv`. Follow the notebook for the head (report D2).
2. ResNet18 with the same pipeline, folds and settings.
3. Stage 1: freeze the backbone, train the head only. Adam, lr = 0.001.
4. Stage 2: unfreeze all, lower learning rate (1e-4 to 1e-5). Record the value used.
5. Loss: cross-entropy with class weights from training-split counts.
6. Augmentation, training split only: rotation about 10 degrees, flips, small shifts. Record exact parameters.
7. Early stopping on validation loss, patience about 10, restore best weights. Batch size 32 to 64. Dropout about 0.5.
8. Fixed seeds (Python, NumPy, torch). Record library versions. Save training and validation curves per fold.
9. Any difference between the ShuffleNetV2 and ResNet18 settings must be documented.
10. Inference: image to 48 patches to model to softmax per patch to majority vote.
11. Final model: train on all specimens. Save `model.pt`, an ONNX copy, `class_names.json` (fixed class order) and `preprocess_config.json` (patch size, grid, normalisation, thresholds).

---

## 10. Evaluation and thesis outputs (P3)

Report on **independent-specimen test sets only** (pooled across folds), at image level (after voting). Also give patch level.

- Accuracy, precision, recall, F1 (per class and macro), MCC
- Confusion matrix and the most confused species pairs
- ShuffleNetV2 vs ResNet18 in one table, with parameters and model size (MB)
- Within-specimen species in a separate table, labelled optimistic
- Confidence intervals: resample **specimens, not images**

Thesis formatting (University of Ibadan): plain black-and-white tables, greyscale figures at 300 dpi with Times New Roman labels, numbered by chapter (Table 4.1, Figure 4.1). Photographs and website screenshots are "Plates". Citations use "and", not "&". *et al.* is italic.

Also write `CHAPTER_4_INPUTS.md` listing each table and figure and the claim it supports. **I write the thesis text. You do not.**

---

## 11. Confidence percentage and quality filter (P3, P4)

Every prediction returns:

- `confidence_pct`: **patch agreement**, the share of the 48 patches that voted for the winning species (38 of 48 = 79.2%). This is the headline percentage.
- `patch_agreement`: "38/48"
- `mean_probability_pct`: mean softmax probability of the winning class (secondary)
- `top3`: three species with vote and probability percentages
- `limited_reference_data`: true when the predicted species had fewer than 3 training specimens

```python
import numpy as np

def aggregate(patch_logits, class_names, n_specimens):
    z = patch_logits
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    n = len(p)
    counts = np.bincount(p.argmax(1), minlength=p.shape[1])
    score = counts + 1e-3 * p.mean(0)                  # ties broken by mean probability
    top = int(score.argmax()); order = np.argsort(score)[::-1][:3]
    return {
        "species": class_names[top],
        "confidence_pct": round(100 * counts[top] / n, 1),
        "patch_agreement": f"{counts[top]}/{n}",
        "mean_probability_pct": round(100 * float(p.mean(0)[top]), 1),
        "top3": [{"species": class_names[i],
                  "vote_pct": round(100 * counts[i] / n, 1),
                  "prob_pct": round(100 * float(p.mean(0)[i]), 1)} for i in order],
        "limited_reference_data": n_specimens[class_names[top]] < 3,
    }
```

**Retake threshold:** if `confidence_pct` is below a threshold, the app asks for another photo instead of naming a species. Choose the threshold from the **out-of-fold test predictions** (every specimen predicted by a model that never saw it), never from the validation set. Report the threshold, the share of images accepted, and the accuracy on accepted images.

**Blur filter:** reject images that are too blurry before inference.

```python
import cv2
def sharpness(img_bgr):
    return cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
```

Set the sharpness threshold from the sharpness distribution of the training images (for example a low percentile). Report the value.

The app text must say the percentage is **patch agreement, not a guarantee** of being correct.

---

## 12. Species information (wood type and uses)

`species_enduse.json` holds, per species: `local_name`, `botanical`, `english_name`, `wood_type`, `uses`, `sources`, `status`, `notes`.

Rules:

1. **Return and display only valid, documented uses.** The `uses` object contains only categories marked `primary` or `secondary`. Categories where no use was found are simply absent, and they must never be returned or shown. If a species has no valid use, hide the "Documented uses" section completely.
2. Uses are **documented literature uses for the species, not predicted from the photo.** The app must label them that way.
3. **Wood type:** all 29 species are angiosperms, so all are **Hardwood**. Show a "Hardwood" badge. The app cannot recognise softwoods (no conifer was in the training data), so state the 29-species scope clearly.
4. Never invent or add a use. New uses are added only from sources I supply.
5. Species #12 (Epin): uses are not yet researched, so nothing is shown for it.
6. Source details (authors, years, page numbers) in `sources` are partly incomplete. Do not put them in a reference list. I will complete them.

---

## 13. Website and user interface (P4 to P6)

### 13.1 Architecture

- One shared `predict()` function. Everything calls it.
- **FastAPI** service with a light static front end. If time is short, build **Gradio** first with the same `predict()`. Ask me which (D8).
- Free CPU hosting (for example Hugging Face Spaces with Docker). Pin Python and CPU-only PyTorch or ONNX Runtime.
- **Backup for my defence:** it must also run on my laptop with one command, and I need a recorded screen video of a successful demo, in case hosting fails on the day.

### 13.2 Shared inference module

```python
def predict(image_bgr):
    if sharpness(image_bgr) < CFG["min_sharpness"]:
        return {"status": "retake", "reason": "blurry"}
    patches = make_patches(image_bgr, CFG)        # same preprocessing as training
    logits = model_logits(patches)                # [48, C]
    out = aggregate(logits, CLASS_NAMES, N_SPECIMENS)
    if out["confidence_pct"] < CFG["min_confidence_pct"]:
        return {"status": "retake", "reason": "low_confidence", "top3": out["top3"]}
    out.update(status="ok", model_version=CFG["model_version"])
    out.update(lookup(out["species"]))            # wood_type + valid uses only
    return out
```

**Training and serving must use exactly the same preprocessing.** Add a test that runs 20 test-split images through both the notebook path and `predict()` and checks the results match.

### 13.3 API

```python
from fastapi import FastAPI, UploadFile, File
import numpy as np, cv2
from inference import predict

app = FastAPI()

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return {"status": "error", "reason": "unreadable_image"}
    return predict(img)

@app.get("/species")
def species(): ...        # the species list for the "Species covered" page

@app.get("/health")
def health(): return {"ok": True, "model_version": CFG["model_version"]}
```

Add a file-size cap (about 8 MB), accepted image types, a timeout (about 30 s), and clear error messages. Never crash on a bad image.

### 13.4 Screens (four only)

1. **Identify** (home page)
2. **Species covered** (list with local name, botanical name, wood type; searchable)
3. **How to photograph**
4. **About and limits** (scope, what the percentage means, privacy, model version)

### 13.5 Identify screen

1. A short expandable "how to photograph" strip: photograph the **cut end-grain (cross-section)** only; flat clean cut; hold steady; fill the frame with wood.
2. A large **Take photo / Choose photo** button (`<input type="file" accept="image/*" capture="environment">`).
3. Preview with a basic quality message (too small, too blurry).
4. **Identify** button. While waiting show a spinner and "Analysing 48 sections of your image".
5. Result states:
   - `ok`: result card (13.6)
   - `retake`: friendly reason ("Too blurry, hold steady and refocus" or "Not sure what this is, try a flatter, cleaner cut surface"). Show top-3 only if low confidence, labelled "not certain".
   - `error`: plain message with a retry button
   - server asleep or offline: "The server is waking up, please try again in a moment"

### 13.6 Result card (in this order)

- Species local name (large), scientific name (italic), English name if available
- **Hardwood** badge
- **Confidence bar with percentage** and "38 of 48 image sections agree"
- Top-3 list with percentages
- If `limited_reference_data`: "This species was trained on fewer than 3 specimens. Treat the result with extra caution."
- **Documented uses** as chips (primary solid, secondary outline, with a legend), only if valid uses exist. Caption: "Documented uses from the literature, not predicted from your photo."
- One-line scope note: "Covers 29 species from Ibadan wood markets only."
- Retake button

### 13.7 Design and accessibility

- Plain and serious. One accent colour on a light background. System fonts. No decoration.
- Mobile first. Test at 360 px width.
- Text at least 16 px, tap targets at least 44 px, contrast at least 4.5:1, labels on all controls.
- Light page. Slow mobile data is normal, so show progress.
- English interface, showing Yoruba/local names.

### 13.8 Privacy

- Do **not** store uploaded images. Process in memory and discard.
- Log only anonymous counts (time, status, confidence, model version), never images.
- The "About and limits" page says the app is not for legal, import/export or structural-grading decisions.

### 13.9 Tests

- `predict()`: normal, blurry, low-confidence, wrong file type, huge file, tiny image
- Parity test (13.2)
- API tests for `/predict`, `/species`, `/health`
- Run it on a real phone with real photos

---

## 14. Short checklist

- [ ] Backup of the current image folder
- [ ] Images confirmed grid-ready (7.2)
- [ ] Manifest and checks reported (7.3, 7.4)
- [ ] Region groups reviewed by me (8.3)
- [ ] Splits written once, leak audit passes for every fold
- [ ] Both models trained on identical folds and settings
- [ ] Metrics on independent-specimen tests only; single-specimen species reported separately
- [ ] Threshold set from out-of-fold predictions
- [ ] Parity test passes
- [ ] Website works on phone and on my laptop; demo video recorded
- [ ] Pinned `requirements.txt`, seeds, configs, original notebook copy kept
- [ ] `README.md` with commands to reproduce every table and figure
- [ ] `CHAPTER_4_INPUTS.md`

---

## 15. Reporting format (after every phase)

1. What you did (5 to 10 lines, plain language)
2. What you found (numbers, tables, anything surprising)
3. What you changed from this brief (should be nothing; if something, why)
4. Open questions (numbered, answerable in one line)
5. What you will do next, then wait for my "go"

Done means: all phases approved, all tests pass, every thesis table and figure reproducible from the README, website deployed AND running locally, demo video recorded.

---

## 16. Later (NOT now, do not build unless I ask)

These were deliberately removed to save time:

- Recording market, lighting, camera or lens, surface preparation, capture time and expert-agreement per image
- Leave-one-market-out testing
- Shortcut-learning checks (capture-time analysis, background checks) and Grad-CAM
- Testing with species outside the 29
- CITES status tracking and badges
- Temperature scaling and reliability diagrams
- Ablation (patch voting vs whole image)
- Model card
- Pilot test with timber traders
- Optional user feedback upload
- Yoruba interface

---

## 17. First reply

Your first reply is Phase 0 only: a summary of the notebook (framework, patch size, normalisation, pooling, number of images), any conflicts with this brief, anything missing from the inputs list, and the open questions in Section 3 that you need answered. Then stop.
