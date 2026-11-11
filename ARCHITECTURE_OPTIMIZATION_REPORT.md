# Architecture and Training Optimization Report

**Date:** 11 November 2026  
**Scope:** diagnosis-driven changes to the timber species classifier, training
recipe, and inference aggregation. No image data, split definitions, species
order, or documented end-use data were changed.

## 1. Executive summary

The low independent-specimen score is not caused by an under-capacity
classifier head. It is a generalization failure: the model fits specimen-
specific colour, lighting, and surface cues that are stable inside a specimen
but do not transfer to unseen specimens.

The existing evidence shows the pattern clearly:

| Model | Cohort | Images | Image accuracy | Patch accuracy |
|---|---|---:|---:|---:|
| ShuffleNetV2 | Same-specimen training sample | 300 | 1.000 | 0.971 |
| ShuffleNetV2 | Validation folds | 413 | 0.978 | 0.853 |
| ShuffleNetV2 | Independent-specimen test | 611 | **0.119** | **0.097** |
| ResNet18 | Same-specimen training sample | 300 | 1.000 | 0.965 |
| ResNet18 | Validation folds | 413 | 0.971 | 0.868 |
| ResNet18 | Independent-specimen test | 611 | **0.115** | **0.100** |

Source: `artifacts/reports/diagnosis/cohort_accuracy.csv`.

The optimization therefore targets **invariance and regularization**, not raw
model capacity:

1. a regularized classifier head with stronger dropout;
2. much stronger label-preserving augmentation for end-grain images;
3. specimen-balanced loss weighting so species with more images/specimens do
   not dominate;
4. AdamW weight decay, label smoothing, gradient clipping, and a warmup +
   cosine schedule;
5. optional test-time augmentation (TTA) and soft voting at inference.

The code and configuration now support all of the above. **No full retraining
was performed in this environment**, so this report does not claim an improved
independent-specimen accuracy. The current artifacts in `artifacts/p2_training`
remain the legacy models and are still readable.

## 2. What was changed

### 2.1 Configurable classifier heads (`src/model.py`)

- Added `HEAD_TYPES = ("legacy", "regularized", "linear")`.
- Added `build_head(...)`:
  - `legacy`: the original `Linear -> LeakyReLU(0.1) -> Dropout(0.5) ->
    Linear` head, retained for backward compatibility.
  - `regularized`: `Linear -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.5)
    -> Linear`, with `hidden_dim` configurable (default 128 in the new config).
  - `linear`: a single `Linear` layer, for a low-capacity ablation.
- `get_model(...)` accepts `head_type` and `hidden_dim` and stores them on the
  model for export/metadata. Unknown head types raise `ValueError`.
- MobileNetV2 remains explicitly forbidden.

### 2.2 Stronger, label-preserving augmentation (`src/train.py`)

`get_transforms(...)` now supports:

- rotation up to 180 degrees (previous recipe: +/-10 degrees);
- horizontal and vertical flips (both already present);
- translation up to 8% of the image;
- scale range 0.9-1.1;
- shear up to 5 degrees;
- brightness, contrast, saturation, and hue jitter;
- optional Gaussian blur;
- optional random erasing.

All of these are training-only. Evaluation and inference transforms are
unchanged.

### 2.3 Specimen-balanced loss (`src/train.py`)

- `compute_class_weights(..., weighting="image" | "specimen")`:
  - `image` reproduces the original inverse-image-frequency weights;
  - `specimen` computes inverse specimen frequency, then normalizes over
    specimens so each species contributes equally despite the 1-3
    specimens/species imbalance.
- `compute_sample_weights(...)`: optional per-image weights that make the
  total training weight of every species equal. Used when
  `class_weighting: specimen`.
- `run_epoch(...)` accepts per-sample weights and gradient clipping.

This is still a single-label cross-entropy objective. No multi-task head,
uncertainty weighting, SMOTE, or MixUp was introduced.

### 2.4 Optimizer and schedule (`src/train.py`)

`train_fold(...)` and `train_final_model(...)` now use:

- AdamW with configurable weight decay (default 0.01);
- label smoothing (default 0.1);
- optional per-sample weighting;
- discriminative learning rates: the stage-2 backbone LR is
  `lr_stage2 * backbone_lr_scale` (default 0.1) while the head keeps
  `lr_stage2`;
- gradient clipping (default norm 1.0);
- warmup + cosine schedule (default 2 warmup epochs);
- the original two-stage freeze/unfreeze structure is preserved.

### 2.5 Inference aggregation and TTA (`src/aggregate.py`)

New shared module used by training-time evaluation and production inference:

- `validate_tta_views(...)`: accepts `identity`, `hflip`, `vflip`, `hvflip`.
- `forward_probabilities(...)`: runs one forward pass per requested view and
  averages the softmax probabilities.
- `aggregate_probabilities(...)`: supports:
  - `majority`: the original winning-patch-vote rule, with mean probability
    as the tie-breaker;
  - `soft`: mean patch probability, then argmax.
- `aggregate_logits(...)`: convenience path for probability-valued tensors.

### 2.6 Evaluation and inference wiring (`src/evaluate.py`, `src/infer.py`)

- Both readers now load `head_type`, `hidden_dim`, `aggregation`, and
  `tta_views` from `final/<architecture>/preprocess_config.json`.
- Missing keys default to the legacy behaviour
  (`legacy`, 256, `majority`, no TTA), so existing artifacts keep working.
- `export_final_model(...)` writes the inference configuration into
  `preprocess_config.json` and `model_info.json`.
- Evaluation on CPU is now explicitly permitted; it still refuses a CUDA
  request when CUDA is unavailable.

### 2.7 Configuration files

- `configs/default.yaml`: new regularized recipe
  (`head_type: regularized`, `hidden_dim: 128`, specimen weighting, AdamW,
  label smoothing, warmup + cosine, stronger augmentation).
- `configs/cpu.yaml`: the same recipe with a reduced 3 + 5 epoch exploratory
  schedule.
- `configs/legacy.yaml`: the original notebook-matched recipe, preserved for
  reproducing the P2/P3 baseline.
- Default inference remains `majority` with no TTA to preserve parity with the
  current browser app. TTA/soft voting are opt-in.

### 2.8 Tests

Added/extended tests for:

- all three head types and invalid-head rejection;
- specimen-based class weights versus image-based weights;
- per-species total sample weight after normalization;
- presence of the new augmentation operations in training transforms only;
- soft voting overriding a hard majority when confidence is high;
- resume compatibility for the new metadata fields.

**Result:** `python -m unittest discover -s tests -v` passes all 32 tests in
the local CPU environment. The existing notebook-parity test
(`tests/test_infer.py`) also passes against the legacy artifact.

## 3. Why these changes are expected to help

- The gap between ~98% validation and ~12% independent-specimen accuracy is
  the signature of specimen-level shortcuts. More head capacity would make
  that worse, not better.
- 180-degree rotation, flips, scale, shear, and colour jitter attack the
  assumption that a specimen has one fixed orientation, scale, and colour
  cast.
- Specimen-balanced weighting prevents species with more images or more
  specimens from dominating the loss. The dataset has 15 single-specimen
  species, 12 two-specimen species, and 2 three-specimen species.
- Weight decay, label smoothing, and early stopping reduce overconfident
  memorization of individual images/patches.
- TTA and soft voting reduce variance from a single 48-patch raster sample;
  they are optional because they must be mirrored in the browser app before
  deployment.

## 4. What was deliberately not changed

- Image data, image order, manifest, hashes, and the approved specimen-grouped
  splits.
- The 6x8 patch grid, 112x112 patch size, ImageNet normalization, and
  majority-vote interface.
- Species class order and the documented end-use lookup.
- The prohibition on MobileNetV2, SMOTE, MixUp, multi-task heads, and
  uncertainty-weighted losses.
- The website remains prediction-disabled. No claim of production readiness
  is made.

## 5. How to reproduce / run

The ready-to-run Colab GPU notebook is
`notebooks/colab_training.ipynb`. It trains into
`artifacts/optimized_training`, then evaluates into
`artifacts/optimized_evaluation`.

Equivalent command-line run:

```bash
python -m src.train --config configs/default.yaml \
  --output-dir artifacts/optimized_training
python -m src.evaluate --training-dir artifacts/optimized_training \
  --output-dir artifacts/optimized_evaluation
```

Reduced CPU exploration:

```bash
python -m src.train --config configs/cpu.yaml \
  --output-dir artifacts/cpu_optimized_training
```

To evaluate the optional TTA/soft-vote path after a new export, set:

```yaml
inference:
  aggregation: "soft"
  tta_views: ["identity", "hflip", "vflip", "hvflip"]
```

and update `app/public/inference-core.mjs` and `app/public/app.js` at the same
time, because they currently hard-code majority voting.

## 6. Exploratory TTA check on the legacy fold-3 model

This is a diagnostic probe, not a headline result. It used the existing legacy
ShuffleNetV2 fold-3 checkpoint, 108 fold-3 test images (44
`independent_specimen`, 64 `within_specimen`), and no retraining:

| Configuration | All 108 images, image accuracy | 44 independent-specimen images |
|---|---:|---:|
| Identity views, majority vote | 54.6% | 0.0% (0/44) |
| 4 flip views, majority vote | 55.6% | 0.0% (0/44) |
| 4 flip views, soft vote | 54.6% | 0.0% (0/44) |

Interpretation: the small mixed-cohort change (+1.0 percentage point for
flip-majority) is within noise, and TTA/soft voting did not rescue the
independent subset. This is consistent with the diagnosis: the bottleneck is
specimen-level generalization, not single-view prediction variance. The
regularized training recipe is the main intervention; TTA/soft voting should
be evaluated only after retraining.

## 7. Limitations and open questions

- **No improved independent-score claim.** The optimized recipe has not been
  fully retrained here. CPU throughput is about 21 patches/s for ShuffleNetV2
  and about 7 patches/s for ResNet18, so a complete 3-fold + final run is not
  practical in this session.
- **TTA/soft voting are not proven gains.** The only empirical check is the
  legacy fold-3 probe in Section 6, which showed no gain on the 44 independent
  images. They remain a configuration-supported option to test after
  retraining.
- **Browser parity:** the deployed app does not implement TTA or soft voting.
  Enabling them in Python without changing the app would create a training/
  serving mismatch.
- **Absolute performance will likely remain low.** With 45 specimens and 29
  species, independent-specimen generalization is intrinsically difficult.
  The changes target the gap; they do not guarantee that 29-way accuracy on
  unseen specimens becomes high.
- **Next measurement:** retrain on a GPU, then compare independent-specimen
  image-level accuracy, macro F1, and MCC on the same 611-image cohort. Report
  the cohort numbers, not validation numbers, as the headline.
- **Do not combine** fold-1 within-specimen single-specimen results with the
  independent-specimen cohort.
