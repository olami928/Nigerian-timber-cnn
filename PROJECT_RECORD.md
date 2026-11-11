# Nigerian Timber Identification — Decisions, Process, and Results

**Last updated:** 11 November 2026  
**Purpose:** Self-contained handoff and copyable project record. Results and
decisions are written here directly; this is not just an index of files.

## 1. Current status

- Phases 0–4 are complete: notebook review, data and split audit, model
  training, evaluation, and shared inference/parity testing.
- Phase 5 is implemented but not approved: a static browser-inference site
  exists in `app/` and prominently reports the poor independent-specimen
  accuracy. It has not been deployed or phone-tested. Phase 6 therefore
  remains incomplete.
- An optimized regularized recipe and resume-safe Colab workflow are prepared
  (`configs/default.yaml`, `notebooks/colab_training.ipynb`). The full
  retraining has not been run: this machine has CPU-only PyTorch, so it must
  run on a GPU runtime. No improved-accuracy claim exists yet.
- Independent-specimen results are poor and do not support presenting the
  model as a reliable identification service.
- The evaluation is complete. It used the local CPU; no result upload to
  Google Drive is confirmed.
- No image data was modified.
- The stated submission deadline was 10 November 2026; no deadline time was
  given. That date has passed.

## 2. Decisions taken

| Topic | Decision / recorded status |
|---|---|
| Primary model | ShuffleNetV2 |
| Comparison model | ResNet18 |
| Prediction target | Species only; one classification head per model |
| Architecture restriction | Do not use MobileNetV2 |
| Loss and imbalance | Cross-entropy with class weights |
| Prohibited training techniques | No multi-task head, uncertainty-weighted loss, SMOTE, or MixUp |
| Inference method | 6-row × 8-column grid (48 patches), classify each patch, majority vote; mean probability breaks ties |
| Surface | Transverse/end-grain only |
| Dataset source | Nigerian timber specimens from Ibadan wood markets; do not train on GOIMAI-Phase-I |
| Splits | Specimen-grouped 3-fold CV; supervisor approval was confirmed |
| Region grouping | ResNet18 ImageNet embeddings, cosine distance threshold 0.25, average linkage; owner reviewed and approved the contact sheet |
| Species order | `configs/species.csv` order was selected as canonical. Epin is class 11 and Gmelina class 12 in this order; the brief's numbered list has the reverse order. Trained fold metadata matches the selected config order. |
| Pooling | Notebook uses the backbone's adaptive average pooling, not global max pooling; classifier head includes LeakyReLU(0.1) and Dropout(0.5). |
| Class names | Follow Keay (1989), *Trees of Nigeria*. Epin is *Ficus exasperata* Vahl; Obobo is *Ficus mucuso*. |
| Wood type and uses | All 29 are hardwoods. Lookup only documented uses from the supplied `species_enduse.json`; do not infer uses from images or invent citations. |
| Hosting | Owner selected Vercel. Website work has not been approved to start. |
| Confidence meaning | Patch agreement (winning patch votes / 48), not a probability guarantee. |
| Confidence threshold | No retake threshold selected: no accepted-accuracy/coverage target was provided, and test performance does not support a useful cutoff. |
| Blur threshold | No sharpness cutoff selected. |
| Names list | No separate Yoruba/English/botanical names list was found in the repository. Do not fill English names from memory. |
| Thesis writing | The owner writes thesis prose; provide evidence and inputs, not thesis chapters. |

## 3. Dataset, validation, and split process

- The prepared dataset contains **900 images**, **29 species**, and **45
  specimens**. Image folders represent specimens.
- Specimen counts: 15 species have one specimen, 12 have two, and 2 have
  three. Single-specimen species are Agboin, Apa, Emido, Epin, Epu, Gmelina,
  Ire, Iya, Iyeye, Mahogany, Mango, Obobo, Okobo, Oro, and Osun.
- All 900 images passed readability and 6×8 divisibility checks; there were
  no exact duplicate SHA-256 image files. No images were changed.
- Patch-ready dimensions yield 271 distinct image sizes; the most common is
  1600×1200 (114 images). Implied source patches range from 79–200 px wide
  and 67–266 px high; 115 images yield square patches and 785 rectangular
  patches. Model input patches are resized to 112×112.
- Epin's `epin` folder exists; there is no `iha` folder in the prepared
  dataset.
- The owner identified `patch_ready_dataset` as the untouched originals.
  A raw backup also exists; its relationship to the originals was not
  independently confirmed.
- The owner approved region groups at distance threshold 0.25 after reviewing
  a contact sheet. Groups are kept together across train/validation/test.
- Split seed is 42. The saved 3-fold split passes specimen, group, and
  duplicate-hash leak audits.
- Fold image counts (train / validation / test):

  | Fold | Train | Validation | Test |
  |---:|---:|---:|---:|
  | 1 | 372 | 129 | 399 |
  | 2 | 451 | 153 | 296 |
  | 3 | 661 | 131 | 108 |

- Independent-specimen evaluation covers 611 test images from 30 specimen
  codes, representing only **14 of 29 species** (species with at least two
  specimens). The remaining 15 single-specimen species cannot contribute to
  independent-specimen headline metrics.
- The separate fold-1 within-specimen (optimistic) cohort contains 64 images
  from 10 single-specimen species. Do not combine these scores with
  independent-specimen results.

## 4. Training process and configuration

- Both architectures were trained on all three approved folds, then final
  models were trained on all 900 images using the median best stage-2 epoch
  count from their folds.
- Training ran in Colab with CUDA. The initial local CPU training attempt did
  not finish a fold in about an hour and was not used as the completed run.
- Seed: 42, with fold-specific seeds recorded in training metadata.
- PyTorch / torchvision: 2.14.1+cu130 / 0.29.1+cu130. Python: 3.13.16.
- Other recorded versions: NumPy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1,
  Pillow 11.3.0.
- Image pipeline: 48 raster-order patches, resized to 112×112; ImageNet
  normalization mean `[0.485, 0.456, 0.406]`, standard deviation
  `[0.229, 0.224, 0.225]`.
- Stage 1: freeze backbone, train head for 10 epochs, Adam, learning rate
  0.001.
- Stage 2: unfreeze all layers, Adam, learning rate 0.0001, up to 20 epochs,
  patience 7.
- Batch size: 64. Augmentation on training data only: rotation ±10°,
  horizontal flip, vertical flip, no translation.
- Final stage-2 epochs: ShuffleNetV2 17 (fold best stage-2 epochs 20, 17,
  15); ResNet18 14 (fold best stage-2 epochs 14, 9, 20).
- Class order for every trained fold and final model matches the chosen
  `configs/species.csv` order.

## 5. Evaluation method and headline results

- Evaluation ran on CPU over the trained fold models. Each image contributes
  48 patch predictions, which are combined by majority vote. Confidence
  intervals are 95% percentile bootstrap intervals resampling specimen
  codes within species, not individual images.
- Image-level metrics below are the independent-specimen cohort and are the
  appropriate headline results. Values are percentages except MCC.

| Model | Level | Images / patches | Accuracy (95% CI) | Macro precision (95% CI) | Macro recall (95% CI) | Macro F1 (95% CI) | MCC (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|
| ShuffleNetV2 | Image | 611 images | 12.1 (8.5–15.7) | 22.0 (16.5–27.3) | 11.8 (8.1–15.7) | 14.8 (10.6–18.0) | 0.083 (0.045–0.121) |
| ResNet18 | Image | 611 images | 11.3 (8.6–14.1) | 21.4 (17.2–29.7) | 10.7 (8.0–14.3) | 13.2 (10.2–15.6) | 0.074 (0.043–0.105) |
| ShuffleNetV2 | Patch | 29,328 patches | 9.7 (7.7–12.0) | 17.4 (14.4–21.2) | 9.6 (7.6–12.4) | 11.9 (9.6–14.3) | 0.056 (0.035–0.084) |
| ResNet18 | Patch | 29,328 patches | 10.0 (8.6–11.8) | 18.2 (15.6–22.2) | 9.6 (8.3–11.7) | 12.0 (10.6–13.8) | 0.062 (0.046–0.080) |

**Interpretation:** independent-specimen generalization is poor. Do not call
these models reliable, enable live predictions, or use the within-specimen
results as headline performance.

### Within-specimen results (optimistic; report separately)

| Model | Level | Images / patches | Accuracy | Macro precision | Macro recall | Macro F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|
| ShuffleNetV2 | Image | 64 images / 10 species | 93.8% | 97.1% | 92.3% | 94.1% | 0.930 |
| ResNet18 | Image | 64 images / 10 species | 95.3% | 100.0% | 93.5% | 96.3% | 0.947 |
| ShuffleNetV2 | Patch | 9,216 patches / 10 species | 81.4% | 93.4% | 80.9% | 86.4% | 0.793 |
| ResNet18 | Patch | 9,216 patches / 10 species | 82.7% | 94.7% | 81.6% | 87.4% | 0.807 |

These are within the same specimens used for training and are optimistic;
they do not demonstrate generalization to new specimens. No confidence
intervals were calculated for this cohort.

## 6. Per-species independent-specimen metrics

Precision, recall, and F1 are percentages. Support is the number of test
images. Only the 14 species with independent specimens are represented.

| Species | Support | Specimens | ShuffleNetV2 P / R / F1 | ResNet18 P / R / F1 |
|---|---:|---:|---:|---:|
| Afara | 50 | 2 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Araba | 55 | 2 | 88.9 / 29.1 / 43.8 | 92.3 / 21.8 / 35.3 |
| Arere | 19 | 2 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Awun | 75 | 3 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Ayin | 38 | 2 | 30.4 / 18.4 / 23.0 | 44.4 / 21.1 / 28.6 |
| Ayunre | 67 | 2 | 33.3 / 7.5 / 12.2 | 44.4 / 11.9 / 18.8 |
| Iroko | 35 | 2 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Irugba | 25 | 2 | 35.7 / 20.0 / 25.6 | 14.3 / 4.0 / 6.3 |
| Ita | 40 | 2 | 14.3 / 15.0 / 14.6 | 4.0 / 5.0 / 4.4 |
| Laoro | 51 | 2 | 10.3 / 5.9 / 7.5 | 3.7 / 2.0 / 2.6 |
| Obi | 28 | 2 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Oporoporo | 58 | 3 | 1.3 / 1.7 / 1.4 | 4.2 / 6.9 / 5.2 |
| Parapuru | 46 | 2 | 93.9 / 67.4 / 78.5 | 83.3 / 65.2 / 73.2 |
| Teak | 24 | 2 | 0.0 / 0.0 / 0.0 | 9.1 / 12.5 / 10.5 |

### Per-species within-specimen results (optimistic)

| Species | Support | ShuffleNetV2 P / R / F1 | ResNet18 P / R / F1 |
|---|---:|---:|---:|
| Epin | 9 | 100 / 100 / 100 | 100 / 100 / 100 |
| Gmelina | 5 | 100 / 100 / 100 | 100 / 100 / 100 |
| Ire | 10 | 90.9 / 100 / 95.2 | 100 / 100 / 100 |
| Iya | 4 | 80 / 100 / 88.9 | 100 / 100 / 100 |
| Iyeye | 4 | 100 / 100 / 100 | 100 / 100 / 100 |
| Mahogany | 14 | 100 / 92.9 / 96.3 | 100 / 100 / 100 |
| Obobo | 4 | 100 / 75 / 85.7 | 100 / 75 / 85.7 |
| Okobo | 5 | 100 / 80 / 88.9 | 100 / 80 / 88.9 |
| Oro | 5 | 100 / 100 / 100 | 100 / 80 / 88.9 |
| Osun | 4 | 100 / 75 / 85.7 | 100 / 100 / 100 |

## 7. Confusions, model size, confidence, and image sharpness

### Most frequent independent-specimen image confusions

| Model | Species pair | First → second | Second → first | Combined |
|---|---|---:|---:|---:|
| ShuffleNetV2 | Ita / Oporoporo | 25 | 14 | 39 |
| ShuffleNetV2 | Awun / Oporoporo | 7 | 13 | 20 |
| ShuffleNetV2 | Ayunre / Mahogany | 20 | 0 | 20 |
| ShuffleNetV2 | Awun / Oro | 18 | 0 | 18 |
| ShuffleNetV2 | Ayunre / Iroko | 10 | 6 | 16 |
| ResNet18 | Ita / Oporoporo | 27 | 14 | 41 |
| ResNet18 | Awun / Ita | 26 | 0 | 26 |
| ResNet18 | Laoro / Teak | 17 | 4 | 21 |
| ResNet18 | Awun / Oporoporo | 12 | 8 | 20 |
| ResNet18 | Laoro / Oporoporo | 6 | 12 | 18 |

### Model size

| Model | Parameters | PyTorch checkpoint | ONNX model |
|---|---:|---:|---:|
| ShuffleNetV2 | 1,523,457 | 5.97 MB | 5.90 MB |
| ResNet18 | 11,315,293 | 43.23 MB | 43.16 MB |

### Patch-agreement threshold evidence

No threshold has been selected. These are test-set trade-off examples, not
an approved app cutoff:

| Minimum patch agreement | Model | Accepted images | Accepted share | Accuracy among accepted |
|---:|---|---:|---:|---:|
| 79.2% | ShuffleNetV2 | 25 / 611 | 4.1% | 52.0% |
| 79.2% | ResNet18 | 27 / 611 | 4.4% | 55.6% |

The confidence-curve values are based on independent-specimen out-of-fold
predictions. Owner/supervisor must set a desired accepted-accuracy/coverage
target before a retake threshold can be chosen. Confidence is patch agreement,
not a guarantee of correctness.

### Sharpness distribution

Laplacian-variance percentile values for the 900 images used to train the
final models:

| Percentile | Sharpness |
|---:|---:|
| 1 | 117.80 |
| 5 | 171.40 |
| 10 | 206.89 |
| 25 | 279.21 |
| 50 | 391.69 |
| 75 | 629.11 |
| 90 | 912.17 |
| 95 | 1,079.46 |
| 99 | 1,532.78 |

No blur threshold was selected.

## 8. Phase 4 inference and checks

- One shared predictor uses the same 48-patch raster-order preprocessing,
  112×112 resize, ImageNet normalization, patch softmax, majority voting, and
  mean-probability tie-break as the notebook.
- It returns species, patch-agreement confidence, mean probability, top
  three, limited-reference-data status, model version, and hardwood/documented
  uses from the supplied lookup. It must not expose undocumented uses or
  invented source details.
- Both confidence and blur cutoffs remain unset (`null`); no arbitrary
  retake decision was introduced.
- A parity test compared predictions with the notebook path on 20
  deterministically selected held-out images and passed. Malformed-image and
  documented-use filtering tests passed (3 inference tests total).
- Evaluation tests passed (4 tests); split tests passed (6 tests); training
  plus split tests passed (13 tests). The full preceding progress log records
  the checks and test outcomes.
- Evaluation verified all 900 backup images against manifest hashes and
  dimensions; all were readable and grid-divisible.
- Three 300-dpi Chapter 4 figures were generated. Times New Roman was
  unavailable, so Matplotlib used a fallback font. Recreate or correct the
  font before thesis use.

## 9. Pending questions and next action

1. What accepted-image coverage and accuracy target, if any, should guide a
   future confidence retake threshold? No threshold is currently approved.
2. What training-image sharpness percentile should be used for a blur cutoff?
   No cutoff is currently approved.
3. Provide the owner's names list if English/local-name reconciliation is
   needed; do not fill missing English names from memory.
4. Phase 5 implementation exists but has not been approved. Vercel is the
   selected host, but authentication, deployment, and phone testing remain
   incomplete.
5. Run the optimized GPU training, then reevaluate on the same independent-
   specimen cohort before updating the website model or making any accuracy
   claim. The current evaluation does not justify a working identification
   claim.

**Next agent:** Read this record and `progress.md` first. Continue from the
current phase gate; do not repeat phases 0–4, change image data, silently
alter the approved splits, or start Phase 5 without explicit approval.
