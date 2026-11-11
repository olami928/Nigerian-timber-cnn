# Project progress and continuation handoff

Last updated: 11 November 2026
Submission deadline stated by owner: **10 November 2026 (passed; no time was specified)**

## Copyable project record

- The self-contained decisions, phase process, training/evaluation
  configuration, Chapter 4 metrics, per-species results, limitations, and
  pending questions are recorded directly in [PROJECT_RECORD.md](./PROJECT_RECORD.md).
- Read that record first when continuing in a new session; it contains the
  results themselves, not just pointers to generated artifacts.
- This progress file remains the short operational handoff. Update both this
  file and `PROJECT_RECORD.md` whenever a phase, decision, test, or result
  changes.

## Current checkpoint

- **Phase 0 (read and report): reported.**
- **Phase 1 (data): complete.**
- **Phase 2 (training): complete; both architectures and all three folds trained in Colab.**
- **Phase 3 (evaluation): complete; reports and figures are in [artifacts/p3_evaluation/](./artifacts/p3_evaluation/).**
- **Phase 4 (inference): complete; shared predictor and 20-image notebook parity test are in place.**
- **Phase 5 (website): implemented as a static browser-inference site in [app/](./app/); not yet owner-approved, deployed, or phone-tested. It carries the low-accuracy limitation notice.**
- **Optimized retraining: prepared but not run.** A regularized recipe is in [configs/default.yaml](./configs/default.yaml), the diagnosis is in [ARCHITECTURE_OPTIMIZATION_REPORT.md](./ARCHITECTURE_OPTIMIZATION_REPORT.md), and a resume-safe Colab workflow is in [notebooks/colab_training.ipynb](./notebooks/colab_training.ipynb). The local machine has CPU-only PyTorch and no CUDA, so the full run must happen on Colab/GPU. All 32 local tests pass. No improved accuracy result exists yet.
- Dimeji said “go” for Phase 1 on 10 November 2026.
- Dimeji approved the proposed threshold-0.25 region groups on 10 November 2026 after review of the contact sheet.
- Supervisor approval for specimen-grouped three-fold cross-validation was confirmed on 10 November 2026.
- The owner selected Vercel for hosting; website work remains gated until the later phases.
- No image data has been changed. The canonical manifest/splits now use the approved groups; previous versions were preserved.
- The initial local CPU training attempt did not complete a fold; the approved full training run later completed on CUDA and its artifacts are saved under [artifacts/p2_training/](./artifacts/p2_training/).
- Phase 3 used the verified image backup and saved specimen-bootstrap confidence intervals. Independent-specimen image accuracy was low (12.1% ShuffleNetV2, 11.3% ResNet18); do not enable live predictions or present these as reliable deployment results.
- Continue from this file and the latest conversation; do not repeat completed reading or start the project over.
- On 11 November 2026, the owner requested a copyable, self-contained record
  of decisions and results. `PROJECT_RECORD.md` was created with the observed
  P0–P4 decisions, process, training details, evaluation metrics, per-species
  scores, and pending decisions.

## Phase 0 findings

- The reference notebook is [notebooks/timber_species_identification.ipynb](./notebooks/timber_species_identification.ipynb).
- It uses PyTorch/torchvision and specifies ShuffleNetV2 as primary and ResNet18 as comparison.
- It specifies a 6-row by 8-column grid: 48 patches per image, followed by patch classification and majority voting with mean-probability tie-breaking.
- Preprocessing uses ImageNet normalization: mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`.
- The torchvision backbone's adaptive average pooling remains in the forward path; the notebook replaces `model.fc` with `Linear -> LeakyReLU(0.1) -> Dropout(0.5) -> Linear`. There is no global max pooling. The notebook comment's “AdaptiveAvgPool -> Linear...” describes the backbone-to-classifier path, not a pooling layer inserted into the replacement `fc` sequence.
- The repository's [data/manifest.csv](./data/manifest.csv) currently has 900 image rows, 29 species, and 45 specimen codes. The notebook itself calculates the count from its manifest at runtime.
- The Phase 1 audit details and limitations are recorded below.
- [species_enduse.json](./species_enduse.json), the notebook, and the project brief are present.
- No separate Yoruba/English/botanical names list was found in the repository.

## Phase 1 data audit and accepted split

The owner identified [patch_ready_dataset](./patch_ready_dataset) as the location of untouched original images. [patch_ready_dataset_raw_backup](./patch_ready_dataset_raw_backup) also exists, but its relationship to that original set has not been independently confirmed. No image files were changed in Phase 1.

- [data/manifest.csv](./data/manifest.csv) has 900 rows with all required columns. Every referenced image path exists; all 900 images passed Pillow verification; there are no repeated SHA-256 hashes.
- All 29 species labels match [configs/species.csv](./configs/species.csv). Specimen indices are consecutive from 1, with no gaps or species above 3 specimens. Epin's `epin` folder exists; there is no `iha` folder in the prepared dataset.
- There are 45 specimens: 15 species have one specimen, 12 have two, and 2 have three. Single-specimen species: Agboin, Apa, Emido, Epin, Epu, Gmelina, Ire, Iya, Iyeye, Mahogany, Mango, Obobo, Okobo, Oro, and Osun.
- All 900 images meet the 8-column/6-row divisibility check; 0 fail. There are 271 distinct image dimensions (103 occur once). Width ranges 632–1600 px and height 402–1596 px; most common size is 1600×1200 (114 images). Implied source patch dimensions range from 79–200 px wide and 67–266 px high; 115 images yield square patches and 785 rectangular patches. The notebook resizes extracted patches to 112×112 before model input.
- **Species order:** Dimeji chose [configs/species.csv](./configs/species.csv) as canonical on 10 November 2026 (Epin ID 11, Gmelina ID 12), although the brief numbers Gmelina #11 and Epin #12. Also, [artifacts/class_names.json](./artifacts/class_names.json) places Epin before Epu, unlike the config. Treat any existing model/class-name artifacts as potentially misaligned until their class-index mapping is verified; new training/serving must use the chosen config order.
- Before review, `data/manifest.csv` and `data/splits.csv` had 16 specimens collapsed into one group (group-count distribution: 16 with 1, 13 with 2, 12 with 3, 2 with 4, 1 with 6, 1 with 8 groups). Dimeji approved the proposed threshold-0.25 grouping after reviewing [artifacts/region_groups_proposed_thr_0.25.pdf](./artifacts/region_groups_proposed_thr_0.25.pdf); it has 2–4 groups per specimen (6 with 2, 6 with 3, 33 with 4), with no specimen collapsed to one group.
- Supervisor approval for specimen-grouped 3-fold CV was confirmed. The accepted groups were installed in [data/manifest.csv](./data/manifest.csv), and [data/splits.csv](./data/splits.csv) plus [artifacts/leak_audit.csv](./artifacts/leak_audit.csv) were regenerated programmatically with seed 42. All folds pass specimen-, group-, and hash-leak checks. Counts are fold 1: 372/129/399, fold 2: 451/153/296, fold 3: 661/131/108 (train/validation/test).
- Previous unreviewed canonical files were preserved as [artifacts/unreviewed_region_groups_manifest.csv](./artifacts/unreviewed_region_groups_manifest.csv), [artifacts/unreviewed_region_groups_splits.csv](./artifacts/unreviewed_region_groups_splits.csv), and [artifacts/unreviewed_region_groups_leak_audit.csv](./artifacts/unreviewed_region_groups_leak_audit.csv).
- `python -m unittest tests.test_splits -v` passed all 6 tests.

## Open decisions and inputs

1. **D8 — hosting:** Vercel selected by the owner.
2. **D9 — deadline:** answered: today, 10 November 2026. No time of day was provided.
3. **D3 — split protocol:** supervisor-approved; accepted for the current plan.
4. **Region grouping:** threshold-0.25 groups approved by the owner on 10 November 2026 and used in the canonical P1 artifacts.
5. **Backup:** owner says the untouched originals are in `patch_ready_dataset`; `patch_ready_dataset_raw_backup` also exists, but its relationship to the original set is not confirmed.
6. **Species order:** owner selected the config order (Epin ID 11, Gmelina ID 12). Training reads this order from `configs/species.csv`; legacy class-name/model artifacts were not modified.
7. **D2 — pooling:** reported as adaptive average pooling (not max pooling); clarify only if the owner intended a different architecture question.
8. Names list: not present in the repository; request it only if needed for later work.

## Phase 2 training run

- The existing full training recipe remains `configs/default.yaml`, seed 42, and output directory `artifacts/p2_training/`.
- It trains ShuffleNetV2 and ResNet18 over all three approved folds, then trains final models using the median best stage-2 epoch count. Frozen-head stage: 10 epochs, Adam 0.001. Fine-tuning stage: up to 20 epochs, Adam 0.0001, patience 7. Batch size 64. Augmentation: ±10° rotation and horizontal/vertical flips, no translation. ImageNet normalization; patches resized to 112×112.
- The local environment has CPU-only PyTorch and 4 CPU cores; ImageNet pretrained weights are cached. The first local CPU run did not complete its first fold in about an hour, so it was not treated as a viable training run.
- The full CUDA run completed for ShuffleNetV2 and ResNet18 across all three folds, followed by final models trained on all 900 images using each architecture's median best stage-2 epoch count. Its summary is [artifacts/p2_training/training_summary.json](./artifacts/p2_training/training_summary.json).
- The canonical training output uses class-index order from [configs/species.csv](./configs/species.csv); the training summary confirms this order for every fold and final model. Existing root-level model artifacts remain untouched.
- [src/train.py](./src/train.py) now supports optional `--cache-patches` and `--patch-cache-path`: it precomputes the 48 cropped/resized 112×112 uint8 patches per image into a memory-mapped cache, so training does not reopen and decode each full-resolution image for every patch/epoch. This is opt-in; it does not alter model or augmentation settings.
- The training pipeline now supports `--resume`, reuses only complete fold/final outputs whose metadata matches the current architecture, fold, class order, and image counts, and writes completed outputs through staging directories. Interrupted training within a fold restarts that fold; incomplete staging directories are reported rather than silently overwritten.
- [notebooks/colab_training.ipynb](./notebooks/colab_training.ipynb) is a separate Colab workflow. It reads the Drive copy, uses Colab local storage for code/images/cache, saves fold and final artifacts to Drive, verifies the approved class order and all three leak audits, and can resume completed outputs after disconnect. It does not modify the original thesis notebook.
- Validation: `python -m unittest tests.test_train tests.test_splits -v` passed all 13 tests; notebook JSON and all code cells compile; `python -m src.train --help` exposes the cache/resume options; Pylance reports no `src/train.py` diagnostics.

## Phase 3 evaluation

- Evaluation used the approved fold files and verified all 900 source images against the saved manifest hashes and dimensions. All 900 remained readable and divisible into the 6×8 patch grid; no image files were changed.
- Independent out-of-fold predictions cover 611 images from 30 specimen codes but only 14 of the 29 species (the 14 species with at least two specimens). Results below are image-level majority-vote accuracy, macro-F1, and MCC, with 95% confidence intervals resampling specimen codes within species:

  | Architecture | Accuracy (95% CI) | Macro-F1 (95% CI) | MCC (95% CI) |
  |---|---:|---:|---:|
  | ShuffleNetV2 | 12.1% (8.5–15.7%) | 14.8% (10.6–18.0%) | 0.083 (0.045–0.121) |
  | ResNet18 | 11.3% (8.6–14.1%) | 13.2% (10.2–15.6%) | 0.074 (0.043–0.105) |

- Patch-level accuracy is 9.7% for ShuffleNetV2 and 10.0% for ResNet18. These independent-specimen results are poor; the strong within-specimen results must not be used as headline results.
- The separate within-specimen (optimistic) fold-1 cohort contains 64 images from 10 single-specimen species: image accuracy is 93.8% for ShuffleNetV2 and 95.3% for ResNet18. These results do not demonstrate generalization to a new specimen.
- The most frequent confusion for both models is Ita versus Oporoporo (39 combined image errors for ShuffleNetV2; 41 for ResNet18). [artifacts/p3_evaluation/most_confused_pairs.csv](./artifacts/p3_evaluation/most_confused_pairs.csv) and confusion matrices contain the complete counts.
- Model comparison: ShuffleNetV2 has 1,523,457 parameters and a 5.97 MB PyTorch checkpoint; ResNet18 has 11,315,293 parameters and a 43.23 MB checkpoint.
- [artifacts/p3_evaluation/confidence_threshold_curve.csv](./artifacts/p3_evaluation/confidence_threshold_curve.csv) reports coverage/accuracy trade-offs. No confidence threshold or blur cutoff was selected. At 79.2% patch agreement, only 4.1% of ShuffleNetV2 predictions are accepted at 52.0% accuracy; ResNet18 accepts 4.4% at 55.6%. This does not support a useful retake threshold yet.
- The 300-dpi figures were generated, but Times New Roman is unavailable in the evaluation environment; Matplotlib emitted font-fallback warnings. Correct the figure font before thesis use.
- Outputs include [artifacts/p3_evaluation/overall_metrics.csv](./artifacts/p3_evaluation/overall_metrics.csv), per-species metrics, patch/image predictions, model comparison, confusion matrices, threshold candidates, sharpness percentiles, figures, and [artifacts/p3_evaluation/evaluation_summary.json](./artifacts/p3_evaluation/evaluation_summary.json). The specimen-bootstrap implementation was vectorized to keep the required 1,000 resamples practical. `python -m unittest tests.test_evaluate -v` passes all 4 tests; Pylance reports no evaluator diagnostics.

The evaluation reveals a substantial generalization problem and independent results exist for only 14 species. Keep the website prediction-disabled; do not select a threshold or claim model readiness.

## Phase 4 shared inference

- [src/infer.py](./src/infer.py) loads the final primary ShuffleNetV2 export and validates its architecture, metadata, 29-class order, 6×8 grid, and preprocessing configuration before inference.
- `TimberPredictor.predict(image_bgr)` and the module-level lazy `predict(image_bgr)` use the notebook's raster-order 48 crops, 112×112 resize, ImageNet normalization, patch softmax, majority vote, and mean-probability tie-break.
- Results include patch-agreement confidence (not a correctness guarantee), mean probability, top three, training-specimen availability, model version, wood type, and documented uses only. Lookup sources are not exposed; categories not marked primary or secondary are omitted.
- The exported confidence and sharpness thresholds are both `null`, so the predictor does not invent a retake cutoff. The poor independent-specimen performance remains a serious limitation: this inference module is not wired into the site and must not be presented as a validated identification service.
- [tests/test_infer.py](./tests/test_infer.py) compares the model's prediction fields against the original notebook's patch-inference path on 20 deterministically selected held-out images. The parity test passed, as did the malformed-image and documented-use filtering tests (3 tests total).

Phase 4 is complete. Do not start Phase 5 until the owner says “go”.

## Working rules

The complete project instructions are in [AGENT_BRIEF_timber_id.md](./AGENT_BRIEF_timber_id.md). Keep its phase gates and locked scientific decisions. In particular, never use MobileNetV2, do not train on GOIMAI-Phase-I, do not modify image data, do not invent names/uses/citations/results, and only show documented uses from the supplied lookup JSON.
