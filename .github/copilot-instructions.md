# GitHub Copilot Instructions for Nigerian Timber Species Identification

## Project Overview
This repository implements a macroscopic wood species identification pipeline and web app for 29 Nigerian timber species traded in Ibadan wood markets, based on transverse (end-grain / cross-section) surface photos.

- Owner: Dimeji (B.Sc. Wood Products Engineering, University of Ibadan)
- Primary Architecture: ShuffleNetV2
- Comparison Architecture: ResNet18
- Banned Architecture: MobileNetV2 must NEVER appear anywhere in this codebase.

## Locked Architectural & Scientific Rules
1. **Model Scope**: Single classification head predicting species ONLY. Loss is cross-entropy with class weighting. No multi-task heads, no uncertainty loss, no SMOTE, no MixUp.
2. **Wood Type & Documented Uses**: All 29 species are angiosperms (Hardwood). Documented uses and wood type are looked up strictly from `species_enduse.json`, NOT predicted from the photo. Never invent or hallucinate uses or citations.
3. **Patch-based Pipeline**: TDLI-PIV style (Herrera-Poyatos et al., 2024). Each transverse image is partitioned into a 6x8 grid (48 patches). Inference evaluates each patch and performs majority voting (ties broken by mean probability).
4. **Surface Orientation**: Transverse (cross-section / end-grain) only. Never use the term "longitudinal" as a surface name.
5. **Specimen-Grouped Splits**: Splits are strictly grouped by specimen (and clustered region groups) to prevent Same-Specimen-Picture Bias. Images from the same specimen must never leak across train/val/test. Augmentation is applied to the training split ONLY.
6. **Confidence Metric**: Confidence percentage represents patch agreement (number of agreeing patches / 48 * 100), not an absolute probability guarantee.
7. **Nomenclature**: Species names strictly follow Keay (1989), *Trees of Nigeria*.

## Coding Standards & Workflow
- Use modular structure under `src/` (`data.py`, `split.py`, `model.py`, `train.py`, `evaluate.py`, `infer.py`).
- All randomness must be seeded (Python `random`, `numpy`, `torch`).
- Type hints and docstrings are required for all functions.
- Training and serving must share the identical `predict()` and preprocessing pipeline.
