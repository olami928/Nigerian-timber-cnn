# Chapter 4 tables and figures

This index describes the generated evidence; it does not draft thesis claims.
The evaluation outputs are produced by `src/evaluate.py` from the saved
specimen-grouped fold models, test splits, and verified source images.

| Output | Supports |
|---|---|
| Table 4.1 — `artifacts/p3_evaluation/overall_metrics.csv` | Image-level voted predictions and patch-level metrics for the independent-specimen test cohort, with specimen-bootstrap confidence intervals; within-specimen results are clearly separate and marked optimistic. |
| Table 4.2 — `artifacts/p3_evaluation/per_species_metrics.csv` | Per-species precision, recall, F1, support, and specimen count for the independent and within-specimen cohorts. |
| Table 4.3 — `artifacts/p3_evaluation/model_comparison.csv` | Parameter count and PyTorch/ONNX model size for ShuffleNetV2 and ResNet18. |
| Table 4.4 — `artifacts/p3_evaluation/most_confused_pairs.csv` | The species pairs with the largest combined off-diagonal confusion counts. |
| Supplementary — `artifacts/p3_evaluation/confidence_threshold_curve.csv` | Accepted-image coverage and accuracy at each observed patch-agreement threshold; the retake threshold remains unselected until an accepted-accuracy/coverage target is approved. |
| Supplementary — `artifacts/p3_evaluation/sharpness_percentiles.csv` | Image sharpness percentiles for the 900 images used to train the final models; no blur cutoff is selected without approval. |
| Figure 4.1 — `artifacts/p3_evaluation/Figure_4.1_confusion_shufflenet_v2.png` | Independent-specimen image-level confusion pattern for ShuffleNetV2. |
| Figure 4.2 — `artifacts/p3_evaluation/Figure_4.2_confusion_resnet18.png` | Independent-specimen image-level confusion pattern for ResNet18. |
| Figure 4.3 — `artifacts/p3_evaluation/Figure_4.3_training_curves.png` | Training and validation loss across the three folds for both architectures. |

`evaluation_summary.json`, prediction CSV files, confusion-matrix CSV files,
sharpness measurements, and the verified input-image mapping are supporting
reproducibility outputs. Metrics are computed from held-out test predictions,
not validation predictions. Confidence intervals resample specimens rather
than individual images.

The generated figures are 300 dpi. Times New Roman was not available in the
evaluation environment, so Matplotlib used a fallback font; set the required
font before using the figures in the thesis.
