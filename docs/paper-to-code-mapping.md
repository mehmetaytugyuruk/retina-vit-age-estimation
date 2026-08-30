# Paper-to-code mapping

| Paper section | Code |
|---|---|
| II. Dataset (train/val/test split) | `csvFiles/{train,validation,test}/trainFilteredImages.csv` etc. — as provided by the dataset authors, patient-level split, Graham-filtered variant only |
| III-A. Image Preprocessing | `src/preprocessing/crop_and_resize.py`, `src/preprocessing/ben_graham_filter.py` |
| III-B. Model Architecture (ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32) | `src/train.py`, `build_model()`, `MODEL_CONFIG` |
| III-C. Model Training | `src/train.py`, `main()` — LDS weighting, AdamW, Smooth L1, ReduceLROnPlateau, 80 epochs |
| III-D. Chronological Age Prediction and Categorization | `src/evaluate.py`, `age_to_class()` |
| III-E. Evaluation Metrics | `src/evaluate.py`, `compute_per_class_table()` (Table IV) and `compute_overall_table()` (Table V) |
| IV. Experimental Results, Tables IV-V | Reproduced by running `src/train.py` then `src/evaluate.py` for each of the 4 ViT models |
| IV, Table VI (ResNet comparison) | See the [ResNet companion study](https://github.com/mehmetaytugyuruk/retina-resnet-age-prediction) |

## Training configuration

Per-model learning rate and batch size are set in `MODEL_CONFIG` in `src/train.py`: ViT-B/16, ViT-B/32, and ViT-L/32 use LR 3e-5; ViT-L/16 uses LR 2e-5. All four variants use a 5-epoch head-only warmup before unfreezing the full backbone (`--no-warmup` to disable).

## Metric definitions

- MAE: mean absolute error between predicted and true chronological age.
- Accuracy (overall, Table V): the test-set-support-weighted mean of the five per-class one-vs-rest accuracies (see `compute_overall_table()` in `evaluate.py`).
- Precision / Recall / F1 (overall): standard scikit-learn weighted multiclass averages over the five age categories.

## Note on the ResNet comparison (Table VI)

Table VI in this paper reports the companion study's ResNet results, reproduced from the published IISEC 2026 paper (not re-run for this paper).
