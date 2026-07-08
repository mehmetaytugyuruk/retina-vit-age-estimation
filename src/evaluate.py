"""
Evaluation script for retinal fundus age prediction with Vision Transformer
backbones.

Loads a trained checkpoint, runs inference on a test CSV, and reports both
per-age-category metrics (Table IV style) and overall weighted metrics
(Table V style), matching the tables reported in the paper. Also renders
a PNG summary table.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

AGE_CLASSES = {
    0: {"range": (0, 17), "label": "Pediatric (0-17)"},
    1: {"range": (18, 39), "label": "Young Adult (18-39)"},
    2: {"range": (40, 59), "label": "Middle Age (40-59)"},
    3: {"range": (60, 74), "label": "Senior (60-74)"},
    4: {"range": (75, 200), "label": "Elderly (75+)"},
}
NUM_CLASSES = len(AGE_CLASSES)

MODEL_BUILDERS = {
    "vit_b16": models.vit_b_16,
    "vit_b32": models.vit_b_32,
    "vit_l16": models.vit_l_16,
    "vit_l32": models.vit_l_32,
}


def age_to_class(age):
    for cls, info in AGE_CLASSES.items():
        lo, hi = info["range"]
        if lo <= age <= hi:
            return cls
    return 4


class FundusDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        img = self.transform(img)
        age = float(row["patient_age"])
        return img, age


def build_model(model_name: str) -> nn.Module:
    model = MODEL_BUILDERS[model_name](weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 1)
    return model


def compute_per_class_table(labels, preds):
    """Per-class metrics using a One-vs-Rest approach (paper's Table IV style)."""
    labels = np.array(labels)
    preds = np.array(preds)
    N = len(labels)

    true_cls = np.array([age_to_class(a) for a in labels])
    pred_cls = np.array([age_to_class(p) for p in preds])

    rows = []
    for c in range(NUM_CLASSES):
        mask = true_cls == c
        mae_c = mean_absolute_error(labels[mask], preds[mask]) if mask.sum() > 0 else float("nan")

        true_bin = (true_cls == c).astype(int)
        pred_bin = (pred_cls == c).astype(int)
        tp = int(((true_bin == 1) & (pred_bin == 1)).sum())
        tn = int(((true_bin == 0) & (pred_bin == 0)).sum())
        acc_c = (tp + tn) / N

        rows.append({
            "label": AGE_CLASSES[c]["label"],
            "MAE": round(mae_c, 4),
            "Accuracy": round(acc_c, 4),
            "Precision": round(precision_score(true_bin, pred_bin, zero_division=0), 4),
            "Recall": round(recall_score(true_bin, pred_bin, zero_division=0), 4),
            "F1-Score": round(f1_score(true_bin, pred_bin, zero_division=0), 4),
        })

    avg_row = {"label": "Average (weighted, mu)"}
    for key in ["MAE", "Accuracy", "Precision", "Recall", "F1-Score"]:
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
        avg_row[key] = round(float(np.mean(vals)), 4)
    rows.append(avg_row)
    return rows


def compute_overall_table(labels, preds):
    """
    Overall metrics across the full test set (paper's Table V style).

    Accuracy is the test-set-support-weighted mean of the per-class
    one-vs-rest accuracies. Precision/Recall/F1 use scikit-learn's
    weighted multiclass average.
    """
    labels = np.array(labels)
    preds = np.array(preds)
    N = len(labels)

    true_cls = np.array([age_to_class(a) for a in labels])
    pred_cls = np.array([age_to_class(p) for p in preds])

    weighted_accuracy = 0.0
    for c in range(NUM_CLASSES):
        true_bin = (true_cls == c).astype(int)
        pred_bin = (pred_cls == c).astype(int)
        support_c = int(true_bin.sum())
        if support_c == 0:
            continue
        tp = int(((true_bin == 1) & (pred_bin == 1)).sum())
        tn = int(((true_bin == 0) & (pred_bin == 0)).sum())
        acc_c = (tp + tn) / N
        weighted_accuracy += acc_c * (support_c / N)

    return {
        "MAE": round(mean_absolute_error(labels, preds), 4),
        "Accuracy": round(weighted_accuracy, 4),
        "Precision": round(precision_score(true_cls, pred_cls, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(true_cls, pred_cls, average="weighted", zero_division=0), 4),
        "F1-Score": round(f1_score(true_cls, pred_cls, average="weighted", zero_division=0), 4),
    }


def save_summary_png(per_class_rows, overall_metrics, model_name, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    col_labels = ["Age Category", "MAE", "Accuracy", "Precision", "Recall", "F1-Score"]
    col_x = [0.0, 3.4, 4.5, 5.4, 6.3, 7.2]
    total_w = 8.2
    row_h, hdr_h, top_pad, bot_pad = 0.42, 0.50, 0.55, 0.30

    n_rows = len(per_class_rows)
    fig_h = top_pad + hdr_h + n_rows * row_h + bot_pad
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    def cell_txt(x, y_bot, h, s, size=10, weight="normal"):
        ax.text(x + 0.12, y_bot + h / 2, s, fontsize=size, ha="left", va="center",
                 fontweight=weight, color="black")

    hdr_y = fig_h - top_pad - hdr_h
    ax.add_patch(mpatches.Rectangle((0, hdr_y), total_w, hdr_h, linewidth=0, facecolor="#f0f0f0"))
    for ci, label in enumerate(col_labels):
        cell_txt(col_x[ci], hdr_y, hdr_h, label, weight="bold")

    for i, row in enumerate(per_class_rows):
        y = hdr_y - (i + 1) * row_h
        is_avg = row["label"].startswith("Average")
        ax.add_patch(mpatches.Rectangle((0, y), total_w, row_h, linewidth=0,
                                          facecolor="#f8f8f8" if is_avg else "white"))
        vals = [row["label"], str(row["MAE"]), str(row["Accuracy"]),
                str(row["Precision"]), str(row["Recall"]), str(row["F1-Score"])]
        for ci, v in enumerate(vals):
            cell_txt(col_x[ci], y, row_h, v, weight="bold" if is_avg else "normal")

    ax.text(0, fig_h - top_pad + 0.08, f"Evaluation Results -- {model_name.upper()}",
             fontsize=11, ha="left", va="bottom", fontweight="bold")

    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Summary PNG saved -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODEL_BUILDERS))
    p.add_argument("--checkpoint", required=True, help="Path to the .pth checkpoint to evaluate.")
    p.add_argument("--test-csv", default="csvFiles/test/testFilteredImages.csv")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-png", default=None, help="Where to save the summary PNG (defaults next to the checkpoint).")
    return p.parse_args()


def main():
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model = build_model(args.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    mean_age = checkpoint["mean_age"]
    std_age = checkpoint["std_age"]

    df_test = pd.read_csv(args.test_csv)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_loader = DataLoader(FundusDataset(df_test, transform), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    preds, labels = [], []
    with torch.no_grad():
        for images, ages in test_loader:
            images = images.to(DEVICE)
            outputs = model(images).view(-1)
            pred_age = outputs.cpu().numpy() * std_age + mean_age
            preds.extend(pred_age.tolist())
            labels.extend(ages.numpy().tolist())

    per_class_rows = compute_per_class_table(labels, preds)
    overall_metrics = compute_overall_table(labels, preds)

    print(f"\n===== Per-class results ({args.model}) =====")
    for r in per_class_rows:
        print(f"  {r['label']:<24}  MAE={r['MAE']}  Acc={r['Accuracy']}  "
              f"P={r['Precision']}  R={r['Recall']}  F1={r['F1-Score']}")

    print(f"\n===== Overall results ({args.model}) =====")
    for k, v in overall_metrics.items():
        print(f"  {k:<12}: {v}")

    out_png = args.out_png or os.path.join(os.path.dirname(args.checkpoint), f"evaluation_{args.model}.png")
    save_summary_png(per_class_rows, overall_metrics, args.model, out_png)


if __name__ == "__main__":
    main()
