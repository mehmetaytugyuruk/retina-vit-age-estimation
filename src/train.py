"""
Training script for retinal fundus age prediction with Vision Transformer backbones.

Reproduces the ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32 experiments from:
  Yuruk, M.A. and Memis, A. "Decoding Chronological Age from the Retinal
  Fundus Images: A Deep Learning-based Analysis with Vision Transformers."
  SIU 2026 (accepted).

Companion study: https://github.com/mehmetaytugyuruk/retina-resnet-age-estimation
(ResNet backbones, same dataset and preprocessing).
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import (
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Per-variant training configuration. ViT-L/16 uses a lower learning rate
# than the other three variants for training stability given its larger size.
MODEL_CONFIG = {
    "vit_b16": {"builder": models.vit_b_16, "weights": ViT_B_16_Weights.IMAGENET1K_V1, "lr": 3e-5, "batch_size": 32},
    "vit_b32": {"builder": models.vit_b_32, "weights": ViT_B_32_Weights.IMAGENET1K_V1, "lr": 3e-5, "batch_size": 32},
    "vit_l16": {"builder": models.vit_l_16, "weights": ViT_L_16_Weights.IMAGENET1K_V1, "lr": 2e-5, "batch_size": 16},
    "vit_l32": {"builder": models.vit_l_32, "weights": ViT_L_32_Weights.IMAGENET1K_V1, "lr": 3e-5, "batch_size": 32},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id, seed):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_lds_weights(df, age_col="patient_age", sigma=2, clamp_max=5.0):
    """Label Distribution Smoothing weights (Yang et al., ICML 2021)."""
    value_counts = df[age_col].value_counts().sort_index()
    min_age = int(df[age_col].min())
    max_age = int(df[age_col].max())

    counts = np.zeros(max_age - min_age + 1, dtype=np.float64)
    for age, count in value_counts.items():
        idx = int(age - min_age)
        if 0 <= idx < len(counts):
            counts[idx] = count

    smoothed_counts = gaussian_filter1d(counts, sigma=sigma)
    weights = 1.0 / (smoothed_counts + 1e-5)
    weights = weights / weights.mean()

    weight_dict = {age: float(weights[int(age - min_age)]) for age in range(min_age, max_age + 1)}
    if clamp_max is not None:
        weight_dict = {k: min(v, clamp_max) for k, v in weight_dict.items()}
    return weight_dict


class FundusAgeDataset(Dataset):
    def __init__(self, df, transform, mean_age, std_age, weights_dict):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mean_age = float(mean_age)
        self.std_age = float(std_age)
        self.weights_dict = weights_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        img = self.transform(img)

        age = float(row["patient_age"])
        norm_age = (age - self.mean_age) / (self.std_age + 1e-8)

        if self.weights_dict is None:
            w = 1.0
        else:
            w = float(self.weights_dict.get(int(round(age)), 1.0))

        return (
            img,
            torch.tensor(norm_age, dtype=torch.float32),
            torch.tensor(age, dtype=torch.float32),
            torch.tensor(w, dtype=torch.float32),
        )


def build_model(model_name: str) -> nn.Module:
    cfg = MODEL_CONFIG[model_name]
    model = cfg["builder"](weights=cfg["weights"])
    in_f = model.heads.head.in_features
    model.heads.head = nn.Linear(in_f, 1)
    return model


def set_requires_grad(model, trainable: bool):
    for p in model.parameters():
        p.requires_grad = trainable


def vit_head_only(model):
    set_requires_grad(model, False)
    for p in model.heads.parameters():
        p.requires_grad = True


def make_param_groups(model: nn.Module, base_lr: float, head_lr_mult: float = 10.0):
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if name.startswith("heads.") else backbone_params).append(p)
    return [
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params, "lr": base_lr * head_lr_mult},
    ]


def denorm(pred_norm, mean_age, std_age) -> torch.Tensor:
    return pred_norm * std_age + mean_age


def train_one_epoch(model, loader, optimizer, criterion, mean_age, std_age, device):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for images, norm_ages, ages, weights in loader:
        images = images.to(device)
        norm_ages = norm_ages.to(device)
        ages = ages.to(device)
        weights = weights.to(device)

        out = model(images).view(-1)
        loss_per = criterion(out, norm_ages)
        loss = (loss_per * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pred_age = denorm(out.detach().cpu(), mean_age, std_age)
        preds_all.extend(pred_age.numpy().tolist())
        labels_all.extend(ages.detach().cpu().numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)
    mae = mean_absolute_error(labels_all, preds_all)
    return avg_loss, mae


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, mean_age, std_age, device):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for images, norm_ages, ages, _ in loader:
        images = images.to(device)
        norm_ages = norm_ages.to(device)
        ages = ages.to(device)

        out = model(images).view(-1)
        loss_per = criterion(out, norm_ages)
        loss = loss_per.mean()

        running_loss += loss.item() * images.size(0)
        pred_age = denorm(out.detach().cpu(), mean_age, std_age)
        preds_all.extend(pred_age.numpy().tolist())
        labels_all.extend(ages.detach().cpu().numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)
    mae = mean_absolute_error(labels_all, preds_all)
    return avg_loss, mae


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODEL_CONFIG))
    p.add_argument("--train-csv", default="csvFiles/train/trainFilteredImages.csv")
    p.add_argument("--val-csv", default="csvFiles/validation/validationFilteredImages.csv")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=None, help="Defaults to the per-model batch size used in the paper.")
    p.add_argument("--lr", type=float, default=None, help="Defaults to the per-model LR used in the paper (see MODEL_CONFIG).")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5, help="Epochs to train only the regression head before unfreezing the backbone.")
    p.add_argument("--no-warmup", action="store_true", help="Disable the head-only warmup phase and fine-tune the full model from epoch 1.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MODEL_CONFIG[args.model]
    lr = args.lr if args.lr is not None else cfg["lr"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    warmup_active = not args.no_warmup

    set_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    mean_age = train_df["patient_age"].mean()
    std_age = train_df["patient_age"].std()

    print(f"\nDevice: {DEVICE}")
    print(f"Train={len(train_df)}  Val={len(val_df)}")
    print(f"Age mean={mean_age:.2f} std={std_age:.2f} range=[{train_df['patient_age'].min()}, {train_df['patient_age'].max()}]")

    print("Computing LDS weights...")
    weights_dict = get_lds_weights(train_df, age_col="patient_age", sigma=2)
    print("LDS weights ready.")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = FundusAgeDataset(train_df, train_transform, mean_age, std_age, weights_dict=weights_dict)
    val_ds = FundusAgeDataset(val_df, eval_transform, mean_age, std_age, weights_dict=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                               worker_init_fn=lambda wid: seed_worker(wid, args.seed), generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                             worker_init_fn=lambda wid: seed_worker(wid, args.seed), generator=g)

    print(f"\nBuilding model: {args.model}  (lr={lr}, batch_size={batch_size}, warmup={warmup_active})")
    model = build_model(args.model).to(DEVICE)

    criterion = nn.SmoothL1Loss(beta=1.0, reduction="none")
    wd = args.weight_decay
    HEAD_LR = 3e-4

    if warmup_active:
        vit_head_only(model)
        optimizer = torch.optim.AdamW([{"params": model.heads.parameters(), "lr": HEAD_LR}], weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7)
    else:
        param_groups = make_param_groups(model, base_lr=lr, head_lr_mult=10.0)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7)

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    best_val_mae = float("inf")
    best_epoch = -1
    ckpt_path = os.path.join(args.output_dir, f"best_{args.model}.pth")

    print(f"\nStart training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}\n{'='*60}")

        if warmup_active and epoch == args.warmup_epochs + 1:
            print("\nUnfreezing full ViT for fine-tuning...")
            set_requires_grad(model, True)
            param_groups = make_param_groups(model, base_lr=lr, head_lr_mult=10.0)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7)

        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, criterion, mean_age, std_age, DEVICE)
        val_loss, val_mae = eval_one_epoch(model, val_loader, criterion, mean_age, std_age, DEVICE)

        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        print(f"Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}y (LDS)")
        print(f"Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}y")
        print(f"LR: {current_lr:.2e}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            print(f"New best val MAE: {best_val_mae:.2f} at epoch {best_epoch}. Saving checkpoint...")
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "mean_age": float(mean_age),
                "std_age": float(std_age),
                "history": history,
                "config": {
                    "BATCH_SIZE": batch_size,
                    "EPOCHS": args.epochs,
                    "LR": lr,
                    "WEIGHT_DECAY": wd,
                    "WARMUP_EPOCHS": args.warmup_epochs if warmup_active else 0,
                    "SEED": args.seed,
                },
            }, ckpt_path)
        else:
            print(f"Val MAE ({val_mae:.2f}) did not improve on best ({best_val_mae:.2f}).")

    torch.save(history, os.path.join(args.output_dir, f"full_training_history_{args.model}.pt"))
    print(f"\nTraining finished. Best val MAE: {best_val_mae:.2f} at epoch {best_epoch}")
    print(f"Best checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
