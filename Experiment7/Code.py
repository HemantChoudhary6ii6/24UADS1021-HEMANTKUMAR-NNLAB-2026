"""
Transfer Learning: Retrain a Pretrained ImageNet Model
for Medical Image Classification

Auto-downloads PneumoniaMNIST (Normal vs Pneumonia).
No file saving — all results shown as inline charts.

Usage:
    python medical_image_classifier.py
    python medical_image_classifier.py --model resnet18 --epochs 10
    python medical_image_classifier.py --model efficientnet_b0 --epochs 20
"""

import os
import copy
import time
import argparse
import platform
import urllib.request

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, precision_recall_curve,
                              average_precision_score)
import seaborn as sns



# 1. Configuration
def get_args():
    parser = argparse.ArgumentParser(description="Medical Image Classifier — Transfer Learning")
    parser.add_argument("--data_dir",         type=str,   default="./data")
    parser.add_argument("--model",             type=str,   default="resnet50",
                        choices=["resnet18", "resnet50", "efficientnet_b0",
                                 "densenet121", "vgg16"])
    parser.add_argument("--epochs",            type=int,   default=10)
    parser.add_argument("--batch_size",        type=int,   default=32)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--img_size",          type=int,   default=224)
    parser.add_argument("--freeze_backbone",   action="store_true")
    parser.add_argument("--use_weighted_loss", action="store_true")
    args, _ = parser.parse_known_args()   # ignore Colab/Jupyter extra args
    return args



# 2. Dataset Download & Loader
DATASET_URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz"
CLASS_NAMES  = ["Normal", "Pneumonia"]


def download_dataset(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, "pneumoniamnist.npz")
    if not os.path.exists(dest):
        print("[DATA] Downloading PneumoniaMNIST ...")
        def _hook(b, bs, tot):
            done = min(b * bs, tot)
            print(f"\r       {done:,} / {tot:,} bytes  ({done/tot*100:.1f}%)",
                  end="", flush=True)
        urllib.request.urlretrieve(DATASET_URL, dest, reporthook=_hook)
        print()
    else:
        print(f"[DATA] Using cached dataset : {dest}")
    return dest


class PneumoniaMNIST(Dataset):
    def __init__(self, npz_path, split, transform=None):
        data           = np.load(npz_path)
        self.images    = data[f"{split}_images"]   # (N,1,28,28) uint8
        self.labels    = data[f"{split}_labels"]   # (N,1)
        self.transform = transform

    def __len__(self):  return len(self.images)

    def __getitem__(self, idx):
        img   = Image.fromarray(self.images[idx][0]).convert("RGB")
        label = int(self.labels[idx].squeeze())
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def build_dataloaders(npz_path, img_size, batch_size):
    train_tf, val_tf = get_transforms(img_size)
    ds = {
        "train": PneumoniaMNIST(npz_path, "train", train_tf),
        "val":   PneumoniaMNIST(npz_path, "val",   val_tf),
        "test":  PneumoniaMNIST(npz_path, "test",  val_tf),
    }
    nw      = 0 if platform.system() == "Windows" else 2
    use_pin = torch.cuda.is_available()
    loaders = {
        s: DataLoader(d, batch_size=batch_size, shuffle=(s == "train"),
                      num_workers=nw, pin_memory=use_pin,
                      persistent_workers=(nw > 0))
        for s, d in ds.items()
    }
    print(f"\n[DATA] Classes : {CLASS_NAMES}")
    for s, d in ds.items():
        print(f"       {s:5s}  : {len(d):,} images")
    return loaders, ds

# 3. Model Builder
def build_model(model_name, num_classes, freeze_backbone):
    print(f"\n[MODEL] Loading pretrained {model_name} ...")
    weights_map = {
        "resnet18":        models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50":        models.ResNet50_Weights.IMAGENET1K_V2,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "densenet121":     models.DenseNet121_Weights.IMAGENET1K_V1,
        "vgg16":           models.VGG16_Weights.IMAGENET1K_V1,
    }
    model = getattr(models, model_name)(weights=weights_map[model_name])
    if freeze_backbone:
        for p in model.parameters(): p.requires_grad = False
        print("       Backbone frozen — only head will train.")

    if model_name.startswith("resnet"):
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 256),
            nn.ReLU(),       nn.Dropout(0.3),
            nn.Linear(256, num_classes))
    elif model_name.startswith("efficientnet"):
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
    elif model_name.startswith("densenet"):
        in_f = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))
    elif model_name.startswith("vgg"):
        in_f = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_f, num_classes)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"       Trainable : {trainable:,} / {total:,} params")
    return model


# 4. Training Loop  (returns full history)
def train_model(model, loaders, criterion, optimizer, scheduler,
                num_epochs, device):
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[AMP]   Mixed precision (FP16) enabled")

    best_weights = copy.deepcopy(model.state_dict())
    best_acc     = 0.0
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr":         [],                  # learning-rate per epoch
    }

    print("\n" + "─" * 70)
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = running_correct = 0

            for inputs, labels in loaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(inputs)
                        loss    = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                running_loss    += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()

            n = len(loaders[phase].dataset)
            history[f"{phase}_loss"].append(running_loss / n)
            history[f"{phase}_acc"].append(running_correct / n)

            if phase == "val":
                scheduler.step()
                history["lr"].append(optimizer.param_groups[0]["lr"])
                if history["val_acc"][-1] > best_acc:
                    best_acc     = history["val_acc"][-1]
                    best_weights = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch:3d}/{num_epochs}]  "
              f"Train loss={history['train_loss'][-1]:.4f}  acc={history['train_acc'][-1]:.4f}  |  "
              f"Val   loss={history['val_loss'][-1]:.4f}  acc={history['val_acc'][-1]:.4f}  "
              f"lr={history['lr'][-1]:.2e}  ({time.time()-t0:.1f}s)")

    print(f"\n[DONE] Best val accuracy : {best_acc:.4f}")
    model.load_state_dict(best_weights)
    return model, history



# 5. Collect predictions + probabilities
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))

# 6. Visualize — everything in one big figure
def visualize_all(history, labels, preds, probs, args, ds_map, npz_path):
    """
    Plots in a single call:
      Row 1 : Training Loss | Training Accuracy | Learning-Rate schedule
      Row 2 : Confusion Matrix | ROC Curve | Precision-Recall Curve
      Row 3 : Per-class Precision / Recall / F1 bar chart
               | Class distribution (train/val/test)
               | Sample augmented images (train batch)
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        f"Medical Image Classification  |  Model: {args.model}  "
        f"|  Epochs: {args.epochs}  |  LR: {args.lr}  "
        f"|  Batch: {args.batch_size}  |  ImgSize: {args.img_size}",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    epochs_range = range(1, len(history["train_loss"]) + 1)
    BLUE, RED, GREEN = "#2563EB", "#DC2626", "#16A34A"

    # 1. Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_range, history["train_loss"], color=BLUE,  marker="o", label="Train")
    ax1.plot(epochs_range, history["val_loss"],   color=RED,   marker="s", label="Val")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.4)

    #2. Training Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_range, [v * 100 for v in history["train_acc"]],
             color=BLUE, marker="o", label="Train")
    ax2.plot(epochs_range, [v * 100 for v in history["val_acc"]],
             color=RED,  marker="s", label="Val")
    ax2.set_title("Training & Validation Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(); ax2.grid(True, alpha=0.4)
    # annotate best val acc
    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_val   = max(history["val_acc"]) * 100
    ax2.annotate(f"Best {best_val:.1f}%",
                 xy=(best_epoch, best_val),
                 xytext=(best_epoch + 0.3, best_val - 6),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=9, color="black")

    # 3. Learning-Rate Schedule
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs_range, history["lr"], color=GREEN, marker="^")
    ax3.set_title("Learning-Rate Schedule (CosineAnnealing)", fontweight="bold")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("LR")
    ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax3.grid(True, alpha=0.4)

    # 4. Confusion Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{v}\n({cm_norm[i,j]*100:.1f}%)"
                       for j, v in enumerate(row)]
                      for i, row in enumerate(cm)])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax4,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, cbar=False)
    ax4.set_title("Confusion Matrix (count + %)", fontweight="bold")
    ax4.set_ylabel("True Label"); ax4.set_xlabel("Predicted Label")

    # 5. ROC Curve
    ax5 = fig.add_subplot(gs[1, 1])
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc     = auc(fpr, tpr)
    ax5.plot(fpr, tpr, color=BLUE, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax5.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax5.fill_between(fpr, tpr, alpha=0.1, color=BLUE)
    ax5.set_title("ROC Curve", fontweight="bold")
    ax5.set_xlabel("False Positive Rate")
    ax5.set_ylabel("True Positive Rate")
    ax5.legend(loc="lower right"); ax5.grid(True, alpha=0.4)

    # 6. Precision-Recall Curve
    ax6 = fig.add_subplot(gs[1, 2])
    precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
    ap = average_precision_score(labels, probs[:, 1])
    ax6.plot(recall, precision, color=RED, lw=2, label=f"AP = {ap:.4f}")
    ax6.fill_between(recall, precision, alpha=0.1, color=RED)
    # baseline
    baseline = labels.sum() / len(labels)
    ax6.axhline(baseline, color="gray", linestyle="--", lw=1,
                label=f"Baseline = {baseline:.2f}")
    ax6.set_title("Precision-Recall Curve", fontweight="bold")
    ax6.set_xlabel("Recall"); ax6.set_ylabel("Precision")
    ax6.legend(); ax6.grid(True, alpha=0.4)

    #7. Per-class Precision / Recall / F1 bar chart
    ax7 = fig.add_subplot(gs[2, 0])
    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics = {
        "Precision": precision_score(labels, preds, average=None),
        "Recall":    recall_score(labels, preds, average=None),
        "F1-Score":  f1_score(labels, preds, average=None),
    }
    x      = np.arange(len(CLASS_NAMES))
    width  = 0.25
    colors = [BLUE, RED, GREEN]
    for i, (metric, vals) in enumerate(metrics.items()):
        bars = ax7.bar(x + i * width, vals, width, label=metric, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax7.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax7.set_title("Per-class Precision / Recall / F1", fontweight="bold")
    ax7.set_xticks(x + width)
    ax7.set_xticklabels(CLASS_NAMES)
    ax7.set_ylim(0, 1.15); ax7.set_ylabel("Score")
    ax7.legend(fontsize=8); ax7.grid(True, alpha=0.4, axis="y")

    #8. Class distribution across splits
    ax8 = fig.add_subplot(gs[2, 1])
    data = np.load(npz_path)
    split_counts = {}
    for split in ["train", "val", "test"]:
        lbl = data[f"{split}_labels"].squeeze()
        split_counts[split] = [int((lbl == c).sum()) for c in range(len(CLASS_NAMES))]

    x2     = np.arange(len(CLASS_NAMES))
    width2 = 0.25
    scols  = ["#6366F1", "#F59E0B", "#10B981"]
    for i, (split, counts) in enumerate(split_counts.items()):
        bars = ax8.bar(x2 + i * width2, counts, width2,
                       label=split.capitalize(), color=scols[i], alpha=0.85)
        for bar, val in zip(bars, counts):
            ax8.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 10,
                     str(val), ha="center", va="bottom", fontsize=8)
    ax8.set_title("Class Distribution per Split", fontweight="bold")
    ax8.set_xticks(x2 + width2)
    ax8.set_xticklabels(CLASS_NAMES)
    ax8.set_ylabel("Image Count")
    ax8.legend(fontsize=8); ax8.grid(True, alpha=0.4, axis="y")

    # 9. Sample training images (augmented) 
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    ax9.set_title("Sample Augmented Training Images", fontweight="bold", y=1.02)

    raw_ds = PneumoniaMNIST(npz_path, "train", transform=None)
    _, val_tf = get_transforms(args.img_size)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    samples_per_class = 3
    inner = gridspec.GridSpecFromSubplotSpec(
        len(CLASS_NAMES), samples_per_class,
        subplot_spec=gs[2, 2], hspace=0.05, wspace=0.05)

    shown = {c: 0 for c in range(len(CLASS_NAMES))}
    idxs  = np.random.permutation(len(raw_ds))
    for idx in idxs:
        img_pil, lbl = raw_ds[idx]
        if shown[lbl] < samples_per_class:
            aug_tf = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
            ])
            img_t  = aug_tf(img_pil).permute(1, 2, 0).numpy()
            img_t  = np.clip(img_t, 0, 1)
            sub_ax = fig.add_subplot(inner[lbl, shown[lbl]])
            sub_ax.imshow(img_t, cmap="gray")
            sub_ax.axis("off")
            if shown[lbl] == 0:
                sub_ax.set_ylabel(CLASS_NAMES[lbl], fontsize=7, rotation=90,
                                  labelpad=2)
                sub_ax.yaxis.set_label_position("left")
                sub_ax.tick_params(left=False, labelleft=False)
            shown[lbl] += 1
        if all(v >= samples_per_class for v in shown.values()):
            break

    plt.show()
    print("\n[CHARTS] All visualizations displayed.")


# 7. Main
def main():
    args = get_args()

    # Resolve data_dir relative to script (works on Windows + Colab)
    try:
        script_dir   = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir   = os.getcwd()
    data_rel     = args.data_dir.lstrip("./").lstrip(".\\")
    args.data_dir = os.path.join(script_dir, data_rel)
    os.makedirs(args.data_dir, exist_ok=True)

    # Dataset
    npz_path        = download_dataset(args.data_dir)
    loaders, ds_map = build_dataloaders(npz_path, args.img_size, args.batch_size)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "none"
    print(f"\n[DEVICE] {device}" +
          (f"  ({gpu_name})" if device.type == "cuda" else "  (no GPU — training will be slow)"))
    if device.type == "cuda" and args.batch_size < 64:
        args.batch_size = 64
        print(f"[DEVICE] batch_size auto-scaled to {args.batch_size} for GPU")

    # Model
    model = build_model(args.model, num_classes=2,
                        freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    # Loss
    if args.use_weighted_loss:
        lbl_arr = np.array([int(ds_map["train"][i][1])
                            for i in range(len(ds_map["train"]))])
        counts  = np.bincount(lbl_arr).astype(np.float32)
        weights = torch.tensor(1.0 / counts)
        weights /= weights.sum()
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        print(f"\n[LOSS] Weighted CE : {dict(zip(CLASS_NAMES, weights.numpy().round(4)))}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Train
    model, history = train_model(
        model, loaders, criterion, optimizer, scheduler,
        args.epochs, device)

    # Collect predictions
    print("\n[EVAL]  Running inference on test set ...")
    labels, preds, probs = collect_predictions(model, loaders["test"], device)

    # Print text report
    print("\n[EVALUATION] Classification Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    # Visualize everything
    visualize_all(history, labels, preds, probs, args, ds_map, npz_path)


if __name__ == "__main__":
    main()