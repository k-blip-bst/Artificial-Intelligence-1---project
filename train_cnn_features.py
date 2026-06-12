"""
CPU 환경용 CNN 학습 — ResNet18을 '고정(frozen) 특징 추출기'로 사용하는 전이학습.

전체 fine-tuning은 CPU에서 1 epoch에 10분 이상 걸려 비현실적이므로,
  1) ImageNet 사전학습 ResNet18(마지막 fc 제거)로 모든 이미지를 '한 번만' 통과 → 512차원 특징 캐싱
  2) 캐싱된 특징 위에서 가벼운 분류 헤드를 빠르게 학습 (100 epoch이 수 초)
하는 방식으로 CNN 기반 분류 결과를 얻는다. (linear probing / frozen transfer learning)

실행:
  python -m src.train_cnn_features --epochs 200
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import NUM_CLASSES
from src.dataset import PostureImageDataset, get_cnn_transforms

FEATURE_CACHE = "data/splits/cnn_features.npz"


def build_feature_extractor(device):
    """ImageNet 사전학습 ResNet18에서 마지막 fc를 제거한 512-d 특징 추출기."""
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_split(extractor, splits_csv, img_root, split, device, batch_size=32, num_workers=6):
    ds = PostureImageDataset(splits_csv, img_root, split=split)
    # 특징 캐싱은 결정적이어야 하므로 train도 증강 없는 eval transform 사용
    ds.transform = get_cnn_transforms("test")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, persistent_workers=num_workers > 0)
    feats, labels = [], []
    for imgs, lbls in tqdm(loader, desc=f"  extract {split}", leave=False):
        out = extractor(imgs.to(device))  # (B, 512)
        feats.append(out.cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def get_features(args, device):
    """특징 캐시가 있으면 로드, 없으면 추출 후 저장."""
    if os.path.exists(FEATURE_CACHE) and not args.refresh:
        print(f"Loading cached features: {FEATURE_CACHE}")
        d = np.load(FEATURE_CACHE)
        return {s: (d[f"{s}_X"], d[f"{s}_y"]) for s in ("train", "val", "test")}

    print("Extracting frozen ResNet18 features (한 번만 수행)...")
    extractor = build_feature_extractor(device)
    data = {}
    for split in ("train", "val", "test"):
        X, y = extract_split(extractor, args.splits_csv, args.img_root, split, device,
                             num_workers=args.num_workers)
        data[split] = (X, y)
        print(f"  {split}: X={X.shape}, y={y.shape}")
    np.savez(FEATURE_CACHE,
             **{f"{s}_X": data[s][0] for s in data},
             **{f"{s}_y": data[s][1] for s in data})
    print(f"Saved feature cache: {FEATURE_CACHE}")
    return data


class CNNHead(nn.Module):
    """512차원 ResNet 특징 → 2 클래스 분류 헤드."""

    def __init__(self, in_dim=512, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = get_features(args, device)
    loaders = {}
    for split in ("train", "val"):
        X, y = data[split]
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
        loaders[split] = DataLoader(ds, batch_size=args.batch_size,
                                    shuffle=(split == "train"))

    model = CNNHead(in_dim=512, dropout=args.dropout).to(device)

    # 클래스 불균형(neutral 소수) 보정: class weight = 1/클래스빈도
    _, ytr = data["train"]
    counts = np.bincount(ytr, minlength=NUM_CLASSES)
    weights = torch.tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float32, device=device)
    print(f"Class counts (train): {counts.tolist()}  → loss weights: {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, optimizer, device, False)
        scheduler.step()
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Train {tr_loss:.4f}/{tr_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/cnn_resnet18_head_best.pt")

    torch.save(model.state_dict(), "models/cnn_resnet18_head_last.pt")
    _plot_history(history)
    with open("results/metrics/cnn_resnet18_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest Val Acc: {best_val_acc:.4f}")


def _plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("results/figures/cnn_resnet18_curves.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--refresh", action="store_true", help="특징 캐시 무시하고 재추출")
    parser.add_argument("--splits_csv", default="data/splits/splits.csv")
    parser.add_argument("--img_root", default="data/raw")
    args = parser.parse_args()
    main(args)
