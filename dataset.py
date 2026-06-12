import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src import CLASS_TO_IDX, CLASSES, LABEL_REMAP


def get_cnn_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class PostureImageDataset(Dataset):
    """CNN용 이미지 데이터셋."""

    def __init__(self, csv_path: str, img_root: str, split: str = "train"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.img_root = img_root
        self.transform = get_cnn_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["filepath"])
        image = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[LABEL_REMAP.get(row["label"], row["label"])]
        return self.transform(image), label


class PosturePoseDataset(Dataset):
    """Pose+MLP용 키포인트 데이터셋 — 51-dim 1D 벡터."""

    def __init__(self, csv_path: str, split: str = "train"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        feat_cols = [c for c in self.df.columns if c.startswith("kp_")]
        self.features = torch.tensor(
            self.df[feat_cols].values, dtype=torch.float32
        )
        self.labels = torch.tensor(
            [CLASS_TO_IDX[LABEL_REMAP.get(l, l)] for l in self.df["label"]], dtype=torch.long
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PosturePoseSeqDataset(Dataset):
    """Pose+RNN용 키포인트 시퀀스 데이터셋 — (17, 3) 2D 텐서.

    각 keypoint(17개)를 timestep으로, (x, y, conf)를 feature로 사용.
    """

    def __init__(self, csv_path: str, split: str = "train"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        feat_cols = [c for c in self.df.columns if c.startswith("kp_")]
        raw = torch.tensor(self.df[feat_cols].values, dtype=torch.float32)
        # reshape: (N, 51) → (N, 17, 3)
        self.features = raw.view(-1, 17, 3)
        self.labels = torch.tensor(
            [CLASS_TO_IDX[LABEL_REMAP.get(l, l)] for l in self.df["label"]], dtype=torch.long
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
