import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score,
)

from src import CLASSES, IDX_TO_CLASS
from src.dataset import PostureImageDataset, PosturePoseDataset, PosturePoseSeqDataset
from src.models import build_resnet18, build_efficientnet_b0, PoseMLP, PoseRNN


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for data, labels in loader:
        data = data.to(device)
        outputs = model(data)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        # non_neutral(idx=1) 확률 = positive class score for ROC-AUC
        all_probs.extend(probs[:, 1].cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def evaluate_cnn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = PostureImageDataset(args.splits_csv, args.img_root, split="test")
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    if args.backbone == "resnet18":
        model = build_resnet18(pretrained=False)
    else:
        model = build_efficientnet_b0(pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    preds, labels, probs = get_predictions(model, loader, device)
    _report(preds, labels, probs, f"CNN ({args.backbone})", f"cnn_{args.backbone}")


def evaluate_pose_mlp(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = PosturePoseDataset(args.pose_csv, split="test")
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    model = PoseMLP(input_dim=51).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    preds, labels, probs = get_predictions(model, loader, device)
    _report(preds, labels, probs, "Pose+MLP", "pose_mlp")


def evaluate_pose_rnn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = PosturePoseSeqDataset(args.pose_csv, split="test")
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    model = PoseRNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    preds, labels, probs = get_predictions(model, loader, device)
    _report(preds, labels, probs, "Pose+RNN", "pose_rnn")


def evaluate_cnn_features(args):
    """frozen ResNet18 특징 캐시 + 학습된 헤드로 평가."""
    from src.train_cnn_features import CNNHead, FEATURE_CACHE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(FEATURE_CACHE)
    X = torch.tensor(d["test_X"], dtype=torch.float32)
    labels = d["test_y"]

    model = CNNHead(in_dim=512).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        out = model(X.to(device))
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(1).cpu().numpy()
    _report(preds, labels, probs, "CNN (ResNet18 frozen+head)", "cnn_resnet18")


def _report(preds, labels, probs, title, tag):
    """클래스 불균형을 고려해 정확도 외에 재현율·F1·balanced accuracy·ROC-AUC를 함께 보고."""
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)

    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    # positive class = non_neutral(idx=1)
    pos_recall = recall_score(labels, preds, pos_label=1)
    pos_precision = precision_score(labels, preds, pos_label=1)
    pos_f1 = f1_score(labels, preds, pos_label=1)
    neg_recall = recall_score(labels, preds, pos_label=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    report = classification_report(labels, preds, target_names=CLASSES, output_dict=True)

    print(f"\n=== {title} ===")
    print(f"Accuracy            : {acc:.4f}")
    print(f"Balanced Accuracy   : {bal_acc:.4f}   (불균형 보정: 클래스별 recall 평균)")
    print(f"Macro F1            : {macro_f1:.4f}   (두 클래스 동등 가중)")
    print(f"Weighted F1         : {weighted_f1:.4f}")
    print(f"ROC-AUC             : {auc:.4f}")
    print(f"non_neutral recall  : {pos_recall:.4f}  precision: {pos_precision:.4f}  F1: {pos_f1:.4f}")
    print(f"neutral     recall  : {neg_recall:.4f}")
    print("\n" + classification_report(labels, preds, target_names=CLASSES, digits=4))

    plot_confusion_matrix(labels, preds, f"Confusion Matrix: {title}",
                          f"results/figures/cm_{tag}.png")

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": auc,
        "non_neutral": {"recall": pos_recall, "precision": pos_precision, "f1": pos_f1},
        "neutral_recall": neg_recall,
        "report": report,
    }
    with open(f"results/metrics/{tag}_report.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: results/metrics/{tag}_report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    cnn_p = sub.add_parser("cnn")
    cnn_p.add_argument("--backbone", default="resnet18")
    cnn_p.add_argument("--model_path", default="models/cnn_resnet18_best.pt")
    cnn_p.add_argument("--splits_csv", default="data/splits/splits.csv")
    cnn_p.add_argument("--img_root", default="data/raw")

    pose_p = sub.add_parser("pose")
    pose_p.add_argument("--model_path", default="models/pose_mlp_best.pt")
    pose_p.add_argument("--pose_csv", default="data/splits/pose_features.csv")

    rnn_p = sub.add_parser("rnn")
    rnn_p.add_argument("--model_path", default="models/pose_rnn_best.pt")
    rnn_p.add_argument("--pose_csv", default="data/splits/pose_features.csv")

    cnnfeat_p = sub.add_parser("cnnfeat")
    cnnfeat_p.add_argument("--model_path", default="models/cnn_resnet18_head_best.pt")

    args = parser.parse_args()
    if args.mode == "cnn":
        evaluate_cnn(args)
    elif args.mode == "pose":
        evaluate_pose_mlp(args)
    elif args.mode == "rnn":
        evaluate_pose_rnn(args)
    elif args.mode == "cnnfeat":
        evaluate_cnn_features(args)
    else:
        parser.print_help()
