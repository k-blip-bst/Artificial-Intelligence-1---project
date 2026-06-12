"""
binary 분류(neutral vs non_neutral) CNN / Pose+MLP / Pose+RNN 학습을 순서대로 실행.

실행:
  python scripts/train_all_binary.py
"""
import subprocess
import sys


def run(cmd):
    print(f"\n{'='*60}\n▶ {' '.join(cmd)}\n{'='*60}")
    subprocess.run([sys.executable] + cmd, check=True)


if __name__ == "__main__":
    # 1. splits CSV 재생성 (binary labels)
    run(["scripts/build_splits.py"])

    # 2. CNN (ResNet18)
    run([
        "-m", "src.train_cnn",
        "--backbone", "resnet18",
        "--epochs", "30",
        "--batch_size", "32",
        "--lr", "1e-4",
        "--splits_csv", "data/splits/splits.csv",
        "--img_root", "data/raw",
    ])

    # 3. Pose + MLP
    run([
        "-m", "src.train_pose_mlp",
        "--epochs", "100",
        "--batch_size", "64",
        "--lr", "3e-4",
        "--pose_csv", "data/splits/pose_features.csv",
    ])

    # 4. Pose + RNN (Bidirectional LSTM)
    run([
        "-m", "src.train_pose_rnn",
        "--epochs", "100",
        "--batch_size", "64",
        "--lr", "3e-4",
        "--pose_csv", "data/splits/pose_features.csv",
    ])

    print("\nAll training complete. Run evaluate.py to see test results.")
