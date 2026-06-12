"""
data/raw/<class>/ 이미지를 스캔해 train/val/test split CSV를 생성한다.
binary 분류: neutral vs non_neutral (forward_head + slouching 합산)
파일 기반 무작위 분할 (70/15/15).

실행:
  python scripts/build_splits.py
"""

import random
import argparse
from pathlib import Path
import pandas as pd

# 원본 폴더명 → binary label 매핑
LABEL_REMAP = {
    "neutral": "neutral",
    "forward_head": "non_neutral",
    "slouching": "non_neutral",
}
RAW_CLASSES = list(LABEL_REMAP.keys())
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_RATIO = (0.70, 0.15, 0.15)
SEED = 42

# 제외할 파일 prefix:
#   har_sitting_* = HuggingFace Human_Action_Recognition의 'sitting' 행동 이미지.
#   'sitting'은 행동 라벨이지 자세 품질(good/bad) 라벨이 아니므로 neutral 근거가 약함 → 제외.
EXCLUDE_PREFIXES = ("har_sitting_",)


def main(args):
    raw_root = Path(args.raw_root)
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    random.seed(SEED)

    for cls in RAW_CLASSES:
        cls_dir = raw_root / cls
        if not cls_dir.exists():
            print(f"  [WARNING] {cls_dir} 없음, 건너뜁니다.")
            continue
        images = [
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
            and not p.name.startswith(EXCLUDE_PREFIXES)
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLIT_RATIO[0])
        n_val = int(n * SPLIT_RATIO[1])

        binary_label = LABEL_REMAP[cls]
        splits = (
            [("train", p) for p in images[:n_train]]
            + [("val", p) for p in images[n_train:n_train + n_val]]
            + [("test", p) for p in images[n_train + n_val:]]
        )
        for split, p in splits:
            rows.append({
                "filepath": str(p.relative_to(raw_root)),
                "label": binary_label,
                "split": split,
            })
        print(f"  {cls} → {binary_label}: {n}장 (train={n_train}, val={n_val}, test={n - n_train - n_val})")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv} ({len(df)} rows)")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="data/raw")
    parser.add_argument("--output", default="data/splits/splits.csv")
    args = parser.parse_args()
    main(args)
