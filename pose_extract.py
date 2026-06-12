"""
YOLOv8-pose 기반 17 keypoint 추출 + 어깨 정규화.
(mediapipe DLL 호환성 문제로 ultralytics YOLOv8-pose로 변경)

17개 COCO keypoints:
  0:nose, 1:l_eye, 2:r_eye, 3:l_ear, 4:r_ear
  5:l_shoulder, 6:r_shoulder
  7:l_elbow, 8:r_elbow
  9:l_wrist, 10:r_wrist
  11:l_hip, 12:r_hip
  13:l_knee, 14:r_knee
  15:l_ankle, 16:r_ankle

특징 벡터: 17 keypoints × 3 (x, y, conf) = 51 dim
"""
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

_MODEL = None
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO("yolov8n-pose.pt")
    return _MODEL


def extract_keypoints(image_bgr: np.ndarray) -> np.ndarray | None:
    """BGR 이미지에서 가장 큰 사람의 17 keypoint (x,y,conf) 추출. 실패 시 None."""
    model = get_model()
    results = model(image_bgr, verbose=False)
    if not results or len(results) == 0:
        return None
    r = results[0]
    if r.keypoints is None or len(r.keypoints) == 0:
        return None

    # 가장 큰 박스의 사람 선택
    if r.boxes is not None and len(r.boxes) > 0:
        areas = (r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])
        idx = int(areas.argmax())
    else:
        idx = 0

    kps_xy = r.keypoints.xyn[idx].cpu().numpy()  # (17, 2) normalized 0~1
    kps_conf = r.keypoints.conf[idx].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)
    return np.concatenate([kps_xy, kps_conf[:, None]], axis=1)  # (17, 3)


def normalize_keypoints(kps: np.ndarray) -> np.ndarray:
    """어깨 중점 원점 + 어깨너비 스케일 정규화 후 1D 벡터(51) 반환."""
    mid = (kps[LEFT_SHOULDER, :2] + kps[RIGHT_SHOULDER, :2]) / 2.0
    shoulder_width = np.linalg.norm(kps[LEFT_SHOULDER, :2] - kps[RIGHT_SHOULDER, :2])
    if shoulder_width < 1e-6:
        shoulder_width = 1.0
    normalized = kps.copy()
    normalized[:, 0] = (kps[:, 0] - mid[0]) / shoulder_width
    normalized[:, 1] = (kps[:, 1] - mid[1]) / shoulder_width
    return normalized.flatten()  # 51


def process_image_dir(img_root: str, output_csv: str, splits_csv: str) -> None:
    df_splits = pd.read_csv(splits_csv)
    rows = []
    failed = 0

    for _, row in tqdm(df_splits.iterrows(), total=len(df_splits), desc="Pose extraction"):
        img_path = Path(img_root) / row["filepath"]
        img = cv2.imread(str(img_path))
        if img is None:
            failed += 1
            continue
        kps = extract_keypoints(img)
        if kps is None:
            failed += 1
            continue
        vec = normalize_keypoints(kps)
        entry = {"filepath": row["filepath"], "label": row["label"], "split": row["split"]}
        for i, v in enumerate(vec):
            entry[f"kp_{i}"] = float(v)
        rows.append(entry)

    print(f"Extracted: {len(rows)}, Failed: {failed}")
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", default="data/raw")
    parser.add_argument("--splits_csv", default="data/splits/splits.csv")
    parser.add_argument("--output_csv", default="data/splits/pose_features.csv")
    args = parser.parse_args()
    process_image_dir(args.img_root, args.output_csv, args.splits_csv)
