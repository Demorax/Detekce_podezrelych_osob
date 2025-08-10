import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# --- Configuration ---
FRAMES_DIR   = 'frames_0.5'
SKELETON_DIR = 'skeletons_yolo_11_new'
MODEL_PATH   = 'yolo11x-pose.pt'

# Ensure base output folder exists
os.makedirs(SKELETON_DIR, exist_ok=True)

# Load YOLOv8 Pose model once
tmp_model = YOLO(MODEL_PATH)


def enhance_image(img):
    """
    Enhance image contrast and brightness for better detection.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def remove_duplicate_skeletons(keypoints, threshold=50):
    """
    Remove duplicate detections based on centroid distance threshold.
    """
    if len(keypoints) <= 1:
        return keypoints

    centroids = []
    valid_idx = []
    for i, sk in enumerate(keypoints):
        if not np.isnan(sk).all():
            c = np.nanmean(sk, axis=0)
            if not np.isnan(c).any():
                centroids.append(c)
                valid_idx.append(i)
    if not centroids:
        return keypoints

    centroids = np.array(centroids)
    unique_mask = np.ones(len(centroids), dtype=bool)
    for i in range(len(centroids)):
        if not unique_mask[i]:
            continue
        for j in range(i+1, len(centroids)):
            if unique_mask[j] and np.linalg.norm(centroids[i] - centroids[j]) < threshold:
                unique_mask[j] = False

    unique_inds = [valid_idx[i] for i in range(len(valid_idx)) if unique_mask[i]]
    return keypoints[unique_inds]


def detect_with_multiple_settings(img, model):
    """
    Optimized for speed while maintaining good accuracy
    """
    all_kps = []

    # Only 2 passes - original and one enhanced
    enhanced = enhance_image(img)

    # Just 2 most effective configs
    configs = [
        {'conf': 0.20, 'imgsz': 1024},  # Good balance
        {'conf': 0.25, 'imgsz': 1280},  # Slightly higher for enhanced
    ]

    # Process original with first config
    res = model(img, verbose=False, **configs[0])
    data = res[0].keypoints.data
    if data.numel() > 0:
        kps = data.cpu().numpy()[:, :, :2]
        all_kps.append(kps)

    # Process enhanced with second config
    res = model(enhanced, verbose=False, **configs[1])
    data = res[0].keypoints.data
    if data.numel() > 0:
        kps = data.cpu().numpy()[:, :, :2]
        all_kps.append(kps)

    if not all_kps:
        return np.full((1, 17, 2), np.nan, dtype=np.float32)

    combined = np.vstack(all_kps)
    return remove_duplicate_skeletons(combined, threshold=40)


def process_folder(frames_dir, skeleton_dir, model):
    """
    Process .jpg frames in a single folder, save keypoints, and report stats.
    Returns (total_frames, detected_frames).
    """
    os.makedirs(skeleton_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))

    total_frames = len(files)
    detected_frames = 0
    total_people = 0

    for fname in tqdm(files, desc=f"Folder {os.path.basename(frames_dir)}"):
        path = os.path.join(frames_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        kps = detect_with_multiple_settings(img, model)
        if not np.isnan(kps).all():
            detected_frames += 1
            total_people += len(kps)
        out_path = os.path.join(skeleton_dir, fname.replace('.jpg', '.npy'))
        np.save(out_path, kps)

    rate = 100 * detected_frames / total_frames if total_frames else 0
    avg_people = total_people / total_frames if total_frames else 0
    print(f"{os.path.basename(frames_dir)}: {detected_frames}/{total_frames} detected ({rate:.1f}%), avg people/frame {avg_people:.1f}")
    return total_frames, detected_frames


def process_all(frames_root, skeleton_root, model):
    """
    Process every subfolder under frames_root and aggregate overall stats.
    """
    total_all = 0
    detected_all = 0

    for vid in os.listdir(frames_root):
        in_dir = os.path.join(frames_root, vid)
        if not os.path.isdir(in_dir):
            continue
        out_dir = os.path.join(skeleton_root, vid)
        t, d = process_folder(in_dir, out_dir, model)
        total_all += t
        detected_all += d

    overall = 100 * detected_all / total_all if total_all else 0
    print(f"\nOverall: {detected_all}/{total_all} detected ({overall:.1f}% )")


if __name__ == '__main__':
    # Default: process all subfolders
    #process_all(FRAMES_DIR, SKELETON_DIR, tmp_model)

    # To process a single folder e.g. '001':
    single_in = os.path.join(FRAMES_DIR, '001')
    single_out = os.path.join(SKELETON_DIR, '001')
    process_folder(single_in, single_out, tmp_model)
