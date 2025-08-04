import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Configure these base paths
FRAMES_DIR = 'frames_0.5'
SKELETON_DIR = 'skeletons_yolo_11_new'
MODEL_PATH = 'yolo11x-pose.pt'

# Initialize model once
model = YOLO(MODEL_PATH)


def process_folder(frames_dir: str, skeleton_dir: str, model) -> tuple:
    os.makedirs(skeleton_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))

    total_frames = 0
    detected_frames = 0

    for fname in tqdm(files, desc=f"Processing {os.path.basename(frames_dir)}"):
        img_path = os.path.join(frames_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model(img, verbose=False)
        kp_data = results[0].keypoints.data

        if kp_data.numel() == 0:
            # No persons detected: fill with NaNs
            keypoints = np.full((1, 17, 2), np.nan, dtype=np.float32)
        else:
            people = kp_data.cpu().numpy()  # shape: (num_people, 17, 3)
            keypoints = people[:, :, :2]    # only X,Y coords
            detected_frames += 1

        total_frames += 1
        out_path = os.path.join(skeleton_dir, fname.replace('.jpg', '.npy'))
        np.save(out_path, keypoints)

    if total_frames > 0:
        rate = 100 * detected_frames / total_frames
        print(f"Folder {os.path.basename(frames_dir)}: {detected_frames}/{total_frames} frames detected ({rate:.1f}%)")
    else:
        print(f"Folder {os.path.basename(frames_dir)}: no .jpg files found.")

    return total_frames, detected_frames


def process_all(frames_root: str, skeleton_root: str, model) -> None:

    os.makedirs(skeleton_root, exist_ok=True)

    total_all = 0
    detected_all = 0

    for video_id in os.listdir(frames_root):
        in_dir = os.path.join(frames_root, video_id)
        if not os.path.isdir(in_dir):
            continue

        out_dir = os.path.join(skeleton_root, video_id)
        t, d = process_folder(in_dir, out_dir, model)
        total_all += t
        detected_all += d

    if total_all > 0:
        overall_rate = 100 * detected_all / total_all
        print(f"\nOverall detection rate: {detected_all}/{total_all} ({overall_rate:.1f}%)")
    else:
        print("\nNo folders processed.")


if __name__ == '__main__':
    # Pro vse 001, 002
    #process_all(FRAMES_DIR, SKELETON_DIR, model)

    # Pouze pro 001
    specific_in = os.path.join(FRAMES_DIR, '001')
    specific_out = os.path.join(SKELETON_DIR, '001')
    process_folder(specific_in, specific_out, model)
