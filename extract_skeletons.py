import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Cesty ke složkám
FRAMES_DIR = 'frames_0.5'
SKELETON_DIR = 'skeletons_yolo_11'
os.makedirs(SKELETON_DIR, exist_ok=True)

# Načti YOLOv8 Pose model
model = YOLO("yolo11x-pose.pt")

# Statistika
total_frames = 0
detected_frames = 0

for video_id in os.listdir(FRAMES_DIR):
    vid_dir = os.path.join(FRAMES_DIR, video_id)
    if not os.path.isdir(vid_dir):
        continue

    out_dir = os.path.join(SKELETON_DIR, video_id)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(vid_dir) if f.endswith('.jpg')])
    video_detected = 0

    for fname in tqdm(files, desc=f"Video {video_id}"):
        img_path = os.path.join(vid_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # YOLO inference
        results = model(img, verbose=False)

        if len(results[0].keypoints.data) == 0:
            # Nenalezeny osoby
            keypoints = np.full((1, 17, 2), np.nan, dtype=np.float32)
        else:
            # Získání klíčových bodů všech osob
            people = results[0].keypoints.data.cpu().numpy()  # shape: (num_people, 17, 3)
            keypoints = people[:, :, :2]  # pouze X,Y souřadnice
            video_detected += 1
            detected_frames += 1

        total_frames += 1

        # Ulož výsledky
        out_path = os.path.join(out_dir, fname.replace('.jpg', '.npy'))
        np.save(out_path, keypoints)

    detection_rate = 100 * video_detected / len(files)
    print(f"Video {video_id}: {video_detected}/{len(files)} frames detected ({detection_rate:.1f}%)")

# Celková statistika
overall_rate = 100 * detected_frames / total_frames
print(f"\nOverall detection rate: {detected_frames}/{total_frames} ({overall_rate:.1f}%)")
