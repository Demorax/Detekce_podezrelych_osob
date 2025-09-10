import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import time

# --- Configuration ---
FRAMES_DIR = 'frames_0.5'
SKELETON_DIR = 'skeletons_yolo_11_new'
MODEL_PATH = 'yolo11x-pose.pt'
SUPER_RESOLUTION_MODEL_PATH = 'models/super_resolution'

# Ensure base output folder exists
os.makedirs(SKELETON_DIR, exist_ok=True)

# Load YOLOv8 Pose model once
tmp_model = YOLO(MODEL_PATH)

# --- Super-resolution setup ---
_SR_NET = None
_SR_SCALE = 4
_SR_PB = os.path.join(SUPER_RESOLUTION_MODEL_PATH, "ESPCN_x4.pb")


def init_super_resolution():
    """
    Initialize super-resolution model with proper error handling
    """
    global _SR_NET

    if _SR_NET is not None:
        return _SR_NET

    try:
        # Check if dnn_superres is available
        if hasattr(cv2, 'dnn_superres'):
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
        else:
            print("dnn_superres not available in your OpenCV build")
            return None

        # Check if model file exists
        if not os.path.exists(_SR_PB):
            print(f"Super-resolution model not found at {_SR_PB}")
            print("Download ESPCN_x4.pb from: https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/ESPCN_x4.pb")
            return None

        # Load model
        sr.readModel(_SR_PB)
        sr.setModel('espcn', _SR_SCALE)

        # Set backend (CPU for compatibility)
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        print("Super-resolution model initialized successfully (CPU backend)")
        _SR_NET = sr
        return sr

    except Exception as e:
        print(f"Failed to initialize super-resolution: {e}")
        return None


def apply_super_resolution(img, max_size=1500):
    """
    Apply super-resolution to image with size limits for performance
    """
    sr = init_super_resolution()
    if sr is None:
        return img

    try:
        h, w = img.shape[:2]

        # Don't super-resolve very large images (too slow)
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img, (new_w, new_h))

            # Apply super-resolution to smaller image
            sr_result = sr.upsample(img_resized)

            # Resize back to original size
            final_result = cv2.resize(sr_result, (w, h))
            return final_result
        else:
            # Apply super-resolution directly
            return sr.upsample(img)

    except Exception as e:
        print(f"Super-resolution failed: {e}")
        return img


def enhance_image(img):
    """
    Enhanced image processing with optional super-resolution
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def enhance_image_with_sr(img):
    """
    Enhanced image processing pipeline including super-resolution
    """
    enhanced_imgs = []

    # 1. Original image
    enhanced_imgs.append(img)

    # 2. Super-resolution enhanced (if available)
    sr_enhanced = apply_super_resolution(img)
    if not np.array_equal(sr_enhanced, img):  # Check if SR actually did something
        enhanced_imgs.append(sr_enhanced)
        print("Applied super-resolution")

    # 3. CLAHE enhanced
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced1 = cv2.merge([l, a, b])
    enhanced1 = cv2.cvtColor(enhanced1, cv2.COLOR_LAB2BGR)
    enhanced_imgs.append(enhanced1)

    # 4. Histogram equalization
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    enhanced2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)  # Fixed: BGR not RGB
    enhanced_imgs.append(enhanced2)

    # 5. Gamma correction for dark areas
    gamma = 0.7
    enhanced3 = np.power(img / 255.0, gamma) * 255.0
    enhanced3 = enhanced3.astype(np.uint8)
    enhanced_imgs.append(enhanced3)

    # 6. Adaptive CLAHE with smaller tiles
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_adaptive = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    l = clahe_adaptive.apply(l)
    enhanced4 = cv2.merge([l, a, b])
    enhanced4 = cv2.cvtColor(enhanced4, cv2.COLOR_LAB2BGR)
    enhanced_imgs.append(enhanced4)

    # 7. Unsharp masking for better edge definition
    gaussian_3 = cv2.GaussianBlur(img, (0, 0), 2.0)
    enhanced5 = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0)
    enhanced_imgs.append(enhanced5)

    # 8. If we have SR, also apply CLAHE to SR result
    if len(enhanced_imgs) > 1 and not np.array_equal(enhanced_imgs[1], img):
        sr_img = enhanced_imgs[1]
        lab_sr = cv2.cvtColor(sr_img, cv2.COLOR_BGR2LAB)
        l_sr, a_sr, b_sr = cv2.split(lab_sr)
        clahe_sr = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6))
        l_sr = clahe_sr.apply(l_sr)
        enhanced_sr_clahe = cv2.merge([l_sr, a_sr, b_sr])
        enhanced_sr_clahe = cv2.cvtColor(enhanced_sr_clahe, cv2.COLOR_LAB2BGR)
        enhanced_imgs.append(enhanced_sr_clahe)

    return enhanced_imgs


def detect_with_super_resolution_pipeline(img, model):
    """
    Enhanced detection pipeline with super-resolution
    """
    all_kps = []

    # Get enhanced versions including super-resolution
    enhanced_imgs = enhance_image_with_sr(img)

    # More aggressive configs since we have better quality images
    configs = [
        {'conf': 0.12, 'imgsz': 1280, 'iou': 0.25},  # Lower confidence for more detections
        {'conf': 0.15, 'imgsz': 1024, 'iou': 0.3},  # Balanced
        {'conf': 0.18, 'imgsz': 1536, 'iou': 0.35},  # High resolution
        {'conf': 0.10, 'imgsz': 960, 'iou': 0.2},  # Ultra low confidence
        {'conf': 0.20, 'imgsz': 1792, 'iou': 0.4},  # Very high resolution
        {'conf': 0.08, 'imgsz': 2048, 'iou': 0.15},  # Maximum settings
    ]

    # Strategic pairing of enhanced images with configs
    for img_idx, enhanced_img in enumerate(enhanced_imgs):
        if img_idx == 0:  # Original
            config_indices = [1, 3]  # Moderate settings
        elif img_idx == 1:  # Super-resolution (if available)
            config_indices = [0, 2, 4]  # Aggressive settings for high-quality image
        elif img_idx == 2:  # CLAHE
            config_indices = [0, 5]  # Low confidence + max resolution
        elif img_idx == 3:  # Histogram equalized
            config_indices = [1]  # Balanced
        else:  # Other enhancements
            config_indices = [0, 2]  # Mix of settings

        for config_idx in config_indices:
            try:
                res = model(enhanced_img, verbose=False, **configs[config_idx])
                data = res[0].keypoints.data
                if data.numel() > 0:
                    kps = data.cpu().numpy()[:, :, :2]
                    all_kps.append(kps)
            except Exception as e:
                print(f"Detection failed for enhancement {img_idx}, config {config_idx}: {e}")
                continue

    if not all_kps:
        return np.full((1, 17, 2), np.nan, dtype=np.float32)

    combined = np.vstack(all_kps)
    print(f"Combined {len(combined)} detections before deduplication")

    # Enhanced duplicate removal
    result = remove_duplicate_skeletons_advanced(combined, threshold=25)
    print(f"Final result: {len(result)} unique detections")

    return result


def remove_duplicate_skeletons_advanced(keypoints, threshold=25):
    """
    Enhanced duplicate removal with skeleton similarity check
    """
    if len(keypoints) <= 1:
        return keypoints

    # Calculate centroids and features
    centroids = []
    valid_idx = []
    skeleton_features = []

    for i, sk in enumerate(keypoints):
        if not np.isnan(sk).all():
            c = np.nanmean(sk, axis=0)
            if not np.isnan(c).any():
                centroids.append(c)
                valid_idx.append(i)

                # Calculate skeleton features for similarity check
                features = []

                # Shoulder width
                if not np.isnan(sk[5:7]).any():
                    shoulder_width = np.linalg.norm(sk[5] - sk[6])
                else:
                    shoulder_width = 0
                features.append(shoulder_width)

                # Hip width
                if not np.isnan(sk[11:13]).any():
                    hip_width = np.linalg.norm(sk[11] - sk[12])
                else:
                    hip_width = 0
                features.append(hip_width)

                # Torso height (shoulders to hips)
                if not (np.isnan(sk[5:7]).any() or np.isnan(sk[11:13]).any()):
                    shoulder_center = np.mean(sk[5:7], axis=0)
                    hip_center = np.mean(sk[11:13], axis=0)
                    torso_height = abs(shoulder_center[1] - hip_center[1])
                else:
                    torso_height = 0
                features.append(torso_height)

                skeleton_features.append(features)

    if not centroids:
        return keypoints

    centroids = np.array(centroids)
    skeleton_features = np.array(skeleton_features)
    unique_mask = np.ones(len(centroids), dtype=bool)

    for i in range(len(centroids)):
        if not unique_mask[i]:
            continue
        for j in range(i + 1, len(centroids)):
            if unique_mask[j]:
                centroid_dist = np.linalg.norm(centroids[i] - centroids[j])

                # Check skeleton similarity
                if skeleton_features[i].sum() > 0 and skeleton_features[j].sum() > 0:
                    feature_dist = np.linalg.norm(skeleton_features[i] - skeleton_features[j])
                    feature_similarity = feature_dist / max(np.linalg.norm(skeleton_features[i]), 1)
                else:
                    feature_similarity = 0

                # Remove duplicate if close centroid AND similar skeleton
                if centroid_dist < threshold and (centroid_dist < threshold * 0.6 or feature_similarity < 0.3):
                    unique_mask[j] = False

    unique_inds = [valid_idx[i] for i in range(len(valid_idx)) if unique_mask[i]]
    return keypoints[unique_inds]


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
        for j in range(i + 1, len(centroids)):
            if unique_mask[j] and np.linalg.norm(centroids[i] - centroids[j]) < threshold:
                unique_mask[j] = False

    unique_inds = [valid_idx[i] for i in range(len(valid_idx)) if unique_mask[i]]
    return keypoints[unique_inds]


def process_folder_with_sr(frames_dir, skeleton_dir, model):
    """
    Process .jpg frames with super-resolution pipeline
    """
    os.makedirs(skeleton_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))

    total_frames = len(files)
    detected_frames = 0
    total_people = 0

    # Initialize super-resolution once
    init_super_resolution()

    for fname in tqdm(files, desc=f"Folder {os.path.basename(frames_dir)} (with SR)"):
        path = os.path.join(frames_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        # Use super-resolution enhanced pipeline
        kps = detect_with_super_resolution_pipeline(img, model)

        if not np.isnan(kps).all():
            detected_frames += 1
            total_people += len(kps)

        out_path = os.path.join(skeleton_dir, fname.replace('.jpg', '.npy'))
        np.save(out_path, kps)

    rate = 100 * detected_frames / total_frames if total_frames else 0
    avg_people = total_people / total_frames if total_frames else 0
    print(
        f"{os.path.basename(frames_dir)}: {detected_frames}/{total_frames} detected ({rate:.1f}%), avg people/frame {avg_people:.1f}")
    return total_frames, detected_frames


def test_super_resolution():
    """
    Test super-resolution on a single image
    """
    print("Testing super-resolution...")

    # Test initialization
    sr = init_super_resolution()
    if sr is None:
        print("Super-resolution not available")
        return

    # Test on sample image
    image_path = 'frames_0.5/001/001_0076.jpg'
    if not os.path.exists(image_path):
        print(f"Test image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Could not load test image")
        return

    print(f"Original image shape: {image.shape}")

    start_time = time.time()
    result = apply_super_resolution(image)
    end_time = time.time()

    print(f"Super-resolution completed in {end_time - start_time:.2f} seconds")
    print(f"Result image shape: {result.shape}")

    # Save results for comparison
    cv2.imwrite('test_original.jpg', image)
    cv2.imwrite('test_super_resolution.jpg', result)
    print("Saved test_original.jpg and test_super_resolution.jpg for comparison")


if __name__ == '__main__':
    # Test super-resolution first
    test_super_resolution()

    # Process with super-resolution pipeline
    single_in = os.path.join(FRAMES_DIR, '001')
    single_out = os.path.join(SKELETON_DIR, '001')
    process_folder_with_sr(single_in, single_out, tmp_model)