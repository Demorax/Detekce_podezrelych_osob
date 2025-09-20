import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import time
import torch
import mmcv
import mmpose.models
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import PoseDataSample

# --- Configuration ---
FRAMES_DIR = 'frames_0.5'
FRAMES_UPSCALED_DIR = 'frames_0.5_upscaled'
SKELETON_DIR = 'skeletons_yolo_11_new'
SKELETON_UPSCALED_DIR = 'skeletons_yolo_11_upscaled_2'
MODEL_PATH = 'yolo11x-pose.pt'
YOLO_MODEL_DETECTION = 'models/yolo/yolo11x.pt'
SUPER_RESOLUTION_MODEL_PATH = 'models/super_resolution'
VITPOSE_MODEL_PATH = 'models/vitpose'

# Ensure base output folders exist
os.makedirs(FRAMES_UPSCALED_DIR, exist_ok=True)
os.makedirs(SKELETON_UPSCALED_DIR, exist_ok=True)

# Load YOLOv8 Pose model once
tmp_model = YOLO(MODEL_PATH)

# Models
_DETECTION_MODEL = None
_VITPOSE_MODEL = None
_DATASET_INFO = None

# --- Super-resolution setup ---
_SR_NET = None
_SR_SCALE = 4
_SR_PB = os.path.join(SUPER_RESOLUTION_MODEL_PATH, "ESPCN_x4.pb")


def init_yolo_vitpose_models():
    """
    Initialize YOLO detection model and ViTPose model for MMPose 1.3.2
    """
    global _DETECTION_MODEL, _VITPOSE_MODEL, _DATASET_INFO

    if _DETECTION_MODEL is None:
        _DETECTION_MODEL = YOLO(YOLO_MODEL_DETECTION)
        print("YOLO detection model loaded")

    if _VITPOSE_MODEL is None:
        try:
            config_file = os.path.join(VITPOSE_MODEL_PATH, 'ViTPose_huge_256x192.py')
            checkpoint_file = os.path.join(VITPOSE_MODEL_PATH, 'vitpose_huge.pth')

            if not os.path.exists(checkpoint_file):
                print(f"ViTPose checkpoint not found: {checkpoint_file}")
                return

            if not os.path.exists(config_file):
                print(f"Config file not found: {config_file}")
                return

            # For MMPose 1.3.2 - use the new MMPoseInferencer API
            from mmpose.apis import MMPoseInferencer

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Initialize with config and checkpoint
            _VITPOSE_MODEL = MMPoseInferencer(
                pose2d=config_file,
                pose2d_weights=checkpoint_file,
                device=device,
                scope='mmpose'
            )

            print("ViTPose model loaded successfully with MMPose 1.3.2 API")
            _DATASET_INFO = None  # Not needed with new API

        except Exception as e:
            print(f"Failed to load ViTPose: {e}")
            print("Trying alternative approach...")

            # Alternative: Use pre-built ViTPose model name
            try:
                from mmpose.apis import MMPoseInferencer

                # Try using a pre-defined model name instead
                _VITPOSE_MODEL = MMPoseInferencer(
                    pose2d='human',  # Use default human pose model
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                print("ViTPose loaded with default human pose model")

            except Exception as e2:
                print(f"Alternative approach also failed: {e2}")
                _VITPOSE_MODEL = None

    return _DETECTION_MODEL, _VITPOSE_MODEL


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


def create_upscaled_frames(input_folder, output_folder):
    """
    Create upscaled versions of all frames in a folder
    """
    os.makedirs(output_folder, exist_ok=True)

    files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])

    print(f"Creating upscaled frames for {os.path.basename(input_folder)}...")

    # Initialize super-resolution once
    init_super_resolution()

    for fname in tqdm(files, desc=f"Upscaling {os.path.basename(input_folder)}"):
        input_path = os.path.join(input_folder, fname)
        output_path = os.path.join(output_folder, fname)

        # Skip if already exists
        if os.path.exists(output_path):
            continue

        img = cv2.imread(input_path)
        if img is None:
            continue

        # Apply super-resolution or high-quality upscaling
        upscaled_img = apply_super_resolution(img)

        # Save upscaled image
        cv2.imwrite(output_path, upscaled_img)

    print(f"Completed upscaling for {os.path.basename(input_folder)}")


def enhanced_detection_pipeline(img, model):
    """
    Enhanced detection pipeline optimized for upscaled images
    """
    all_kps = []

    # For upscaled images, we can use more aggressive settings
    enhanced_imgs = []

    # 1. Original upscaled image
    enhanced_imgs.append(img)

    # 2. CLAHE enhancement on upscaled image
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced1 = cv2.merge([l, a, b])
    enhanced1 = cv2.cvtColor(enhanced1, cv2.COLOR_LAB2BGR)
    enhanced_imgs.append(enhanced1)

    # 3. Histogram equalization
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    enhanced2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    enhanced_imgs.append(enhanced2)

    # 4. Gamma correction
    gamma = 0.8
    enhanced3 = np.power(img / 255.0, gamma) * 255.0
    enhanced3 = enhanced3.astype(np.uint8)
    enhanced_imgs.append(enhanced3)

    # Higher confidence configs for upscaled images (better quality detections)
    configs = [
        {'conf': 0.25, 'imgsz': 1280, 'iou': 0.45},  # High confidence, good balance
        {'conf': 0.20, 'imgsz': 1024, 'iou': 0.4},   # Medium-high confidence
        {'conf': 0.30, 'imgsz': 1536, 'iou': 0.5},   # High confidence, high res
        {'conf': 0.35, 'imgsz': 1792, 'iou': 0.55},  # Very high confidence, very high res
        {'conf': 0.15, 'imgsz': 2048, 'iou': 0.35},  # Moderate confidence, max res (catch distant people)
    ]

    # Process each enhanced image
    for img_idx, enhanced_img in enumerate(enhanced_imgs):
        if img_idx == 0:  # Original upscaled
            config_indices = [0, 2]  # High confidence, good coverage
        elif img_idx == 1:  # CLAHE
            config_indices = [1, 3]  # Medium-high and very high confidence
        elif img_idx == 2:  # Histogram equalized
            config_indices = [0, 4]  # High confidence + distant people coverage
        else:  # Gamma corrected
            config_indices = [1]

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

    # Enhanced duplicate removal for crowds
    return remove_duplicate_skeletons_advanced(combined, threshold=12)  # Even tighter for crowds


def remove_duplicate_skeletons_advanced(keypoints, threshold=15):
    """
    Enhanced duplicate removal with skeleton similarity check - optimized for crowds
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
                if centroid_dist < threshold and (centroid_dist < threshold * 0.7 or feature_similarity < 0.25):
                    unique_mask[j] = False

    unique_inds = [valid_idx[i] for i in range(len(valid_idx)) if unique_mask[i]]
    return keypoints[unique_inds]


def process_upscaled_folder(frames_dir, skeleton_dir, model):
    """
    Process upscaled frames for skeleton detection
    """
    os.makedirs(skeleton_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))

    total_frames = len(files)
    detected_frames = 0
    total_people = 0

    for fname in tqdm(files, desc=f"Detecting skeletons in {os.path.basename(frames_dir)}"):
        path = os.path.join(frames_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        # Use enhanced detection pipeline for upscaled images
        kps = enhanced_detection_pipeline(img, model)

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


def process_video_with_upscaling(video_name):
    """
    Complete pipeline: create upscaled frames -> detect skeletons
    """
    print(f"\n=== Processing {video_name} with upscaling ===")

    # Paths
    input_frames = os.path.join(FRAMES_DIR, video_name)
    upscaled_frames = os.path.join(FRAMES_UPSCALED_DIR, video_name)
    upscaled_skeletons = os.path.join(SKELETON_UPSCALED_DIR, video_name)

    if not os.path.exists(input_frames):
        print(f"Input folder not found: {input_frames}")
        return

    # Step 1: Create upscaled frames
    print("Step 1: Creating upscaled frames...")
    start_time = time.time()
    create_upscaled_frames(input_frames, upscaled_frames)
    upscale_time = time.time() - start_time
    print(f"Upscaling completed in {upscale_time:.2f} seconds")

    # Step 2: Detect skeletons in upscaled frames
    print("Step 2: Detecting skeletons in upscaled frames...")
    start_time = time.time()
    total, detected = process_upscaled_folder(upscaled_frames, upscaled_skeletons, tmp_model)
    detection_time = time.time() - start_time
    print(f"Skeleton detection completed in {detection_time:.2f} seconds")

    print(f"Total processing time: {upscale_time + detection_time:.2f} seconds")
    print(f"Results saved to: {upscaled_skeletons}")


def process_single_image_skeleton_detection(img_path, model, save_skeleton=True, save_visualization=True):
    """
    Process a single image for skeleton detection

    Args:
        img_path: Path to the image file
        model: YOLO pose model
        save_skeleton: Whether to save skeleton data as .npy
        save_visualization: Whether to save visualization image

    Returns:
        keypoints: Detected keypoints array
    """
    print(f"\n=== Processing single image: {img_path} ===")

    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return None

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return None

    print(f"Image loaded: {img.shape}")

    # Step 1: Detect skeletons
    start_time = time.time()
    keypoints = enhanced_detection_pipeline(img, model)
    detection_time = time.time() - start_time

    num_people = len(keypoints) if not np.isnan(keypoints).all() else 0
    print(f"Detection completed in {detection_time:.2f} seconds")
    print(f"Found {num_people} people")

    # Step 2: Save results
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    if save_skeleton:
        # Save skeleton data
        skeleton_output_dir = os.path.join(SKELETON_UPSCALED_DIR, 'single_tests')
        os.makedirs(skeleton_output_dir, exist_ok=True)
        skeleton_file = os.path.join(skeleton_output_dir, f"{base_name}_skeletons.npy")
        np.save(skeleton_file, keypoints)
        print(f"Skeleton data saved to: {skeleton_file}")

    if save_visualization:
        # Create and save visualization
        vis_img = visualize_skeletons_on_image(img, keypoints)
        vis_output_dir = os.path.join(SKELETON_UPSCALED_DIR, 'visualizations')
        os.makedirs(vis_output_dir, exist_ok=True)
        vis_file = os.path.join(vis_output_dir, f"{base_name}_visualization.jpg")
        cv2.imwrite(vis_file, vis_img)
        print(f"Visualization saved to: {vis_file}")

    return keypoints


def visualize_skeletons_on_image(img, keypoints):
    """
    Draw skeletons on image for visualization
    """
    if np.isnan(keypoints).all():
        return img

    img_vis = img.copy()

    # COCO skeleton connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
        (11, 12)  # Hip
    ]

    # Colors for different people
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (128, 0, 255), (255, 128, 128), (128, 255, 128), (128, 128, 255),
    ]

    for person_idx, skeleton in enumerate(keypoints):
        if skeleton.shape != (17, 2) or np.isnan(skeleton).all():
            continue

        color = colors[person_idx % len(colors)]

        # Draw connections
        for connection in connections:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(skeleton) and pt2_idx < len(skeleton):
                pt1 = skeleton[pt1_idx]
                pt2 = skeleton[pt2_idx]

                if not (np.isnan(pt1).any() or np.isnan(pt2).any()):
                    cv2.line(img_vis,
                             (int(pt1[0]), int(pt1[1])),
                             (int(pt2[0]), int(pt2[1])),
                             color, 3)

        # Draw joints
        for joint_idx, joint in enumerate(skeleton):
            if not np.isnan(joint).any():
                cv2.circle(img_vis, (int(joint[0]), int(joint[1])), 6, color, -1)
                cv2.circle(img_vis, (int(joint[0]), int(joint[1])), 7, (255, 255, 255), 2)

        # Add person label
        centroid = np.nanmean(skeleton, axis=0)
        if not np.isnan(centroid).any():
            label = f'Person {person_idx}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw background rectangle
            cv2.rectangle(img_vis,
                          (int(centroid[0]) - text_width // 2 - 5,
                           int(centroid[1]) - 60),
                          (int(centroid[0]) + text_width // 2 + 5,
                           int(centroid[1]) - 35),
                          color, -1)

            # Draw text
            cv2.putText(img_vis, label,
                        (int(centroid[0]) - text_width // 2,
                         int(centroid[1]) - 40),
                        font, font_scale, (255, 255, 255), thickness)

    # Add title
    title = f"Skeleton Detection - {len(keypoints)} person(s) detected"
    cv2.putText(img_vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img_vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

    return img_vis


def test_single_upscaled_image(img_name):
    """
    Test skeleton detection on a single upscaled image

    Args:
        img_name: Relative path from FRAMES_UPSCALED_DIR (e.g., '001/001_0076.jpg')
    """
    print(f"\n=== Testing single upscaled image: {img_name} ===")

    # Full path to upscaled image
    img_path = os.path.join(FRAMES_UPSCALED_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"Error: Upscaled image not found at {img_path}")
        print(f"Make sure you've run the upscaling step first or check the path")
        return None

    # Process the image
    keypoints = process_single_image_skeleton_detection(
        img_path,
        tmp_model,
        save_skeleton=True,
        save_visualization=True
    )

    if keypoints is not None:
        print(f"\nResults summary:")
        print(f"- Number of people detected: {len(keypoints)}")
        print(f"- Keypoints shape: {keypoints.shape}")
        print(f"- Valid detections: {np.sum(~np.isnan(keypoints).all(axis=(1, 2)))}")

    return keypoints


# Replace your main section with this:
if __name__ == '__main__':

    # Process video 001 with upscaling pipeline
    #process_video_with_upscaling('001')

    # Test single upscaled image
    img_to_process = '001/001_0076.jpg'

    # Option 1: Test just the upscaled image
    #test_single_upscaled_image(img_to_process)
    #init_yolo_vitpose_models()

