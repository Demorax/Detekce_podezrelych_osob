import sys
import os
import torch
import cv2
import numpy as np
from mmpose.apis import init_model
import torchvision.transforms as transforms

# Add your models directory to path
models_path = 'models'
if os.path.exists(models_path):
    sys.path.insert(0, models_path)

from models.vitpose.models.backbone.vit import ViT
from models.vitpose.models.detectors.top_down import TopDown
from models.vitpose.models.head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from models.vitpose.models.head import topdown_heatmap_simple_head
from mmpose.registry import MODELS

# Register components
MODELS.register_module(module=ViT, force=True)
MODELS.register_module(module=TopDown, force=True)
MODELS.register_module(module=TopdownHeatmapSimpleHead, force=True)
print("✓ Registered ViT from original files")


def create_person_bboxes(img, method='grid'):
    """Create multiple bounding boxes to detect multiple people"""
    h, w = img.shape[:2]
    bboxes = []

    if method == 'grid':
        # Grid-based approach - divide image into overlapping regions
        grid_h, grid_w = 2, 3  # 2x3 grid
        overlap = 0.2  # 20% overlap

        for i in range(grid_h):
            for j in range(grid_w):
                # Calculate region with overlap
                region_h = h / grid_h * (1 + overlap)
                region_w = w / grid_w * (1 + overlap)

                y1 = max(0, int(i * h / grid_h - overlap * h / grid_h / 2))
                x1 = max(0, int(j * w / grid_w - overlap * w / grid_w / 2))
                y2 = min(h, int(y1 + region_h))
                x2 = min(w, int(x1 + region_w))

                # Only add if region is large enough
                if (x2 - x1) > 100 and (y2 - y1) > 100:
                    bboxes.append([x1, y1, x2, y2, 1.0])

    elif method == 'sliding':
        # Sliding window approach
        window_w, window_h = w // 2, h // 2
        step_x, step_y = window_w // 3, window_h // 3

        for y in range(0, h - window_h + 1, step_y):
            for x in range(0, w - window_w + 1, step_x):
                bboxes.append([x, y, x + window_w, y + window_h, 1.0])

    else:  # 'full' - just the full image
        bboxes.append([0, 0, w, h, 1.0])

    return bboxes


def preprocess_image(img, bbox, input_size=(192, 256)):
    """Manually preprocess image for ViTPose"""
    x1, y1, x2, y2 = bbox[:4]

    # Crop person region
    person_img = img[int(y1):int(y2), int(x1):int(x2)]

    # Skip if region is too small
    if person_img.shape[0] < 50 or person_img.shape[1] < 50:
        return None, None

    # Resize to model input size
    person_img = cv2.resize(person_img, input_size)

    # Convert to RGB and normalize
    person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    person_img = person_img.astype(np.float32) / 255.0

    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(person_img).permute(2, 0, 1)
    img_tensor = normalize(img_tensor)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    return img_tensor, (x1, y1, x2 - x1, y2 - y1)


def direct_inference(model, img, bbox):
    """Direct model inference bypassing MMPose API"""
    try:
        # Preprocess
        result = preprocess_image(img, bbox)
        if result[0] is None:
            return None

        img_tensor, crop_info = result

        # Move to GPU if available
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        # Set model to eval mode
        model.eval()

        with torch.no_grad():
            # Forward pass through backbone
            features = model.backbone(img_tensor)

            # Forward pass through head
            heatmaps = model.keypoint_head(features)

            # Convert heatmaps to keypoints
            keypoints = heatmap_to_keypoints(heatmaps, crop_info, img.shape[:2])

        return keypoints

    except Exception as e:
        print(f"Direct inference failed for bbox {bbox}: {e}")
        return None


def heatmap_to_keypoints(heatmaps, crop_info, original_img_size):
    """Convert heatmaps to keypoint coordinates"""
    if isinstance(heatmaps, (list, tuple)):
        heatmaps = heatmaps[-1]  # Take final heatmap

    # Get heatmap dimensions
    batch_size, num_joints, hm_height, hm_width = heatmaps.shape
    heatmaps = heatmaps.cpu().numpy()

    keypoints = []
    x_offset, y_offset, crop_w, crop_h = crop_info

    for b in range(batch_size):
        person_kpts = []
        for j in range(num_joints):
            hm = heatmaps[b, j]

            # Find maximum location
            flat_idx = np.argmax(hm)
            y_coord, x_coord = np.unravel_index(flat_idx, hm.shape)

            # Get confidence (max value)
            confidence = hm[y_coord, x_coord]

            # Convert to original image coordinates
            x_orig = (x_coord / hm_width) * crop_w + x_offset
            y_orig = (y_coord / hm_height) * crop_h + y_offset

            person_kpts.append([x_orig, y_orig, confidence])

        keypoints.append(np.array(person_kpts))

    return keypoints[0] if len(keypoints) == 1 else keypoints


def filter_valid_detections(all_keypoints, min_keypoints=5, min_avg_confidence=0.3):
    """Filter out low-quality detections"""
    valid_detections = []

    for i, keypoints in enumerate(all_keypoints):
        if keypoints is None:
            continue

        # Count keypoints with good confidence
        good_keypoints = np.sum(keypoints[:, 2] > 0.3)
        avg_confidence = np.mean(keypoints[:, 2])

        if good_keypoints >= min_keypoints and avg_confidence >= min_avg_confidence:
            valid_detections.append((i, keypoints))

    return valid_detections


def remove_duplicate_detections(valid_detections, distance_threshold=100):
    """Remove duplicate detections based on distance between centroids"""
    if len(valid_detections) <= 1:
        return valid_detections

    unique_detections = []

    for i, (bbox_idx, keypoints) in enumerate(valid_detections):
        # Calculate centroid of current detection
        valid_kpts = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_kpts) == 0:
            continue

        centroid = np.mean(valid_kpts[:, :2], axis=0)

        # Check if this detection is too close to existing ones
        is_duplicate = False
        for existing_idx, existing_kpts in unique_detections:
            existing_valid = existing_kpts[existing_kpts[:, 2] > 0.3]
            if len(existing_valid) == 0:
                continue

            existing_centroid = np.mean(existing_valid[:, :2], axis=0)
            distance = np.linalg.norm(centroid - existing_centroid)

            if distance < distance_threshold:
                # Keep the detection with higher average confidence
                current_conf = np.mean(keypoints[:, 2])
                existing_conf = np.mean(existing_kpts[:, 2])

                if current_conf > existing_conf:
                    # Replace existing detection
                    unique_detections = [(idx, kpts) for idx, kpts in unique_detections if
                                         not np.array_equal(kpts, existing_kpts)]
                    unique_detections.append((bbox_idx, keypoints))
                is_duplicate = True
                break

        if not is_duplicate:
            unique_detections.append((bbox_idx, keypoints))

    return unique_detections


def test_multi_person_vitpose():
    """Test ViTPose with multiple person detection"""
    config_file = 'models/vitpose/configs/ViTPose_huge_crowdpose_256x192_without_training.py'
    checkpoint_file = 'models/vitpose/vitpose-h-multi-crowdpose.pth'

    print("Loading ViTPose model...")
    try:
        model = init_model(config_file, checkpoint_file, device='cuda')
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

    # Load image
    img_path = 'frames_0.5_upscaled/001/001_0076.jpg'
    img = cv2.imread(img_path)

    if img is None:
        print("Creating dummy image...")
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    else:
        print(f"✓ Loaded image: {img.shape}")

    # Create multiple bounding boxes
    print("Creating detection regions...")
    bboxes = create_person_bboxes(img, method='grid')  # Try 'grid', 'sliding', or 'full'
    print(f"Created {len(bboxes)} detection regions")

    # Run inference on all regions
    print("Running multi-person inference...")
    all_keypoints = []

    for i, bbox in enumerate(bboxes):
        print(f"Processing region {i + 1}/{len(bboxes)}...")
        keypoints = direct_inference(model, img, bbox)
        all_keypoints.append(keypoints)

    # Filter and deduplicate detections
    print("Filtering detections...")
    valid_detections = filter_valid_detections(all_keypoints)
    print(f"Found {len(valid_detections)} valid detections")

    unique_detections = remove_duplicate_detections(valid_detections)
    print(f"After removing duplicates: {len(unique_detections)} unique persons")

    if len(unique_detections) > 0:
        # Visualize all detected persons
        visualize_multi_person_results(img, unique_detections, bboxes, 'multi_person_output.jpg')

        return unique_detections, img
    else:
        print("No valid person detections found!")
        return None, img


def visualize_multi_person_results(img, detections, bboxes, save_path):
    """Visualize multiple person detections"""
    vis_img = img.copy()

    # Colors for different persons
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
    ]

    # Draw detection regions (optional - comment out if too cluttered)
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox[:4]
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 1)

    # Draw each detected person
    for person_idx, (bbox_idx, keypoints) in enumerate(detections):
        color = colors[person_idx % len(colors)]

        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:
                cv2.circle(vis_img, (int(x), int(y)), 4, color, -1)
                cv2.circle(vis_img, (int(x), int(y)), 6, (255, 255, 255), 1)

        # Draw skeleton connections
        connections = [
            (10, 8), (8, 6), (11, 9), (9, 7),  # legs
            (6, 7),  # hips
            (0, 6), (1, 7),  # torso
            (0, 1),  # shoulders
            (0, 2), (1, 3),  # upper arms
            (2, 4), (3, 5),  # forearms
            (12, 13), (13, 0), (13, 1)  # head/neck
        ]

        for start_idx, end_idx in connections:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                    keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(vis_img, start_point, end_point, color, 2)

        # Add person number
        valid_kpts = keypoints[keypoints[:, 2] > 0.3]
        if len(valid_kpts) > 0:
            centroid = np.mean(valid_kpts[:, :2], axis=0)
            cv2.putText(vis_img, f'Person {person_idx + 1}',
                        (int(centroid[0] - 30), int(centroid[1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Add detection count at top of image
    cv2.putText(vis_img, f'Detected: {len(detections)} persons',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_img, f'Detected: {len(detections)} persons',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    cv2.imwrite(save_path, vis_img)
    print(f"✓ Multi-person visualization saved to {save_path}")

    # Print summary
    print(f"\nDetection Summary:")
    print(f"Total persons detected: {len(detections)}")
    for person_idx, (bbox_idx, keypoints) in enumerate(detections):
        valid_kpts = np.sum(keypoints[:, 2] > 0.3)
        avg_conf = np.mean(keypoints[:, 2])
        print(f"Person {person_idx + 1}: {valid_kpts}/14 keypoints, avg confidence: {avg_conf:.3f}")


if __name__ == "__main__":
    result = test_multi_person_vitpose()
    if result[0] is not None:
        print(f"\n✓ Success! Detected {len(result[0])} persons in the image!")
    else:
        print("\n✗ Multi-person detection failed. Check model files.")