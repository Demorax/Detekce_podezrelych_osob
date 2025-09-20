import sys
import os
import torch
import cv2
import numpy as np
from mmpose.apis import init_model, inference_topdown
import torchvision.transforms as transforms

# Add your models directory to path
models_path = 'models'
if os.path.exists(models_path):
    sys.path.insert(0, models_path)

from models.vitpose.models.backbone.vit import ViT
from models.vitpose.models.detectors.top_down import TopDown
from models.vitpose.models.head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from models.vitpose.models.head import topdown_heatmap_simple_head  # This triggers the @HEADS.register_module()
from mmpose.registry import MODELS

# Register all components with MMPose
MODELS.register_module(module=ViT, force=True)
MODELS.register_module(module=TopDown, force=True)
MODELS.register_module(module=TopdownHeatmapSimpleHead, force=True)
print("✓ Registered ViT from original files")

from mmpose.apis import init_model


def preprocess_image(img, bbox, input_size=(192, 256)):
    """Manually preprocess image for ViTPose"""
    x1, y1, x2, y2 = bbox[:4]

    # Crop person region
    person_img = img[int(y1):int(y2), int(x1):int(x2)]

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
        img_tensor, crop_info = preprocess_image(img, bbox)

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
        print(f"Direct inference failed: {e}")
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


def test_direct_vitpose():
    """Test ViTPose with direct inference"""
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

    # Create bounding box (full image for now)
    h, w = img.shape[:2]
    bbox = [0, 0, w, h, 1.0]

    print("Running direct inference...")
    keypoints = direct_inference(model, img, bbox)

    if keypoints is not None:
        print("✓ Direct inference successful!")
        print(f"Keypoints shape: {keypoints.shape}")

        # Count valid keypoints
        valid_kpts = np.sum(keypoints[:, 2] > 0.3)
        print(f"Valid keypoints (conf > 0.3): {valid_kpts}/{len(keypoints)}")

        # Visualize
        visualize_direct_results(img, keypoints, 'direct_pose_output.jpg')

        return keypoints, img
    else:
        print("Direct inference failed!")
        return None, img


def visualize_direct_results(img, keypoints, save_path):
    """Visualize keypoints directly"""
    vis_img = img.copy()

    # CrowdPose keypoint names for reference
    keypoint_names = [
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        'top_head', 'neck'
    ]

    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:  # Only draw confident keypoints
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)  # Green for high conf, yellow for medium
            cv2.circle(vis_img, (int(x), int(y)), 4, color, -1)
            cv2.circle(vis_img, (int(x), int(y)), 6, (255, 255, 255), 1)

            # Add keypoint index
            cv2.putText(vis_img, str(i), (int(x) - 5, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Draw skeleton connections (CrowdPose format)
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

            cv2.line(vis_img, start_point, end_point, (255, 0, 0), 2)

    cv2.imwrite(save_path, vis_img)
    print(f"✓ Visualization saved to {save_path}")

    # Print keypoint summary
    print("\nKeypoint Detection Summary:")
    for i, (x, y, conf) in enumerate(keypoints):
        if i < len(keypoint_names):
            status = "✓" if conf > 0.3 else "✗"
            print(f"{status} {keypoint_names[i]}: ({x:.1f}, {y:.1f}) conf={conf:.3f}")


if __name__ == "__main__":
    result = test_direct_vitpose()
    if result[0] is not None:
        print("\n✓ Success! Direct ViTPose inference working!")
    else:
        print("\n✗ Direct inference failed. Check model files.")