import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_detections(img, boxes, skeletons=None, title="Detection Results", save_path=None, show=True):
    """
    Visualize YOLO bounding boxes and optionally ViTPose skeleton keypoints.

    Args:
        img: Input image (BGR format from cv2)
        boxes: Array of bounding boxes with shape (N, 5+) where columns are [x1, y1, x2, y2, conf, ...]
        skeletons: Optional array of skeleton keypoints with shape (N, num_keypoints, 3) where last dim is [x, y, score]
        title: Title for the plot
        save_path: Optional path to save the visualization
        show: Whether to display the plot (default: True)
    """
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=16, weight='bold')
    ax.axis('off')

    # Colors for different people
    num_detections = len(boxes) if boxes is not None else 0
    colors = plt.cm.rainbow(np.linspace(0, 1, max(num_detections, 1)))

    # Draw bounding boxes
    if boxes is not None and len(boxes) > 0:
        for idx, (box, color) in enumerate(zip(boxes, colors)):
            x1, y1, x2, y2 = box[:4].astype(int)
            conf = box[4] if len(box) > 4 else 0.0

            # Draw rectangle
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                fill=False, edgecolor=color[:3], linewidth=3, alpha=0.8)
            ax.add_patch(rect)

            # Add label with confidence
            label = f'Person {idx}: {conf:.2f}'
            ax.text(x1, y1 - 10, label, fontsize=10, color='white', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color[:3], alpha=0.7))

    # Draw skeletons if provided
    if skeletons is not None and len(skeletons) > 0:
        # COCO skeleton connections (17 keypoints format)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (11, 12)  # Hip
        ]

        for person_idx, (skeleton, color) in enumerate(zip(skeletons, colors)):
            # Handle different skeleton formats
            if skeleton.shape[-1] == 3:  # (num_keypoints, 3) with scores
                keypoints = skeleton[:, :2]
                scores = skeleton[:, 2]
            elif skeleton.shape[-1] == 2:  # (num_keypoints, 2) without scores
                keypoints = skeleton
                scores = np.ones(len(skeleton))
            else:
                continue

            # Draw connections
            for connection in connections:
                pt1_idx, pt2_idx = connection

                # Check if indices are valid for this skeleton
                if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
                    continue

                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]
                score1 = scores[pt1_idx]
                score2 = scores[pt2_idx]

                # Only draw if both keypoints have sufficient confidence
                if score1 > 0.3 and score2 > 0.3:
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                           color=color[:3], linewidth=2, alpha=0.7)

            # Draw keypoints
            for joint_idx, (joint, score) in enumerate(zip(keypoints, scores)):
                if score > 0.3:  # Only draw confident keypoints
                    ax.scatter(joint[0], joint[1], color=color[:3], s=60,
                             edgecolors='white', linewidth=2, alpha=0.9, zorder=10)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def visualize_skeletons_only(img, skeletons, title="Skeleton Detection", save_path=None, show=True):
    """
    Visualize only skeleton keypoints without bounding boxes.

    Args:
        img: Input image (BGR format from cv2)
        skeletons: Array of skeleton keypoints with shape (N, num_keypoints, 2 or 3)
        title: Title for the plot
        save_path: Optional path to save the visualization
        show: Whether to display the plot
    """
    visualize_detections(img, boxes=None, skeletons=skeletons, title=title,
                        save_path=save_path, show=show)
