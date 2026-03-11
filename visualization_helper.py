import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


SUSPICIOUS_CLASS_IDS = {
    24: 'backpack',
    34: 'baseball bat',
    44: 'knife',
    76: 'scissors',
}

_ITEM_COLORS = {
    24: (1.0, 0.6, 0.0),   # orange - backpack
    34: (0.9, 0.1, 0.1),   # red - baseball bat
    44: (0.8, 0.0, 0.8),   # magenta - knife
    76: (0.1, 0.7, 0.9),   # cyan - scissors
}


def visualize_detections(img, boxes, skeletons=None, item_boxes=None,
                         title="Detection Results", save_path=None, show=True):
    """
    Visualize bounding boxes, ViTPose skeletons, and suspicious items.

    Args:
        img: Input image (BGR format from cv2)
        boxes: (N, 5+) person boxes [x1, y1, x2, y2, conf, ...]
        skeletons: Optional (N, num_keypoints, 3) with [x, y, score]
        item_boxes: Optional (M, 6) suspicious items [x1, y1, x2, y2, conf, class_id]
        title: Plot title
        save_path: Optional path to save
        show: Whether to display
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=16, weight='bold')
    ax.axis('off')

    num_detections = len(boxes) if boxes is not None else 0
    distinct_colors = [
        (1.0, 0.0, 0.0),
        (0.0, 0.5, 1.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.5, 0.0),
        (0.5, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.5, 1.0, 0.0),
        (1.0, 0.0, 0.5),
        (0.0, 1.0, 0.5),
        (0.5, 0.5, 1.0),
        (1.0, 0.5, 0.5),
        (0.5, 1.0, 0.5),
        (1.0, 1.0, 0.5),
        (0.5, 0.5, 0.5),
        (0.75, 0.25, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.25, 0.75, 0.5),
    ]

    if num_detections <= len(distinct_colors):
        colors = distinct_colors[:num_detections]
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, num_detections))

    # Person bounding boxes
    if boxes is not None and len(boxes) > 0:
        for idx, (box, color) in enumerate(zip(boxes, colors)):
            x1, y1, x2, y2 = box[:4].astype(int)
            conf = box[4] if len(box) > 4 else 0.0
            edge_color = color if isinstance(color, tuple) else color[:3]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  fill=False, edgecolor=edge_color, linewidth=3, alpha=0.8)
            ax.add_patch(rect)
            face_color = color if isinstance(color, tuple) else color[:3]
            ax.text(x1, y1 - 10, f'Person {idx}: {conf:.2f}', fontsize=10,
                    color='white', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=face_color, alpha=0.7))

    # Suspicious item boxes
    if item_boxes is not None and len(item_boxes) > 0:
        legend_patches = []
        seen_classes = set()
        for item in item_boxes:
            x1, y1, x2, y2, conf, cls_id = item
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(cls_id)
            name = SUSPICIOUS_CLASS_IDS.get(cls_id, f'item_{cls_id}')
            color = _ITEM_COLORS.get(cls_id, (1.0, 0.8, 0.0))

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  fill=True, facecolor=color, edgecolor=color,
                                  linewidth=2, alpha=0.25)
            ax.add_patch(rect)
            rect_border = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        fill=False, edgecolor=color, linewidth=2.5, alpha=0.9,
                                        linestyle='--')
            ax.add_patch(rect_border)
            ax.text(x1, y2 + 14, f'{name} {conf:.2f}', fontsize=9,
                    color='white', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.85))

            if cls_id not in seen_classes:
                legend_patches.append(mpatches.Patch(color=color, label=name))
                seen_classes.add(cls_id)

        if legend_patches:
            ax.legend(handles=legend_patches, loc='upper right', fontsize=10,
                      framealpha=0.7, facecolor='black', labelcolor='white')

    # Skeletons
    if skeletons is not None and len(skeletons) > 0:
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 12)
        ]

        for person_idx, (skeleton, color) in enumerate(zip(skeletons, colors)):
            if skeleton.shape[-1] == 3:
                keypoints = skeleton[:, :2]
                scores = skeleton[:, 2]
            elif skeleton.shape[-1] == 2:
                keypoints = skeleton
                scores = np.ones(len(skeleton))
            else:
                continue

            line_color = color if isinstance(color, tuple) else color[:3]
            for pt1_idx, pt2_idx in connections:
                if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
                    continue
                if scores[pt1_idx] > 0.3 and scores[pt2_idx] > 0.3:
                    pt1, pt2 = keypoints[pt1_idx], keypoints[pt2_idx]
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                            color=line_color, linewidth=2, alpha=0.7)

            point_color = color if isinstance(color, tuple) else color[:3]
            for joint, score in zip(keypoints, scores):
                if score > 0.3:
                    ax.scatter(joint[0], joint[1], color=point_color, s=60,
                               edgecolors='white', linewidth=2, alpha=0.9, zorder=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_skeletons_only(img, skeletons, title="Skeleton Detection", save_path=None, show=True):
    visualize_detections(img, boxes=None, skeletons=skeletons, title=title,
                         save_path=save_path, show=show)
