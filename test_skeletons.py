import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_skeleton_on_image(image_path, skeleton_path, save_path=None):

    img = cv2.imread(image_path)

    if img is None:
        print("Failed to load image")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load skeleton data
    skeletons = np.load(skeleton_path)
    print(f"Loaded {len(skeletons)} skeleton(s)")

    # COCO skeleton connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
        (11, 12)  # Hip
    ]

    # Joint names for reference
    joint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')

    # Colors for different people
    colors = plt.cm.rainbow(np.linspace(0, 1, len(skeletons)))

    # Draw each skeleton
    for person_idx, (skeleton, color) in enumerate(zip(skeletons, colors)):
        if skeleton.shape != (17, 2) or np.isnan(skeleton).all():
            continue

        # Convert color to RGB values (0-255)
        color_rgb = tuple(int(c * 255) for c in color[:3])

        # Draw connections
        for connection in connections:
            pt1_idx, pt2_idx = connection
            pt1 = skeleton[pt1_idx]
            pt2 = skeleton[pt2_idx]

            # Skip if either point is NaN
            if np.isnan(pt1).any() or np.isnan(pt2).any():
                continue

            # Draw line
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                     color=color[:3], linewidth=2, alpha=0.7)

        # Draw joints
        for joint_idx, joint in enumerate(skeleton):
            if not np.isnan(joint).any():
                plt.scatter(joint[0], joint[1], color=color[:3], s=50,
                            edgecolors='white', linewidth=1.5, alpha=0.9)

                # Optionally add joint numbers
                # plt.text(joint[0]+5, joint[1]+5, str(joint_idx),
                #         fontsize=8, color='white',
                #         bbox=dict(boxstyle="round,pad=0.1", facecolor=color[:3], alpha=0.5))

        # Add person label
        centroid = np.nanmean(skeleton, axis=0)
        if not np.isnan(centroid).any():
            plt.text(centroid[0], centroid[1] - 50, f'Person {person_idx}',
                     fontsize=12, color='white', weight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=color[:3], alpha=0.7))

    plt.title(f'Skeleton Detection - {len(skeletons)} person(s) detected')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def show_skeleton_on_image_cv2(image_path, skeleton_path, save_path=None):
    """
    Alternative version using OpenCV for drawing (faster, no matplotlib)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Create a copy for drawing
    img_draw = img.copy()

    # Load skeleton data
    skeletons = np.load(skeleton_path)
    print(f"Loaded {len(skeletons)} skeleton(s)")

    # COCO skeleton connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
        (11, 12)  # Hip
    ]

    # Colors for different people (BGR format for OpenCV)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]

    # Draw each skeleton
    for person_idx, skeleton in enumerate(skeletons):
        if skeleton.shape != (17, 2) or np.isnan(skeleton).all():
            continue

        color = colors[person_idx % len(colors)]

        # Draw connections
        for connection in connections:
            pt1_idx, pt2_idx = connection
            pt1 = skeleton[pt1_idx]
            pt2 = skeleton[pt2_idx]

            if np.isnan(pt1).any() or np.isnan(pt2).any():
                continue

            cv2.line(img_draw,
                     (int(pt1[0]), int(pt1[1])),
                     (int(pt2[0]), int(pt2[1])),
                     color, 2)

        # Draw joints
        for joint in skeleton:
            if not np.isnan(joint).any():
                cv2.circle(img_draw, (int(joint[0]), int(joint[1])), 5, color, -1)
                cv2.circle(img_draw, (int(joint[0]), int(joint[1])), 6, (255, 255, 255), 1)

        # Add person label
        centroid = np.nanmean(skeleton, axis=0)
        if not np.isnan(centroid).any():
            label = f'Person {person_idx}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw background rectangle
            cv2.rectangle(img_draw,
                          (int(centroid[0]) - 5, int(centroid[1]) - 55),
                          (int(centroid[0]) + text_width + 5, int(centroid[1]) - 35),
                          color, -1)

            # Draw text
            cv2.putText(img_draw, label,
                        (int(centroid[0]), int(centroid[1]) - 40),
                        font, font_scale, (255, 255, 255), thickness)

    # Show result
    cv2.imshow('Skeleton Detection', img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, img_draw)
        print(f"Saved visualization to {save_path}")


# Example usage
if __name__ == "__main__":
    # Example paths
    image_path = "frames_0.5/001/001_0055.jpg"
    skeleton_path = "skeletons_yolo_11_new/001/001_0055.npy"

    # Using matplotlib version (better for Jupyter notebooks)
    show_skeleton_on_image(image_path, skeleton_path, save_path="test_55_2.png")