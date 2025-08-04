import cv2
import numpy as np
import os
import json
from collections import defaultdict
from scipy.spatial.distance import cdist
from tkinter import messagebox
import tkinter as tk


class SinglePersonTracker:
    def __init__(self, video_path, skeleton_base_dir, output_dir="labeled_behaviors_05"):
        # Create a hidden tkinter root for message boxes
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()

        self.video_path = video_path
        self.skeleton_base_dir = skeleton_base_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Behavior mapping
        self.behaviors = {
            0: "nothing/walking",
            1: "suspicious",
            2: "running/panic"
        }

        # Load video
        self.cap = cv2.VideoCapture(video_path)
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.skeleton_fps = 2.0

        # Load all skeletons
        self.skeletons = self.load_skeletons()
        self.skeleton_frames = sorted(self.skeletons.keys())
        self.n_skeleton_frames = len(self.skeleton_frames)

        # Tracking state
        self.selected_person_idx = None  # Index of person in first frame
        self.tracked_path = []  # List of (frame_num, person_idx) tuples
        self.current_skeleton_idx = 0
        self.tracking_confirmed = {}  # frame_num: bool

        # Labeling state
        self.labels = {"segments": []}
        self.labeling_start_frame = None
        self.current_behavior = None
        self.is_labeling = False

        # UI state
        self.mode = "select"  # "select", "track", "label", "reselect", "select_new"
        self.needs_confirmation = False
        self.confirmation_threshold = 100  # pixels
        self.reselect_frame = None  # Frame where we're reselecting

    def load_skeletons(self):
        """Load all skeleton files for the video"""
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        skeleton_path = os.path.join(self.skeleton_base_dir, video_name)

        if not os.path.exists(skeleton_path):
            print(f"Skeleton directory not found: {skeleton_path}")
            return {}

        skeletons = {}
        files = sorted([f for f in os.listdir(skeleton_path) if f.endswith('.npy')])
        print(f"\nLoading skeletons from {skeleton_path}")
        print(f"Found {len(files)} skeleton files")

        for f in files:
            frame_num = int(f.split('_')[-1].split('.')[0])
            skeletons[frame_num] = np.load(os.path.join(skeleton_path, f))

        return skeletons

    def get_person_centroid(self, skeleton):
        """Calculate centroid of a skeleton"""
        if skeleton.shape == (17, 2) and not np.isnan(skeleton).all():
            return np.nanmean(skeleton, axis=0)
        return None

    def find_closest_person(self, target_skeleton, candidates):
        """Find the closest matching person in candidates"""
        target_centroid = self.get_person_centroid(target_skeleton)
        if target_centroid is None:
            return None, float('inf')

        min_dist = float('inf')
        best_idx = None

        for idx, candidate in enumerate(candidates):
            candidate_centroid = self.get_person_centroid(candidate)
            if candidate_centroid is not None:
                dist = np.linalg.norm(target_centroid - candidate_centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx

        return best_idx, min_dist

    def auto_track_person(self):
        """Automatically track the selected person through all frames"""
        if self.selected_person_idx is None:
            return

        self.tracked_path = []

        # Start with selected person in first frame
        first_frame = self.skeleton_frames[0]
        first_skeleton = self.skeletons[first_frame][self.selected_person_idx]
        self.tracked_path.append((first_frame, self.selected_person_idx))
        self.tracking_confirmed[first_frame] = True

        # Track through subsequent frames
        prev_skeleton = first_skeleton

        for frame_idx in range(1, len(self.skeleton_frames)):
            frame_num = self.skeleton_frames[frame_idx]
            candidates = self.skeletons[frame_num]

            # Find best match
            best_idx, min_dist = self.find_closest_person(prev_skeleton, candidates)

            if best_idx is not None:
                self.tracked_path.append((frame_num, best_idx))
                # Mark as needing confirmation if distance is large
                self.tracking_confirmed[frame_num] = min_dist < self.confirmation_threshold
                prev_skeleton = candidates[best_idx]
            else:
                # Lost tracking
                self.tracked_path.append((frame_num, -1))
                self.tracking_confirmed[frame_num] = False

        print(
            f"Auto-tracking complete. {sum(self.tracking_confirmed.values())}/{len(self.tracking_confirmed)} frames confirmed")

    def re_track_from_frame(self, start_frame_idx):
        """Re-track from a specific frame after manual correction"""
        if start_frame_idx >= len(self.skeleton_frames) - 1:
            return

        # Get the corrected person at this frame
        current_frame = self.skeleton_frames[start_frame_idx]
        current_person_idx = None

        for frame_num, person_idx in self.tracked_path:
            if frame_num == current_frame:
                current_person_idx = person_idx
                break

        if current_person_idx is None or current_person_idx < 0:
            return

        # Get skeleton of corrected person
        if current_person_idx >= len(self.skeletons[current_frame]):
            return

        prev_skeleton = self.skeletons[current_frame][current_person_idx]

        # Re-track from this point forward
        for frame_idx in range(start_frame_idx + 1, len(self.skeleton_frames)):
            frame_num = self.skeleton_frames[frame_idx]
            candidates = self.skeletons[frame_num]

            # Find best match
            best_idx, min_dist = self.find_closest_person(prev_skeleton, candidates)

            # Update tracked path
            updated = False
            for i, (f_num, _) in enumerate(self.tracked_path):
                if f_num == frame_num:
                    self.tracked_path[i] = (frame_num, best_idx if best_idx is not None else -1)
                    updated = True
                    break

            if not updated:
                self.tracked_path.append((frame_num, best_idx if best_idx is not None else -1))

            # Update confirmation status
            if best_idx is not None:
                self.tracking_confirmed[frame_num] = min_dist < self.confirmation_threshold
                prev_skeleton = candidates[best_idx]
            else:
                self.tracking_confirmed[frame_num] = False

        print(f"Re-tracked from frame {current_frame} onwards")

    def draw_frame(self, frame, skeleton_frame_num):
        """Draw current frame with annotations"""
        overlay = frame.copy()

        if skeleton_frame_num not in self.skeletons:
            return overlay

        people = self.skeletons[skeleton_frame_num]

        if self.mode == "select" or self.mode == "reselect" or self.mode == "select_new":
            # Draw all people for selection with clear visibility
            for idx, person in enumerate(people):
                if person.shape == (17, 2) and not np.isnan(person).all():
                    # Make selection more visible
                    if self.mode == "reselect":
                        color = (0, 255, 255)  # Yellow for all in reselect mode
                    elif self.mode == "select_new":
                        color = (255, 0, 255)  # Magenta for new person selection
                    else:
                        color = (0, 255, 0) if idx == self.selected_person_idx else (255, 255, 255)

                    self.draw_skeleton(overlay, person, color, thickness=3)

                    centroid = self.get_person_centroid(person)
                    if centroid is not None:
                        # Draw large, highly visible numbers
                        text = f"{idx}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 3.0  # Much larger
                        thickness = 4

                        # Get text size for background
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                        # Position above the person
                        text_x = int(centroid[0] - text_width // 2)
                        text_y = int(centroid[1] - 30)

                        # Draw white background rectangle with border
                        padding = 15
                        cv2.rectangle(overlay,
                                      (text_x - padding, text_y - text_height - padding),
                                      (text_x + text_width + padding, text_y + padding),
                                      (255, 255, 255), -1)
                        cv2.rectangle(overlay,
                                      (text_x - padding, text_y - text_height - padding),
                                      (text_x + text_width + padding, text_y + padding),
                                      (0, 0, 0), 3)

                        # Draw black text
                        cv2.putText(overlay, text,
                                    (text_x, text_y),
                                    font, font_scale, (0, 0, 0), thickness)

                        # Also draw a circle around the person for clarity
                        cv2.circle(overlay, (int(centroid[0]), int(centroid[1])), 50, color, 3)

        elif self.mode in ["track", "label"]:
            # Find current person in tracked path
            current_person_idx = None
            for frame_num, person_idx in self.tracked_path:
                if frame_num == skeleton_frame_num:
                    current_person_idx = person_idx
                    break

            # Draw all people dimmed
            for idx, person in enumerate(people):
                if person.shape == (17, 2) and not np.isnan(person).all():
                    color = (100, 100, 100)
                    self.draw_skeleton(overlay, person, color)

            # Highlight tracked person
            if current_person_idx is not None and current_person_idx >= 0:
                if current_person_idx < len(people):
                    person = people[current_person_idx]
                    if person.shape == (17, 2) and not np.isnan(person).all():
                        # Color based on confirmation status
                        if skeleton_frame_num in self.tracking_confirmed:
                            if self.tracking_confirmed[skeleton_frame_num]:
                                color = (0, 255, 0)  # Green for confirmed
                            else:
                                color = (0, 255, 255)  # Yellow for needs confirmation
                        else:
                            color = (0, 0, 255)  # Red for lost

                        self.draw_skeleton(overlay, person, color, thickness=3)

                        centroid = self.get_person_centroid(person)
                        if centroid is not None:
                            cv2.putText(overlay, "TRACKED",
                                        (int(centroid[0]) - 50, int(centroid[1]) - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        self.draw_ui(overlay, skeleton_frame_num)
        return overlay

    def draw_skeleton(self, img, keypoints, color, thickness=2):
        """Draw skeleton connections"""
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (11, 12)  # Hip
        ]

        for conn in connections:
            if conn[0] < len(keypoints) and conn[1] < len(keypoints):
                pt1 = keypoints[conn[0]]
                pt2 = keypoints[conn[1]]
                if not (np.isnan(pt1).any() or np.isnan(pt2).any()):
                    cv2.line(img, (int(pt1[0]), int(pt1[1])),
                             (int(pt2[0]), int(pt2[1])), color, thickness)

        # Draw joints
        joint_radius = max(3, thickness)
        for point in keypoints:
            if not np.isnan(point).any():
                cv2.circle(img, (int(point[0]), int(point[1])), joint_radius, color, -1)

    def draw_ui(self, img, skeleton_frame_num):
        """Draw UI information"""
        h, w = img.shape[:2]

        # Top info bar
        cv2.rectangle(img, (0, 0), (w, 100), (0, 0, 0), -1)

        # Frame info
        current_display = self.current_skeleton_idx + 1
        cv2.putText(img, f"Frame: {current_display}/{self.n_skeleton_frames}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Mode info
        mode_text = f"Mode: {self.mode.upper()}"
        if self.mode == "select":
            mode_text += " - Press number key to select person"
        elif self.mode == "reselect":
            mode_text += f" - Select correct person for frame {self.current_skeleton_idx + 1}"
        elif self.mode == "select_new":
            mode_text += f" - Select NEW person starting from frame {self.current_skeleton_idx + 1}"
        elif self.mode == "track":
            mode_text += " - Press C to confirm/reject tracking"
        elif self.mode == "label":
            mode_text += " - Press 1/2/3 to label behavior"

        cv2.putText(img, mode_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Behavior labeling info
        if self.is_labeling and self.current_behavior is not None:
            cv2.putText(img,
                        f"Labeling: {self.behaviors[self.current_behavior]} (from frame {self.labeling_start_frame})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Tracking confirmation status
        if self.mode in ["track", "label"] and skeleton_frame_num in self.tracking_confirmed:
            if not self.tracking_confirmed[skeleton_frame_num]:
                cv2.putText(img, "NEEDS CONFIRMATION! Press Y to confirm, N to select correct person",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Instructions
        if self.mode == "select":
            instructions = [
                "SELECT PERSON TO TRACK:",
                "0-9: Select person 0-9",
                "A-Z: Select person 10-35 (A=10, B=11, ... Z=35)",
                "ESC: Exit"
            ]
        elif self.mode == "reselect":
            instructions = [
                "RE-SELECT CORRECT PERSON:",
                "0-9: Select person 0-9",
                "A-Z: Select person 10-35",
                "ESC: Cancel re-selection"
            ]
        elif self.mode == "track":
            instructions = [
                "TRACKING MODE:",
                "SPACE: Play/Pause",
                "A/D: Previous/Next frame",
                "Y: Confirm current tracking",
                "N: Mark as wrong person",
                "R: Re-select person for this frame",
                "P: Track NEW person from current frame",
                "L: Start labeling",
                "ESC: Exit"
            ]
        else:  # label mode
            instructions = [
                "LABELING MODE:",
                "SPACE: Play/Pause",
                "A/D: Previous/Next frame",
                "1/2/3: Start labeling behavior",
                "ENTER: End behavior segment",
                "S: Save all labels",
                "T: Back to tracking mode",
                "ESC: Exit"
            ]

        y_offset = h - len(instructions) * 20 - 10
        for i, text in enumerate(instructions):
            cv2.putText(img, text, (10, y_offset + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Show labeled segments
        if len(self.labels["segments"]) > 0:
            cv2.putText(img, "Labeled segments:", (w - 300, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y = 40
            for segment in self.labels["segments"][-5:]:  # Show last 5
                text = f"Frames {segment['start']}-{segment['end']}: {self.behaviors[segment['behavior']]}"
                cv2.putText(img, text, (w - 300, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                y += 20

    def run(self):
        """Main loop"""
        paused = True

        # Create window with specific size
        cv2.namedWindow("Single Person Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Single Person Tracker", 1400, 900)

        while True:
            # Get current skeleton frame
            current_skeleton_frame = self.skeleton_frames[self.current_skeleton_idx]

            # Calculate video frame
            video_frame = int(current_skeleton_frame * (self.video_fps / self.skeleton_fps))

            # Get frame from video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
            ret, frame = self.cap.read()
            if not ret:
                self.current_skeleton_idx = 0
                continue

            # Auto advance if playing
            if not paused and self.mode in ["track", "label"]:
                self.current_skeleton_idx = (self.current_skeleton_idx + 1) % self.n_skeleton_frames

            # Draw frame
            display_frame = self.draw_frame(frame, current_skeleton_frame)
            cv2.imshow("Single Person Tracker", display_frame)

            key = cv2.waitKey(30 if not paused else 0) & 0xFF

            # Global controls
            if key == 27:  # ESC
                break

            # Mode-specific controls
            if self.mode == "select":
                if ord('0') <= key <= ord('9'):
                    person_idx = key - ord('0')
                    people = self.skeletons[current_skeleton_frame]
                    if person_idx < len(people):
                        self.selected_person_idx = person_idx
                        print(f"Selected person {person_idx}")
                        self.auto_track_person()
                        self.mode = "track"
                        self.current_skeleton_idx = 0

            elif self.mode == "reselect":
                if ord('0') <= key <= ord('9'):
                    person_idx = key - ord('0')
                    people = self.skeletons[current_skeleton_frame]
                    if person_idx < len(people):
                        # Update tracking for this frame
                        for i, (frame_num, _) in enumerate(self.tracked_path):
                            if frame_num == current_skeleton_frame:
                                self.tracked_path[i] = (frame_num, person_idx)
                                break
                        self.tracking_confirmed[current_skeleton_frame] = True
                        print(f"Re-selected person {person_idx} for frame {current_skeleton_frame}")

                        # Re-track from this frame forward
                        self.re_track_from_frame(self.current_skeleton_idx)

                        self.mode = "track"
                elif key == 27:  # ESC to cancel reselection
                    self.mode = "track"

            elif self.mode == "select_new":
                if ord('0') <= key <= ord('9'):
                    person_idx = key - ord('0')
                    people = self.skeletons[current_skeleton_frame]
                    if person_idx < len(people):
                        # Clear previous tracking from this point forward
                        self.tracked_path = [(f, p) for f, p in self.tracked_path if f < current_skeleton_frame]

                        # Add this person as the new tracked person from current frame
                        self.tracked_path.append((current_skeleton_frame, person_idx))
                        self.tracking_confirmed[current_skeleton_frame] = True

                        # Re-track from this frame forward
                        prev_skeleton = people[person_idx]
                        for frame_idx in range(self.current_skeleton_idx + 1, len(self.skeleton_frames)):
                            frame_num = self.skeleton_frames[frame_idx]
                            candidates = self.skeletons[frame_num]

                            best_idx, min_dist = self.find_closest_person(prev_skeleton, candidates)

                            if best_idx is not None:
                                self.tracked_path.append((frame_num, best_idx))
                                self.tracking_confirmed[frame_num] = min_dist < self.confirmation_threshold
                                prev_skeleton = candidates[best_idx]
                            else:
                                self.tracked_path.append((frame_num, -1))
                                self.tracking_confirmed[frame_num] = False

                        print(f"Started tracking new person {person_idx} from frame {current_skeleton_frame}")
                        self.mode = "track"
                elif key == 27:  # ESC to cancel
                    self.mode = "track"

            elif self.mode == "track":
                if key == ord(' '):  # SPACE
                    paused = not paused
                elif key == ord('a'):  # Previous frame
                    self.current_skeleton_idx = max(0, self.current_skeleton_idx - 1)
                elif key == ord('d'):  # Next frame
                    self.current_skeleton_idx = min(self.n_skeleton_frames - 1, self.current_skeleton_idx + 1)
                elif key == ord('y'):  # Confirm tracking
                    self.tracking_confirmed[current_skeleton_frame] = True
                    print(f"Confirmed tracking for frame {current_skeleton_frame}")
                elif key == ord('n'):  # Wrong person
                    self.tracking_confirmed[current_skeleton_frame] = False
                    print(f"Marked frame {current_skeleton_frame} as wrong person")
                elif key == ord('r'):  # Re-select person
                    # Enter reselect mode for current frame
                    self.mode = "reselect"
                    self.reselect_frame = current_skeleton_frame
                elif key == ord('p'):  # Track new person from current frame
                    self.mode = "select_new"
                    print(f"Select a new person to track from frame {current_skeleton_frame}")
                elif key == ord('l'):  # Start labeling
                    self.mode = "label"

            elif self.mode == "label":
                if key == ord(' '):  # SPACE
                    paused = not paused
                elif key == ord('a'):  # Previous frame
                    self.current_skeleton_idx = max(0, self.current_skeleton_idx - 1)
                elif key == ord('d'):  # Next frame
                    self.current_skeleton_idx = min(self.n_skeleton_frames - 1, self.current_skeleton_idx + 1)
                elif key in [ord('1'), ord('2'), ord('3')]:  # Start labeling
                    behavior = int(chr(key)) - 1
                    if not self.is_labeling:
                        self.is_labeling = True
                        self.labeling_start_frame = current_skeleton_frame
                        self.current_behavior = behavior
                        print(f"Started labeling {self.behaviors[behavior]} from frame {current_skeleton_frame}")
                elif key == 13 and self.is_labeling:  # ENTER - end segment
                    self.labels["segments"].append({
                        "start": self.labeling_start_frame,
                        "end": current_skeleton_frame,
                        "behavior": self.current_behavior
                    })
                    print(f"Saved segment: frames {self.labeling_start_frame}-{current_skeleton_frame}")
                    self.is_labeling = False
                    self.current_behavior = None
                elif key == ord('s'):  # Save
                    self.save_labels()
                elif key == ord('t'):  # Back to tracking
                    self.mode = "track"

        self.cap.release()
        cv2.destroyAllWindows()

    def save_labels(self):
        """Save tracked person and labels"""
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]

        # Check existing files to avoid overwriting
        existing_files = [f for f in os.listdir(self.output_dir) if f.startswith(f"{video_name}_person_")]

        # Find next available person ID
        person_save_id = 0
        while f"{video_name}_person_{person_save_id}_tracking.json" in existing_files:
            person_save_id += 1

        # Save tracked path with labels
        output_data = {
            "video": video_name,
            "selected_person_idx": self.selected_person_idx,
            "person_save_id": person_save_id,
            "tracked_path": self.tracked_path,
            "tracking_confirmed": {k: bool(v) for k, v in self.tracking_confirmed.items()},
            # Convert numpy bool to Python bool
            "behavior_segments": self.labels["segments"]
        }

        # Save JSON
        json_file = os.path.join(self.output_dir, f"{video_name}_person_{person_save_id}_tracking.json")
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Create LSTM arrays
        skeleton_sequence = []
        behavior_sequence = []

        for frame_num, person_idx in self.tracked_path:
            if person_idx >= 0 and frame_num in self.skeletons:
                if person_idx < len(self.skeletons[frame_num]):
                    skeleton = self.skeletons[frame_num][person_idx].flatten()
                else:
                    skeleton = np.full(34, np.nan)
            else:
                skeleton = np.full(34, np.nan)

            skeleton_sequence.append(skeleton)

            # Get behavior for this frame
            behavior = 0  # default
            for segment in self.labels["segments"]:
                if segment["start"] <= frame_num <= segment["end"]:
                    behavior = segment["behavior"]
                    break
            behavior_sequence.append(behavior)

        # Save NPZ
        npz_file = os.path.join(self.output_dir, f"{video_name}_person_{person_save_id}_data.npz")
        np.savez(npz_file,
                 skeletons=np.array(skeleton_sequence),
                 behaviors=np.array(behavior_sequence))

        print(f"Saved tracking to {json_file}")
        print(f"Saved LSTM data to {npz_file}")
        print(f"This is saved as person #{person_save_id} (originally selected as person {self.selected_person_idx})")
        print(f"Sequence shape: {np.array(skeleton_sequence).shape}")


# Usage
if __name__ == "__main__":
    video_path = "data/Motion_Emotion/002.mp4"
    skeleton_base_dir = "skeletons_yolo_11"

    tracker = SinglePersonTracker(video_path, skeleton_base_dir)
    tracker.run()