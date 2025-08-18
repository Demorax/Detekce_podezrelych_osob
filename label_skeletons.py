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
        self.connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (11, 12)  # Hip
        ]
        # Colors for different people (BGR format for OpenCV)
        self.person_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
            (128, 0, 255), (255, 128, 128), (128, 255, 128), (128, 128, 255),
            (255, 255, 128), (255, 128, 255), (128, 255, 255), (64, 128, 255)
        ]

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

        # Footer panel (space under the video for labels/instructions)
        self.footer_height = 160  # pixels

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
        # FIX: correct attribute name (was self.tracke_path)
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
        """Draw current frame with annotations and add a footer area under the video."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        if skeleton_frame_num not in self.skeletons:
            # Even if no skeletons, still create footer canvas to keep layout consistent
            canvas = np.zeros((h + self.footer_height, w, 3), dtype=np.uint8)
            canvas[:h] = overlay
            self.draw_ui(canvas, skeleton_frame_num, video_h=h)
            return canvas

        people = self.skeletons[skeleton_frame_num]

        if self.mode in ["select", "reselect", "select_new"]:
            # Draw all people for selection
            for idx, person in enumerate(people):
                if person.shape == (17, 2) and not np.isnan(person).all():
                    # Use rainbow colors for each person
                    color = self.person_colors[idx % len(self.person_colors)]

                    # Draw skeleton
                    self.draw_skeleton(overlay, person, color, thickness=2)

                    # Get centroid
                    centroid = self.get_person_centroid(person)
                    if centroid is not None:
                        # Draw person label with nice background
                        label = f'Person {idx}'
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2

                        # Get text size
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, thickness)

                        # Draw background rectangle
                        cv2.rectangle(overlay,
                                      (int(centroid[0]) - text_width // 2 - 5,
                                       int(centroid[1]) - 55),
                                      (int(centroid[0]) + text_width // 2 + 5,
                                       int(centroid[1]) - 35),
                                      color, -1)

                        # Draw text
                        cv2.putText(overlay, label,
                                    (int(centroid[0]) - text_width // 2,
                                     int(centroid[1]) - 40),
                                    font, font_scale, (255, 255, 255), thickness)

        elif self.mode in ["track", "label"]:
            # Find current person in tracked path
            current_person_idx = None
            for frame_num, person_idx in self.tracked_path:
                if frame_num == skeleton_frame_num:
                    current_person_idx = person_idx
                    break

            # Draw all people with their colors
            for idx, person in enumerate(people):
                if person.shape == (17, 2) and not np.isnan(person).all():
                    if idx == current_person_idx:
                        # Tracked person - use special color based on status
                        if skeleton_frame_num in self.tracking_confirmed:
                            if self.tracking_confirmed[skeleton_frame_num]:
                                color = (0, 255, 0)  # Green for confirmed
                            else:
                                color = (0, 255, 255)  # Yellow for needs confirmation
                        else:
                            color = (0, 0, 255)  # Red for lost
                        thickness = 3
                    else:
                        # Other people - dimmed
                        color = (100, 100, 100)
                        thickness = 1

                    self.draw_skeleton(overlay, person, color, thickness)

                    # Add label for tracked person
                    if idx == current_person_idx:
                        centroid = self.get_person_centroid(person)
                        if centroid is not None:
                            label = "TRACKED"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.7
                            thickness = 2

                            (text_width, text_height), _ = cv2.getTextSize(
                                label, font, font_scale, thickness)

                            # Background
                            cv2.rectangle(overlay,
                                          (int(centroid[0]) - text_width // 2 - 5,
                                           int(centroid[1]) - 55),
                                          (int(centroid[0]) + text_width // 2 + 5,
                                           int(centroid[1]) - 35),
                                          color, -1)

                            cv2.putText(overlay, label,
                                        (int(centroid[0]) - text_width // 2,
                                         int(centroid[1]) - 40),
                                        font, font_scale, (255, 255, 255), thickness)

        # Compose final canvas with footer under the video
        canvas = np.zeros((h + self.footer_height, w, 3), dtype=np.uint8)
        canvas[:h] = overlay

        # Draw UI (top bar stays over video, labels/instructions go to footer)
        self.draw_ui(canvas, skeleton_frame_num, video_h=h)
        return canvas

    def draw_skeleton(self, img, keypoints, color, thickness=2):
        """Draw skeleton connections like in test_skeletons.py"""
        # Draw connections
        for connection in self.connections:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]

                if not (np.isnan(pt1).any() or np.isnan(pt2).any()):
                    cv2.line(img,
                             (int(pt1[0]), int(pt1[1])),
                             (int(pt2[0]), int(pt2[1])),
                             color, thickness)

        # Draw joints with white edge like in test version
        for point in keypoints:
            if not np.isnan(point).any():
                cv2.circle(img, (int(point[0]), int(point[1])), 5, color, -1)
                cv2.circle(img, (int(point[0]), int(point[1])), 6, (255, 255, 255), 1)

    def draw_ui(self, img, skeleton_frame_num, video_h):
        """Draw UI information. Top bar overlays the video, instructions/labels live in footer below."""
        h, w = img.shape[:2]

        # ----- Top bar (over video) -----
        # Draw solid black background for top bar
        cv2.rectangle(img, (0, 0), (w, 100), (0, 0, 0), -1)

        # Frame info
        current_display = self.current_skeleton_idx + 1
        cv2.putText(img, f"Frame: {current_display}/{self.n_skeleton_frames}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Mode info
        mode_text = f"Mode: {self.mode.upper()}"
        if self.mode == "select":
            mode_text += " - Press number/letter to select person"
        elif self.mode == "reselect":
            mode_text += f" - Select correct person for frame {self.current_skeleton_idx + 1}"
        elif self.mode == "select_new":
            mode_text += f" - Select NEW person starting from frame {self.current_skeleton_idx + 1}"
        elif self.mode == "track":
            mode_text += " - Press C to confirm/reject tracking"
        elif self.mode == "label":
            mode_text += " - Press 1/2/3 to label behavior"

        cv2.putText(img, mode_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # ----- Footer panel (under video) -----
        footer_top = video_h
        footer_bottom = video_h + self.footer_height
        cv2.rectangle(img, (0, footer_top), (w, footer_bottom), (0, 0, 0), -1)

        # Behavior labeling info
        if self.is_labeling and self.current_behavior is not None:
            cv2.putText(img,
                        f"Labeling: {self.behaviors[self.current_behavior]} (from frame {self.labeling_start_frame})",
                        (10, footer_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Instructions
        instructions = []
        if self.mode == "select":
            instructions = [
                "SELECT PERSON TO TRACK:",
                "0-9: Select person 0-9",
                "A-Z/a-z: Select person 10-35 (A=10, B=11, ... Z=35)",
                "ESC: Exit"
            ]
        elif self.mode == "reselect":
            instructions = [
                "RE-SELECT CORRECT PERSON:",
                "0-9: Select person 0-9",
                "A-Z/a-z: Select person 10-35",
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

        # Draw instructions text in footer (left side)
        y_offset = footer_top + 60
        for i, text in enumerate(instructions):
            cv2.putText(img, text, (10, y_offset + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show labeled segments (right side in footer)
        if len(self.labels["segments"]) > 0:
            cv2.putText(img, "Labeled segments (last 5):", (w - 360, footer_top + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            y = footer_top + 55
            for segment in self.labels["segments"][-5:]:  # Show last 5
                text = f"{segment['start']}-{segment['end']}: {self.behaviors[segment['behavior']]}"
                cv2.putText(img, text, (w - 360, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1)
                y += 22

    def run(self):
        """Main loop"""
        paused = True

        # Create window with specific size
        cv2.namedWindow("Single Person Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Single Person Tracker", 1400, 900)

        while True:
            if self.n_skeleton_frames == 0:
                # No skeleton data available
                ret, frame = self.cap.read()
                if not ret:
                    break
                display_frame = self.draw_frame(frame, 0)
                cv2.imshow("Single Person Tracker", display_frame)
                key = cv2.waitKey(30 if not paused else 0) & 0xFF
                if key == 27:
                    break
                continue

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

            # Helper: map key to person index (supports 0-9 and A-Z/a-z -> 10..35)
            def key_to_person_idx(k):
                if ord('0') <= k <= ord('9'):
                    return k - ord('0')
                if ord('A') <= k <= ord('Z'):
                    return 10 + (k - ord('A'))
                if ord('a') <= k <= ord('z'):
                    return 10 + (k - ord('a'))
                return None

            # Global controls
            if key == 27:  # ESC
                break

            # Mode-specific controls
            if self.mode == "select":
                person_idx = key_to_person_idx(key)
                if person_idx is not None:
                    people = self.skeletons[current_skeleton_frame]
                    if person_idx < len(people):
                        self.selected_person_idx = person_idx
                        print(f"Selected person {person_idx}")
                        self.auto_track_person()
                        self.mode = "track"
                        self.current_skeleton_idx = 0

            elif self.mode == "reselect":
                person_idx = key_to_person_idx(key)
                if person_idx is not None:
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
                person_idx = key_to_person_idx(key)
                if person_idx is not None:
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

        # Save traced path with labels
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
