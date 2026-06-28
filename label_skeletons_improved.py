"""
Vylepšený nástroj pro sledování osob a označování chování s lepším UI
Používá matplotlib pro vizualizaci a tkinter pro ovládání
"""
import cv2
import numpy as np
import os
import json
from collections import defaultdict
from scipy.spatial.distance import cdist
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches


class ImprovedPersonTracker:
    """
    Vylepšený tracker s matplotlib vizualizací a moderním UI

    Args:
        video_path: Cesta k video souboru
        skeleton_base_dir: Složka se skeleton daty
        output_dir: Výstupní složka pro labeled data
    """

    def __init__(self, video_path, skeleton_base_dir, output_dir="labeled_behaviors_05"):
        print(f"\n{'='*70}")
        print("INITIALIZING IMPROVED PERSON TRACKER")
        print('='*70)

        self.video_path = video_path
        self.skeleton_base_dir = skeleton_base_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"Video path: {video_path}")
        print(f"Skeleton dir: {skeleton_base_dir}")
        print(f"Output dir: {output_dir}")

        # Check if video exists
        if not os.path.exists(video_path):
            print(f"⚠ WARNING: Video file not found: {video_path}")
        else:
            print(f"✓ Video file found")

        # Check if skeleton dir exists
        if not os.path.exists(skeleton_base_dir):
            print(f"⚠ WARNING: Skeleton directory not found: {skeleton_base_dir}")
        else:
            print(f"✓ Skeleton directory found")

        # Behavior mapping
        self.behaviors = {
            0: "nothing/walking",
            1: "suspicious",
            2: "running/panic"
        }

        # Barvy pro chování
        self.behavior_colors = {
            0: '#4CAF50',  # Zelená pro walking
            1: '#FFC107',  # Žlutá pro suspicious
            2: '#F44336'   # Červená pro running/panic
        }

        # Load video
        print("\nLoading video...")
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("⚠ ERROR: Could not open video file!")
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.skeleton_fps = 2.0

        print(f"✓ Video FPS: {self.video_fps}")
        print(f"✓ Total video frames: {self.total_video_frames}")

        # Load all skeletons
        print("\nLoading skeletons...")
        self.skeletons = self.load_skeletons()
        self.skeleton_frames = sorted(self.skeletons.keys())
        self.n_skeleton_frames = len(self.skeleton_frames)

        print(f"✓ Loaded {self.n_skeleton_frames} skeleton frames")
        if self.n_skeleton_frames > 0:
            print(f"✓ Frame range: {self.skeleton_frames[0]} - {self.skeleton_frames[-1]}")
        print('='*70)

        # Tracking state
        self.selected_person_idx = None
        self.tracked_path = []
        self.current_skeleton_idx = 0
        self.tracking_confirmed = {}

        # COCO skeleton connections
        self.connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (11, 12)  # Hip
        ]

        # Labeling state
        self.labels = {"segments": []}
        self.labeling_start_frame = None
        self.current_behavior = None
        self.is_labeling = False

        # UI state
        self.mode = "select"  # "select", "track", "label", "reselect", "select_new"
        self.confirmation_threshold = 100  # pixels
        self.paused = True

        # UI window
        self.root = None
        self.canvas = None
        self.fig = None
        self.ax = None

    def load_skeletons(self):
        """
        Načte všechny skeleton soubory pro video

        Returns:
            dict: Dictionary {frame_num: skeleton_array}
        """
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
        """
        Vypočítá centroid skeletu

        Args:
            skeleton: numpy array (17, 2)

        Returns:
            numpy array (2,) nebo None
        """
        if skeleton.shape == (17, 2) and not np.isnan(skeleton).all():
            return np.nanmean(skeleton, axis=0)
        return None

    def find_closest_person(self, target_skeleton, candidates):
        """
        Najde nejpodobnější osobu v candidates

        Args:
            target_skeleton: numpy array (17, 2)
            candidates: list of numpy arrays

        Returns:
            tuple: (best_idx, min_dist)
        """
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
        """Automaticky sleduje vybranou osobu přes všechny framy"""
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
                self.tracking_confirmed[frame_num] = min_dist < self.confirmation_threshold
                prev_skeleton = candidates[best_idx]
            else:
                self.tracked_path.append((frame_num, -1))
                self.tracking_confirmed[frame_num] = False

        confirmed_count = sum(self.tracking_confirmed.values())
        print(f"Auto-tracking complete. {confirmed_count}/{len(self.tracking_confirmed)} frames confirmed")

    def re_track_from_frame(self, start_frame_idx):
        """
        Znovu sleduje od konkrétního framu po manuální korekci

        Args:
            start_frame_idx: Index framu odkud začít re-tracking
        """
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

        if current_person_idx >= len(self.skeletons[current_frame]):
            return

        prev_skeleton = self.skeletons[current_frame][current_person_idx]

        # Re-track from this point forward
        for frame_idx in range(start_frame_idx + 1, len(self.skeleton_frames)):
            frame_num = self.skeleton_frames[frame_idx]
            candidates = self.skeletons[frame_num]

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

            if best_idx is not None:
                self.tracking_confirmed[frame_num] = min_dist < self.confirmation_threshold
                prev_skeleton = candidates[best_idx]
            else:
                self.tracking_confirmed[frame_num] = False

        print(f"Re-tracked from frame {current_frame} onwards")

    def draw_skeleton_matplotlib(self, ax, keypoints, color, linewidth=2, alpha=0.8, label=None):
        """
        Vykreslí skeleton pomocí matplotlib (lepší kvalita než OpenCV)

        Args:
            ax: matplotlib axes
            keypoints: numpy array (17, 2)
            color: barva (RGB tuple nebo hex)
            linewidth: šířka čar
            alpha: průhlednost
            label: volitelný label pro osobu
        """
        # Draw connections
        for connection in self.connections:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]

                if not (np.isnan(pt1).any() or np.isnan(pt2).any()):
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                           color=color, linewidth=linewidth, alpha=alpha, zorder=5)

        # Draw joints
        valid_points = keypoints[~np.isnan(keypoints).any(axis=1)]
        if len(valid_points) > 0:
            ax.scatter(valid_points[:, 0], valid_points[:, 1],
                      color=color, s=80, edgecolors='white',
                      linewidth=2, alpha=alpha, zorder=10)

        # Draw label if provided
        if label is not None:
            centroid = self.get_person_centroid(keypoints)
            if centroid is not None:
                ax.text(centroid[0], centroid[1] - 40, label,
                       fontsize=11, color='white', weight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8))

    def update_display(self):
        """Aktualizuje zobrazení s matplotlib vizualizací"""
        if self.n_skeleton_frames == 0:
            print("No skeleton frames available")
            # Show error on canvas
            if self.ax is not None:
                self.ax.clear()
                self.ax.text(0.5, 0.5, 'No skeleton data found!\nCheck skeleton_base_dir path.',
                           ha='center', va='center', fontsize=16, color='red',
                           transform=self.ax.transAxes)
                self.ax.axis('off')
                if self.canvas is not None:
                    self.canvas.draw()
            return

        # Get current skeleton frame
        current_skeleton_frame = self.skeleton_frames[self.current_skeleton_idx]

        # Calculate video frame
        video_frame = int(current_skeleton_frame * (self.video_fps / self.skeleton_fps))

        # Get frame from video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
        ret, frame = self.cap.read()
        if not ret:
            print(f"Failed to read frame {video_frame} from video")
            self.current_skeleton_idx = 0
            return

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Clear and redraw
        self.ax.clear()
        self.ax.imshow(frame_rgb)
        self.ax.axis('off')

        # Draw skeletons
        if current_skeleton_frame in self.skeletons:
            people = self.skeletons[current_skeleton_frame]

            # Rainbow colors for people
            num_people = len(people)
            colors = plt.cm.rainbow(np.linspace(0, 1, max(num_people, 1)))

            if self.mode in ["select", "reselect", "select_new"]:
                # Draw all people for selection with person labels
                for idx, person in enumerate(people):
                    if person.shape == (17, 2) and not np.isnan(person).all():
                        color = colors[idx][:3]
                        self.draw_skeleton_matplotlib(self.ax, person, color,
                                                     label=f'Person {idx}')

            elif self.mode in ["track", "label"]:
                # Find current person in tracked path
                current_person_idx = None
                for frame_num, person_idx in self.tracked_path:
                    if frame_num == current_skeleton_frame:
                        current_person_idx = person_idx
                        break

                # Draw all people
                for idx, person in enumerate(people):
                    if person.shape == (17, 2) and not np.isnan(person).all():
                        if idx == current_person_idx:
                            # Tracked person - color based on confirmation status
                            if current_skeleton_frame in self.tracking_confirmed:
                                if self.tracking_confirmed[current_skeleton_frame]:
                                    color = '#4CAF50'  # Green for confirmed
                                    status = "TRACKED ✓"
                                else:
                                    color = '#FFC107'  # Yellow for needs confirmation
                                    status = "TRACKED ?"
                            else:
                                color = '#F44336'  # Red for lost
                                status = "LOST ✗"

                            self.draw_skeleton_matplotlib(self.ax, person, color,
                                                         linewidth=3, label=status)
                        else:
                            # Other people - dimmed gray
                            self.draw_skeleton_matplotlib(self.ax, person, '#757575',
                                                         linewidth=1, alpha=0.3)

        # Update title
        title = f"Frame {self.current_skeleton_idx + 1}/{self.n_skeleton_frames} | Mode: {self.mode.upper()}"
        if self.is_labeling and self.current_behavior is not None:
            title += f" | Labeling: {self.behaviors[self.current_behavior]}"
        self.ax.set_title(title, fontsize=14, weight='bold', pad=10)

        # Force tight layout and update canvas
        try:
            self.fig.tight_layout()
        except:
            pass

        self.canvas.draw()
        self.canvas.flush_events()

        # Update status bar
        self.update_status_bar()

    def update_status_bar(self):
        """Aktualizuje status bar s informacemi"""
        current_skeleton_frame = self.skeleton_frames[self.current_skeleton_idx]

        # Frame info
        self.frame_label.config(text=f"Frame: {self.current_skeleton_idx + 1}/{self.n_skeleton_frames}")

        # Mode info
        mode_text = self.mode.upper()
        if self.mode == "select":
            mode_text += " - Vyber osobu k sledování"
        elif self.mode == "track":
            mode_text += " - Kontrola sledování"
        elif self.mode == "label":
            mode_text += " - Označování chování"
        self.mode_label.config(text=f"Mode: {mode_text}")

        # Playback status
        play_status = "⏸ PAUSED" if self.paused else "▶ PLAYING"
        self.play_label.config(text=play_status)

        # Labeling status
        if self.is_labeling and self.current_behavior is not None:
            label_text = f"🏷 Labeling: {self.behaviors[self.current_behavior]} (from frame {self.labeling_start_frame})"
            self.label_status.config(text=label_text, foreground=self.behavior_colors[self.current_behavior])
        else:
            self.label_status.config(text="")

        # Segments count
        self.segments_label.config(text=f"Saved segments: {len(self.labels['segments'])}")

    def create_ui(self):
        """Vytvoří hlavní UI okno s matplotlib canvas"""
        self.root = tk.Tk()
        self.root.title("Improved Person Tracker")
        self.root.geometry("1600x1000")

        # Styling
        style = ttk.Style()
        style.theme_use('clam')

        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top: Matplotlib canvas
        canvas_frame = ttk.Frame(main_container)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig = Figure(figsize=(14, 8), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('white')  # Bílé pozadí místo černého
        self.ax.axis('off')

        # Add initial placeholder
        self.ax.text(0.5, 0.5, 'Loading...', ha='center', va='center',
                    fontsize=20, transform=self.ax.transAxes)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Draw initial canvas
        self.fig.tight_layout()
        self.canvas.draw()

        # Bottom: Status bar and controls
        bottom_frame = ttk.Frame(main_container)
        bottom_frame.pack(fill=tk.X, pady=(5, 0))

        # Status bar
        status_frame = ttk.LabelFrame(bottom_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 5))

        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=tk.X)

        self.frame_label = ttk.Label(status_grid, text="Frame: 0/0", font=('Arial', 10))
        self.frame_label.grid(row=0, column=0, padx=10, sticky=tk.W)

        self.mode_label = ttk.Label(status_grid, text="Mode: SELECT", font=('Arial', 10))
        self.mode_label.grid(row=0, column=1, padx=10, sticky=tk.W)

        self.play_label = ttk.Label(status_grid, text="⏸ PAUSED", font=('Arial', 10))
        self.play_label.grid(row=0, column=2, padx=10, sticky=tk.W)

        self.segments_label = ttk.Label(status_grid, text="Saved segments: 0", font=('Arial', 10))
        self.segments_label.grid(row=0, column=3, padx=10, sticky=tk.W)

        self.label_status = ttk.Label(status_grid, text="", font=('Arial', 10, 'bold'))
        self.label_status.grid(row=1, column=0, columnspan=4, pady=5, sticky=tk.W)

        # Controls
        controls_frame = ttk.LabelFrame(bottom_frame, text="Controls & Instructions", padding=10)
        controls_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook for different modes
        notebook = ttk.Notebook(controls_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Selection mode tab
        select_tab = ttk.Frame(notebook, padding=10)
        notebook.add(select_tab, text="Selection Mode")

        select_text = """VÝBĚR OSOBY K SLEDOVÁNÍ:

• 0-9: Vybrat osobu 0-9
• A-Z (a-z): Vybrat osobu 10-35 (A=10, B=11, ... Z=35)
• ESC: Ukončit aplikaci"""

        ttk.Label(select_tab, text=select_text, font=('Courier', 10),
                 justify=tk.LEFT).pack(anchor=tk.W)

        # Tracking mode tab
        track_tab = ttk.Frame(notebook, padding=10)
        notebook.add(track_tab, text="Tracking Mode")

        track_text = """KONTROLA A ÚPRAVA SLEDOVÁNÍ:

• SPACE: Přehrát/Pozastavit
• A / D: Předchozí/Další frame
• Y: Potvrdit sledování v aktuálním framu
• N: Označit jako špatnou osobu
• R: Znovu vybrat osobu pro tento frame
• P: Sledovat NOVOU osobu od tohoto framu
• L: Začít označování chování
• ESC: Ukončit"""

        ttk.Label(track_tab, text=track_text, font=('Courier', 10),
                 justify=tk.LEFT).pack(anchor=tk.W)

        # Labeling mode tab
        label_tab = ttk.Frame(notebook, padding=10)
        notebook.add(label_tab, text="Labeling Mode")

        label_text = """OZNAČOVÁNÍ CHOVÁNÍ:

• SPACE: Přehrát/Pozastavit
• A / D: Předchozí/Další frame
• 1: Začít označovat "nothing/walking" (zelená)
• 2: Začít označovat "suspicious" (žlutá)
• 3: Začít označovat "running/panic" (červená)
• ENTER: Ukončit aktuální segment
• S: Uložit všechna označení
• T: Zpět do tracking mode
• ESC: Ukončit"""

        ttk.Label(label_tab, text=label_text, font=('Courier', 10),
                 justify=tk.LEFT).pack(anchor=tk.W)

        # Saved segments display
        segments_display = ttk.LabelFrame(controls_frame, text="Recent Segments", padding=10)
        segments_display.pack(fill=tk.X, pady=(10, 0))

        self.segments_text = tk.Text(segments_display, height=4, font=('Courier', 9),
                                    state=tk.DISABLED, background='#f0f0f0')
        self.segments_text.pack(fill=tk.BOTH, expand=True)

        # Bind keyboard events
        self.root.bind('<Key>', self.on_key_press)

        # Force initial draw
        print("Initializing display...")
        self.root.update_idletasks()  # Process pending UI updates

        # Small delay to ensure UI is ready
        self.root.after(100, self._delayed_init)

    def _delayed_init(self):
        """Delayed initialization after UI is ready"""
        print("Drawing first frame...")
        # Update display
        self.update_display()

        # Force canvas refresh
        if self.canvas is not None:
            self.canvas.draw()
            self.canvas.flush_events()

        print("✓ Display initialized")

        # Start auto-update loop
        self.auto_update()

    def auto_update(self):
        """Auto-update loop pro animaci"""
        if not self.paused and self.mode in ["track", "label"]:
            self.current_skeleton_idx = (self.current_skeleton_idx + 1) % self.n_skeleton_frames
            self.update_display()

        # Schedule next update
        if self.root is not None:
            self.root.after(100, self.auto_update)  # 10 FPS

    def on_key_press(self, event):
        """
        Handler pro klávesové zkratky

        Args:
            event: tkinter key event
        """
        key = event.char.lower()
        keysym = event.keysym

        # Helper: map key to person index
        def key_to_person_idx(k):
            if k.isdigit():
                return int(k)
            if k.isalpha():
                return 10 + (ord(k.upper()) - ord('A'))
            return None

        # ESC to exit
        if keysym == 'Escape':
            if self.mode in ["reselect", "select_new"]:
                self.mode = "track"
                self.update_display()
            else:
                self.root.quit()
            return

        # Mode-specific controls
        if self.mode == "select":
            person_idx = key_to_person_idx(key)
            if person_idx is not None:
                current_skeleton_frame = self.skeleton_frames[self.current_skeleton_idx]
                people = self.skeletons[current_skeleton_frame]
                if person_idx < len(people):
                    self.selected_person_idx = person_idx
                    print(f"Selected person {person_idx}")
                    self.auto_track_person()
                    self.mode = "track"
                    self.current_skeleton_idx = 0
                    self.update_display()

        elif self.mode == "reselect":
            person_idx = key_to_person_idx(key)
            if person_idx is not None:
                current_skeleton_frame = self.skeleton_frames[self.current_skeleton_idx]
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
                    self.update_display()

        elif self.mode == "select_new":
            person_idx = key_to_person_idx(key)
            if person_idx is not None:
                current_skeleton_frame = self.skeleton_frames[self.current_skeleton_idx]
                people = self.skeletons[current_skeleton_frame]
                if person_idx < len(people):
                    # Clear previous tracking from this point forward
                    self.tracked_path = [(f, p) for f, p in self.tracked_path if f < current_skeleton_frame]

                    # Add this person as the new tracked person
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
                    self.update_display()

        elif self.mode == "track":
            if key == ' ':  # SPACE
                self.paused = not self.paused
            elif key == 'a':  # Previous frame
                self.current_skeleton_idx = max(0, self.current_skeleton_idx - 1)
                self.update_display()
            elif key == 'd':  # Next frame
                self.current_skeleton_idx = min(self.n_skeleton_frames - 1, self.current_skeleton_idx + 1)
                self.update_display()
            elif key == 'y':  # Confirm tracking
                current_skeleton_frame = self.skeleton_frames[self.current_skeleton_idx]
                self.tracking_confirmed[current_skeleton_frame] = True
                print(f"Confirmed tracking for frame {current_skeleton_frame}")
                self.update_display()
            elif key == 'n':  # Wrong person
                current_skeleton_frame = self.skeleton_frames[self.current_skeleton_idx]
                self.tracking_confirmed[current_skeleton_frame] = False
                print(f"Marked frame {current_skeleton_frame} as wrong person")
                self.update_display()
            elif key == 'r':  # Re-select person
                self.mode = "reselect"
                self.update_display()
            elif key == 'p':  # Track new person
                self.mode = "select_new"
                self.update_display()
            elif key == 'l':  # Start labeling
                self.mode = "label"
                self.update_display()

        elif self.mode == "label":
            if key == ' ':  # SPACE
                self.paused = not self.paused
            elif key == 'a':  # Previous frame
                self.current_skeleton_idx = max(0, self.current_skeleton_idx - 1)
                self.update_display()
            elif key == 'd':  # Next frame
                self.current_skeleton_idx = min(self.n_skeleton_frames - 1, self.current_skeleton_idx + 1)
                self.update_display()
            elif key in ['1', '2', '3']:  # Start labeling behavior
                behavior = int(key) - 1
                if not self.is_labeling:
                    current_skeleton_frame = self.skeleton_frames[self.current_skeleton_idx]
                    self.is_labeling = True
                    self.labeling_start_frame = current_skeleton_frame
                    self.current_behavior = behavior
                    print(f"Started labeling {self.behaviors[behavior]} from frame {current_skeleton_frame}")
                    self.update_display()
            elif keysym == 'Return' and self.is_labeling:  # ENTER - end segment
                current_skeleton_frame = self.skeleton_frames[self.current_skeleton_idx]
                self.labels["segments"].append({
                    "start": self.labeling_start_frame,
                    "end": current_skeleton_frame,
                    "behavior": self.current_behavior
                })
                print(f"Saved segment: frames {self.labeling_start_frame}-{current_skeleton_frame}")
                self.is_labeling = False
                self.current_behavior = None
                self.update_display()
                self.update_segments_display()
            elif key == 's':  # Save
                self.save_labels()
            elif key == 't':  # Back to tracking
                self.mode = "track"
                self.update_display()

    def update_segments_display(self):
        """Aktualizuje zobrazení uložených segmentů"""
        self.segments_text.config(state=tk.NORMAL)
        self.segments_text.delete(1.0, tk.END)

        if len(self.labels["segments"]) > 0:
            for segment in self.labels["segments"][-5:]:  # Last 5 segments
                text = f"Frames {segment['start']:4d}-{segment['end']:4d}: {self.behaviors[segment['behavior']]}\n"
                self.segments_text.insert(tk.END, text)
        else:
            self.segments_text.insert(tk.END, "No segments saved yet")

        self.segments_text.config(state=tk.DISABLED)

    def save_labels(self):
        """
        Uloží tracked path a labels

        Returns:
            None (ukládá do JSON a NPZ souborů)
        """
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

        # Show success message
        messagebox.showinfo("Save Successful",
                          f"Data saved successfully!\n\n"
                          f"Person ID: {person_save_id}\n"
                          f"Segments: {len(self.labels['segments'])}\n"
                          f"Sequence length: {len(skeleton_sequence)}")

    def run(self):
        """Spustí aplikaci"""
        self.create_ui()
        self.root.mainloop()

        # Cleanup
        self.cap.release()


# Usage
if __name__ == "__main__":
    video_path = "data/Motion_Emotion/002.mp4"
    skeleton_base_dir = "skeletons_yolo_11"

    tracker = ImprovedPersonTracker(video_path, skeleton_base_dir)
    tracker.run()
