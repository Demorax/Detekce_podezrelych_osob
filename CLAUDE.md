# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Human behavior detection and classification system using pose estimation from video surveillance footage. Processes the Motion Emotion Dataset to detect people, extract skeletal keypoints, track individuals across frames, and classify behaviors into three categories: walking/normal, suspicious, and running/panic.

## Core Architecture

### Four-Stage Pipeline

1. **Frame Extraction** (`extract_frames.py`)
   - Extracts frames from videos at 2 FPS
   - Input: `data/Motion_Emotion/*.mp4` (31 videos)
   - Output: `frames_0.5/`

2. **Super-Resolution Enhancement**
   - ESPCN 4x upscaling using `models/super_resolution/ESPCN_x4.pb`
   - Additional preprocessing: CLAHE, histogram equalization, gamma correction (γ=0.8)
   - Output: `frames_0.5_upscaled/`

3. **Two-Stage Detection Pipeline** (`extract_skeletons_two_stage.ipynb` - CURRENT APPROACH)
   - **Stage 1 - YOLO Person Detection:**
     - Model: YOLO11x (`models/yolo/yolo11x.pt`)
     - Multiple detection passes with varying confidence thresholds (0.15-0.35), image sizes (1024-2048px), and IoU thresholds
     - Runs on 4 image variants (original, CLAHE, histogram equalized, gamma corrected)
   - **Stage 2 - ViTPose Skeleton Extraction:**
     - Model: ViTPose-Huge on CrowdPose dataset
     - Checkpoint: `models/vitpose/vitpose-h-multi-crowdpose.pth` (2.4GB)
     - Config: `models/vitpose/configs/ViTPose_huge_crowdpose_256x192_without_training.py`
     - Custom MMPose components in `models/vitpose/models/`
     - Output: 14 or 17 keypoints per person
   - Output: `.npy` files in `skeletons_yolo_11_upscaled_2/` with shape (num_people, 17, 2)

4. **Tracking & Labeling** (`label_skeletons.py`)
   - Interactive GUI for tracking single person across frames
   - Automatic centroid-based tracking with manual correction capability
   - Behavior annotation: walking/normal (0), suspicious (1), running/panic (2)
   - Output: `labeled_behaviors/*.json` (tracking metadata) and `*.npz` (skeleton + behavior sequences)

### Behavior Classification Models

- **MLP Frame Classifier** (`clean_detekce_osob.ipynb`): Simple feedforward network, 67% accuracy, saved as `frame_classifier_mlp.keras`
- **LSTM Sequence Classifier** (`detekce_novy_pristub.ipynb`): 2-layer LSTM with masking for temporal analysis, saved as `best_lstm_behavior_model.h5` or `best_lstm_behavior_model/`

## Running the Pipeline

```bash
# 1. Extract frames from videos
python extract_frames.py

# 2. Extract skeletons (two-stage approach - RECOMMENDED)
jupyter notebook extract_skeletons_two_stage.ipynb
# Run all cells to process frames with super-resolution + two-stage detection

# 3. Track and label behaviors interactively
python label_skeletons.py

# 4. Train behavior classifier
jupyter notebook detekce_novy_pristub.ipynb

# 5. Visualize skeleton detection results
python test_skeletons.py
```

## Key Technical Details

### ViTPose Custom Integration

- Custom MMPose registry components in `models/vitpose/models/`:
  - ViT backbone: `backbone/vit.py`
  - TopDown detector: `detectors/top_down.py`
  - Heatmap head: `head/topdown_heatmap_simple_head.py`
  - Builder: `builder.py`
- **Important modification in `vit.py`**: Line with `super().init_weights(pretrained, patch_padding=self.patch_padding)` is commented out to avoid training pipeline dependencies
- Config files modified to exclude training-specific transformations

### OpenCV CUDA Support

This project requires OpenCV built with CUDA support for GPU-accelerated super-resolution. Installation steps are documented in `materiály.md`:

1. Remove CPU-only OpenCV wheels
2. Copy custom-built `cv2.pyd` to Python site-packages
3. Copy OpenCV DLLs, CUDA 12.2 DLLs, and cuDNN 8.9.7 DLLs to site-packages
4. Verify CUDA support: `cv2.cuda.getCudaEnabledDeviceCount()` should return > 0

### Data Flow

```
MP4 Videos → extract_frames.py → JPEG Frames (2 FPS)
           → Super-Resolution (ESPCN 4x) → Enhanced Frames
           → YOLO11x (person detection) → Bounding boxes
           → ViTPose-Huge (pose estimation) → Skeleton keypoints (.npy)
           → label_skeletons.py (tracking + labeling) → Labeled sequences (.json + .npz)
           → LSTM training → Behavior classifier (.h5)
```

## Known Issues and Limitations

1. **Severe class imbalance**: Training data is ~96% walking, ~4% running, ~0% suspicious behaviors
2. **Small labeled dataset**: Only 4 person sequences (196 total frames) currently labeled
3. **Memory constraints**: Large models (6GB+ for ViTPose) can cause OOM errors on machines with limited VRAM
4. **Hardcoded paths**: Many scripts use relative paths that may need adjustment
5. **No requirements file**: Dependencies must be manually installed

## Dependencies

Core libraries (no requirements.txt exists - install manually):
- PyTorch (with CUDA support)
- TensorFlow 2.10+
- MMPose, MMEngine
- ultralytics (YOLO)
- OpenCV (custom CUDA build)
- NumPy, SciPy, scikit-learn
- Matplotlib, Seaborn
- tkinter (for GUI)
- Pillow, tqdm

Hardware: CUDA-capable GPU required for efficient processing.

## Git Notes

- Currently on `master` branch (no main branch configured)
- Many generated files (.npy, .jpg, model weights) are tracked in git
- Untracked directories contain output data: `annotation/`, `skeletons_yolo_11_upscaled_2/`, etc.
- Recent commits indicate ongoing work on two-stage detection pipeline and memory optimization

## Alternative Approaches in Codebase

`extract_skeletons.py` implements an earlier single-stage approach using YOLO pose models directly (YOLOv8/11-pose). It includes sophisticated duplicate removal using centroid distances and skeleton feature similarity. The current preferred approach is the two-stage pipeline in `extract_skeletons_two_stage.ipynb`.