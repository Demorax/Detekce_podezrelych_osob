# Human Motion Analysis in Video Using Neural Networks for Spatiotemporal Processing

> Czech version: [README.md](README.md)

## Project Goal

**Main goal:** Detection of threats and suspicious individuals in crowds using automated analysis of surveillance camera footage.

This project implements a system for analyzing human movement in video sequences, combining:
- **Spatial analysis** - person detection and skeleton extraction using convolutional neural networks (RT-DETR, ViTPose)
- **Temporal analysis** - behavior classification using LSTM for processing motion time sequences

**Tasks solved:**
1. **Normal vs. abnormal motion detection** - classifying human behavior (normal walking, suspicious behavior, running/panic)
2. **Individual pose analysis** - extracting and evaluating skeleton keypoints (17 COCO keypoints)
3. **Person tracking in crowds** - tracking individuals across frames in video
4. **Suspicious object detection** - identifying weapons and dangerous objects (in progress)

**Practical applications:**
- Automated surveillance systems (CCTV)
- Real-time security threat detection
- Crowd behavior analysis
- Forensic analysis of security camera footage

---

## Project Structure

### Main Scripts and Notebooks

#### Data Processing and Extraction
- **`extract_frames.py`** - Frame extraction from videos (2 FPS)
- **`extract_skeletons_two_stage.ipynb`** - Main pipeline for two-stage detection:
  - Stage 1: RT-DETR person detection with preprocessing (CLAHE, histogram eq, gamma)
  - Stage 2: ViTPose skeleton extraction (17 keypoints)
  - Implements `better_detection()` method with multiple configs
  - Saves skeletons as `.npy` files

#### Visualization and Labeling
- **`visualization_helper.py`** - Tool for visualizing detections and skeletons:
  - 20 predefined colors for distinguishing individuals
  - Rendering bounding boxes and skeletal keypoints
  - COCO skeleton connections (17 keypoints)
  - Visualization support with/without skeletons

- **`label_skeletons.py`** - Interactive GUI for behavior annotation:
  - Tracking individuals across frames
  - Manual correction of automatic tracking
  - Behavior classification: normal walking (0), suspicious (1), running/panic (2)
  - Export to `.json` (metadata) and `.npz` (skeletons + labels)

- **`label_skeletons_improved.py`** - Improved labeler version (in progress)

#### Testing and Comparison
- **`test_rtdetr.py`** - RT-DETR vs YOLO11x comparison:
  - Fair comparison (same conf, imgsz=640)
  - Graph and visualization generation
  - Statistics (precision, recall, FPS)

### `models/` Directory

Contains all models used and their configurations:

#### RT-DETR (Real-Time Detection Transformer)
```
models/rtdetr/
├── configs/
│   ├── rtdetr/
│   │   ├── rtdetr_r18vd_6x_coco.yml      # R18 backbone (faster)
│   │   ├── rtdetr_r101vd_6x_coco.yml     # R101 backbone (more accurate)
│   │   └── include/
│   │       ├── dataloader.yml
│   │       └── optimizer.yml
│   └── rtdetrv2/
│       ├── rtdetrv2_r18vd_120e_coco.yml  # RT-DETRv2 R18
│       └── rtdetrv2_r101vd_6x_coco.yml   # RT-DETRv2 R101
├── src/                                   # RT-DETR implementation
└── *.pth                                  # Pretrained weights
```

**Available checkpoints:**
- `rtdetrv2_r101vd_6x_coco_from_paddle.pth` - **RT-DETRv2 R101** (currently used)
  - Used in `extract_skeletons_two_stage.ipynb` (production pipeline)
  - Used in `test_rtdetr.py` (comparison with YOLO)
  - **ResNet-101 backbone** - higher accuracy, slower
  - **Reason for use:** Dataset creation requires maximum accuracy

- `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth` - **RT-DETRv2 R18** (prepared for deployment)
  - **ResNet-18 backbone** - faster, smaller model
  - **Reason for preparation:** For future real-time deployment where speed is critical
  - Commented out in `test_rtdetr.py` for easy swapping

**Model selection notes:**
- **Dataset creation (now):** R101 - we need the most accurate skeletons for LSTM training
- **Production deployment (future):** R18 - faster inference for real-time CCTV analysis

#### ViTPose (Vision Transformer Pose Estimation)
```
models/vitpose/
├── configs/
│   ├── ViTPose_huge_crowdpose_256x192_without_training_v3.py  # Production config
│   ├── ViTPose_huge_crowdpose_256x192.py
│   └── _base_/
│       ├── datasets/crowdpose.py
│       └── default_runtime.py
├── models/                                # Custom MMPose components
│   ├── backbone/vit.py                    # ViT backbone
│   ├── detectors/top_down.py              # TopDown detector
│   └── head/topdown_heatmap_simple_head.py
└── vitpose-h-multi-crowdpose.pth         # ViTPose-Huge weights (2.4GB)
```

**Important:**
- Config `*_without_training_v3.py` removes dependencies on the training pipeline
- Custom components in `models/` for integration with MMPose

#### YOLO (Ultralytics)
```
models/yolo/
├── yolo11x.pt          # YOLO11x detection model
└── yolo11x-pose.pt     # YOLO11x pose estimation (1-stage, deprecated)
```

#### Super-Resolution
```
models/super_resolution/
└── ESPCN_x4.pb         # ESPCN model for 4x upscaling
```

#### Weapon Detection (in progress)
```
models/weapon_detection_guns/    # Dataset: firearms
models/weapon_detection_new/     # Dataset: various suspicious objects
```

### Data Directories

```
frames_0.5/              # Extracted frames (2 FPS)
frames_0.5_upscaled/     # Upscaled frames (ESPCN 4x)
skeletons_yolo_11_upscaled_2/  # Skeletons from RT-DETR + ViTPose
├── *.npy                # Skeleton data (N, 17, 2 or 3)
└── visualizations/      # Detection visualizations
labeled_behaviors/       # Annotated data for LSTM training
├── *.json               # Tracking metadata
└── *.npz                # Skeletons + behavior labels
```

### Behavior Models

```
best_lstm_behavior_model.h5      # Trained LSTM classifier
best_lstm_behavior_model/        # SavedModel format
frame_classifier_mlp.keras       # MLP baseline (67% accuracy)
```

---

## 1. Dataset Creation
To train a model for suspicious person detection, a quality dataset is required. Since no such public dataset exists, I'm creating my own from the Motion Emotion dataset. For maximum accuracy I use state-of-the-art models, even though they are more computationally demanding.
1. Frame extraction from video
   - `extract_frames.py` captures one frame every 0.5 seconds from videos
2. Image resolution enhancement
   - Upscaling to 4K using ESPCN_x4.pb
   - For deployment, quality will likely be max 2K
3. Person detection model
4. Suspicious object detection model (not yet implemented)
5. Skeleton extraction model
6. Dataset labeling
   - Using `label_skeletons.py`
   - Manually tracking a person and describing their behavior
   ![label_skeletons_showcase](docs/label_skeletons_showcase.png)
7. Custom neural network

## 2. First Attempt - Single-Stage Detection (1 stage)
I tested YOLO models `yolo8x-pose.pt` and `yolo11x-pose.pt` for simultaneous person detection and skeleton extraction.
This approach uses a single model that detects both persons and skeleton keypoints in one step (1-stage detection).

### Settings and Results:
- **Confidence threshold**: 0.5–0.6 for both person and keypoint detection
- **Performance**: The model performs well in simpler scenes with fewer people

### Problems:
Despite the best possible settings, the model fails in crowded scenes where people overlap or are in close proximity.
In these situations:
- Some people are missed (false negatives)
- Inaccurate skeleton keypoint detection
- Keypoint swapping between nearby individuals

See examples below:
![test_76_1.png](test_76_1.png)
![test_76_1_upscaled_2.png](test_76_1_upscaled_2.png)

## 3. Second Attempt - Two-Stage Detection (2 stage)
From the first attempt I found that the YOLO model is excellent for person detection but has problems with accurate skeleton extraction in crowded scenes.
I therefore decided to use two-stage detection:

**Stage 1**: Person detection using YOLO (fast and accurate bounding box localization)
**Stage 2**: Skeleton extraction using ViTPose (accurate keypoint detection inside each bounding box separately)

This method allows leveraging the advantages of both models — the speed and robustness of YOLO for person detection and the high accuracy of ViTPose for skeleton keypoints.

### 3.1. YOLO 11x and ViTPose - Basic Setup
**Settings:**
- **YOLO 11x**: Confidence threshold 0.5–0.6 for person detection
- **ViTPose**: Keypoint extraction from detected bounding boxes

**Results:**
The combination of these two models provides significantly better results than single-stage detection, especially in crowded scenes.
![001_0076_bboxes](reference_img/yolo_vitpose/normalni/001_0076_bboxes.jpg)
![001_0076](reference_img/yolo_vitpose/normalni/001_0076.jpg)

**Advanced YOLO setup with preprocessing techniques:**

To improve person detection in various scenes, I use the `better_detection` method, which combines multiple preprocessing techniques with different YOLO configurations:

**Preprocessing techniques:**
1. **Original image** - no modifications
2. **Shadow Suppression** - shadow removal using dilation and median filter
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) - adaptive histogram equalization in LAB color space
4. **Histogram Equalization** - global histogram equalization in YUV space
5. **Gamma Correction** (γ=1.3) - brightness adjustment

**YOLO configurations for each technique:**
```python
configs = [
    {'conf': 0.65, 'imgsz': 1280, 'iou': 0.6},  # High confidence, medium resolution
    {'conf': 0.60, 'imgsz': 1024, 'iou': 0.6},  # Balanced configuration
    {'conf': 0.60, 'imgsz': 1536, 'iou': 0.6},  # Higher resolution for details
    {'conf': 0.55, 'imgsz': 1792, 'iou': 0.7},  # High resolution, lower confidence
    {'conf': 0.40, 'imgsz': 2048, 'iou': 0.6},  # Maximum resolution, low confidence
]
```

**How it works:**
- Each preprocessing creates an image variant optimized for different lighting conditions
- Detection is performed on each variant with the corresponding configuration
- The system selects results with the highest overall confidence score
- NMS (Non-Maximum Suppression) is applied to final detections to remove duplicates

**Advantages of this approach:**
- Robust detection in various lighting conditions
- Better detection of people in shadow or overexposed areas
- Higher recall while maintaining acceptable precision


#### 3.1.1. Optimization: Shadow Removal
The first optimization step was removing shadows from images, which sometimes caused false detections.

**Before shadow removal:**
![before_shadows](reference_img/before_removing_shadows/002_0000.jpg)

**After shadow removal:**
![after_shadows](reference_img/before_removing_shadows/002_0000_removed_shadows.jpg)

#### 3.1.2. Optimization: SAHI (Slicing Aided Hyper Inference)
The SAHI technique splits the image into smaller parts and detects objects in each part separately, helping with detection of small or distant people.
After testing, SAHI was found not to provide sufficient improvement for the added computational cost, so it was removed.

**Without SAHI:**
![without_sahi](reference_img/before_using_sahi/001_0076_bboxes.jpg)

**With SAHI:**
![with_sahi](reference_img/using_sahi/001_0076_bboxes.jpg)

#### 3.1.3. Optimization: NMS (Non-Maximum Suppression)
The NMS algorithm removes duplicate detections of the same person by suppressing overlapping bounding boxes with lower confidence scores.

**Without NMS:**
![without_nms](reference_img/yolo_vitpose/normalni/001_0076_bboxes.jpg)

**With NMS:**
![with_nms](reference_img/yolo_vitpose/nms/001_0076_bboxes.jpg)

### 3.2. Detector Comparison: RT-DETR v2 vs YOLO 11
To improve person detection I compared two state-of-the-art object detectors:
- **YOLO 11x**: Fast, real-time detector
- **RT-DETR v2**: Transformer-based detector with potentially higher accuracy

**Test notes:**
This test compares the **raw performance of both models** without any optimizations:
- **No NMS** - only built-in NMS from models is used
- **No preprocessing** - no CLAHE, histogram equalization, or gamma correction
- **No multiple configs** - only one configuration per model
- **Standard inference** - original images, same parameters (conf, imgsz=640)

This is a fair comparison of the baseline capabilities of both models.

#### Test 1: Confidence threshold 0.5

**RT-DETR R101 (current production model):**

*Visual comparison:*
![R101_comparison](reference_img/yolo_vs_rtdetr/confidence_05/R101/rtdetr_vs_yolo_comparison.jpg)

*Performance metrics:*
![R101_graphs](reference_img/yolo_vs_rtdetr/confidence_05/R101/rtdetr_vs_yolo_graphs.jpg)

**RT-DETR R18 (for reference - faster variant):**

*Visual comparison:*
![R18_comparison](reference_img/yolo_vs_rtdetr/confidence_05/R18/rtdetr_vs_yolo_comparison.jpg)

*Performance metrics:*
![R18_graphs](reference_img/yolo_vs_rtdetr/confidence_05/R18/rtdetr_vs_yolo_graphs.jpg)

**Conclusions:**
Both RT-DETR models (R101 and R18) consistently show **higher confidence scores** for all detections compared to YOLO 11x at the same confidence threshold of 0.5.
This means:
- I can **raise the confidence threshold** (e.g., to 0.7–0.8) and still maintain or improve recall
- A higher threshold helps **eliminate false positive detections**
- RT-DETR is also **faster** than YOLO 11x at inference

Thanks to these advantages I can:
1. Use a **stricter confidence threshold** for cleaner detections
2. Use a **larger and more accurate ViTPose model** in the second stage, since RT-DETR saves computation in the first stage
3. Achieve **better overall pipeline accuracy** without performance compromises

#### Test 2: Confidence threshold 0.3 (for comparison)

For comparison with a lower confidence threshold that allows detecting more people (higher recall):

**Note:** All results in this folder use the **RT-DETR R18** model.

**Visual comparison:**
![confidence_03_comparison](reference_img/yolo_vs_rtdetr/confidence_03/rtdetr_vs_yolo_comparison.jpg)

**Performance metrics:**
![confidence_03_graphs](reference_img/yolo_vs_rtdetr/confidence_03/rtdetr_vs_yolo_graphs.jpg)

**Conclusions:**
Even at the lower threshold of 0.3 and with the lighter R18 model, RT-DETR consistently maintains higher confidence scores and detects more people with greater certainty. This confirms that even the faster RT-DETR variant provides more robust detections than YOLO 11x.

### 3.3. RT-DETR and ViTPose
The combination of RT-DETR for person detection and ViTPose for skeleton extraction is the final solution for the production pipeline.

**Scenario 1: Fewer people (3 people close together)**

RT-DETR handles detection well even when people are close to each other:
![002_0000_bboxes](reference_img/rtdetr_vitpose/002_0000_bboxes.jpg)

**Scenario 2: Large crowd**

RT-DETR detects nearly all people even in a crowded scene. When people are very close together (overlapping), minor inaccuracies can occur — this is an area for future improvement:
![001_0076_bboxes](reference_img/rtdetr_vitpose/001_0076_bboxes.jpg)

**Strengths of the RT-DETR + ViTPose combination:**
- Robust person detection even in challenging scenes
- Accurate skeleton extraction thanks to the ViTPose-Huge model
- 3 people close together: handles perfectly
- Large crowds: detects almost all people, but heavily overlapping figures require further optimization

---

## References

### Person Detection

**RT-DETR** (Real-Time Detection Transformer):
```
Lv, W., Xu, S., Zhao, Y., Wang, G., Wei, J., Cui, C., Du, Y., Dang, Q., & Liu, Y. (2023).
DETRs Beat YOLOs on Real-time Object Detection.
arXiv preprint arXiv:2304.08069.
https://arxiv.org/abs/2304.08069
```

**RT-DETRv2**:
```
Lv, W., Zhao, Y., Chang, Q., Huang, K., Wang, G., & Liu, Y. (2024).
RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer.
arXiv preprint arXiv:2407.17140.
https://arxiv.org/abs/2407.17140
```

**YOLO11** (Ultralytics):
```
Jocher, G., Chaurasia, A., & Qiu, J. (2023).
Ultralytics YOLO (Version 11.0.0) [Computer software].
https://github.com/ultralytics/ultralytics
```

### Skeleton Extraction

**ViTPose**:
```
Xu, Y., Zhang, J., Zhang, Q., & Tao, D. (2022).
ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation.
In Advances in Neural Information Processing Systems (NeurIPS).
arXiv preprint arXiv:2204.12484.
https://arxiv.org/abs/2204.12484
```

**MMPose**:
```
MMPose Contributors. (2020).
OpenMMLab Pose Estimation Toolbox and Benchmark.
https://github.com/open-mmlab/mmpose
```

### Dataset

**Motion Emotion Dataset (MED)**:
```
Hossein Mousavi. (2023).
Motion Emotion Dataset (MED) - A dataset for human behavior analysis.
GitHub repository.
https://github.com/hosseinm/med
```

### Super-Resolution

**ESPCN** (Efficient Sub-Pixel Convolutional Neural Network):
```
Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A. P., Bishop, R., ... & Wang, Z. (2016).
Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
```

---
