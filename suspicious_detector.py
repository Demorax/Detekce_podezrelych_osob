"""
Detektor podezřelých osob a objektů s plnou pipeline.

Replikuje detekci a vizualizaci z extract_skeletons_two_stage.ipynb:
- RT-DETR v2 R101 (better_detection s multi-pass: original + shadow_norm + CLAHE + hist_eq + gamma)
- ViTPose-Huge (CrowdPose) - extrakce skeletů
- Vizualizace přes visualization_helper.visualize_detections (boxy + skelety)

Navíc detekuje podezřelé objekty:
- YOLO11x (COCO) - pálka, nůž, nůžky, batoh, taška, kufr
- Custom YOLOv8 (Roboflow Firearms+Knives) - gun, knife

TODO: kladivo (hammer) - není v žádném modelu, vyžaduje custom trénink.
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# RT-DETR
sys.path.insert(0, os.path.join(os.getcwd(), 'models', 'rtdetr'))
from src.core import YAMLConfig

# ViTPose
from mmpose.apis import init_model, inference_topdown
from mmpose.registry import MODELS
from models.vitpose.models.backbone.vit import ViT
from models.vitpose.models.detectors.top_down import TopDown
from models.vitpose.models.head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead

# Vizualizace (pro extract_keypoints visualize=True — používá notebook)
from visualization_helper import visualize_detections

MODELS.register_module(module=ViT, force=True)
MODELS.register_module(module=TopDown, force=True)
MODELS.register_module(module=TopdownHeatmapSimpleHead, force=True)


# ===== Konfigurace =====
_RTDETR_CONFIG = 'models/rtdetr/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml'
_RTDETR_MODEL = 'models/rtdetr/rtdetrv2_r101vd_6x_coco_from_paddle.pth'
_VITPOSE_CONFIG = 'models/vitpose/configs/ViTPose_huge_crowdpose_256x192_without_training_v3.py'
_VITPOSE_MODEL = 'models/vitpose/vitpose-h-multi-crowdpose.pth'
_YOLO_MODEL = 'models/yolo/yolo11x.pt'
_WEAPON_MODEL = 'models/weapon_detection_new/runs/detect/Normal/weights/best.pt'

# COCO třídy YOLO11x relevantní pro podezřelé chování (BEZ person - tu řeší RT-DETR)
COCO_HIGH_THREAT = {
    34: 'baseball bat',
    43: 'knife',
}
COCO_MEDIUM_THREAT = {
    76: 'scissors',
}
COCO_CARRY = {
    24: 'backpack',
    26: 'handbag',
    28: 'suitcase',
}

# Custom weapon model (Roboflow Firearms+Knives)
WEAPON_HIGH_THREAT = {
    0: 'gun',
    1: 'knife',
}

# COCO 17 keypoints connections (pro skeleton vizualizaci)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),                              # hlava
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),                     # ruce
    (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),    # nohy
    (11, 12),                                                    # boky
]


class RTDETRDetector:
    """
    RT-DETR detektor osob s multi-pass better_detection.
    Identický s tříddou v extract_skeletons_two_stage.ipynb.
    """

    def __init__(self, config_path, checkpoint_path, device='cuda'):
        self.device = device

        cfg = YAMLConfig(config_path, resume=checkpoint_path, PResNet={'pretrained': False})
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        cfg.model.load_state_dict(state, strict=False)

        class Model(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                return self.postprocessor(outputs, orig_target_sizes)

        self.model = Model(cfg).to(device).eval()
        self.transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        print(f"  ✓ RT-DETR loaded on {device}")

    def detect(self, image, conf_threshold=0.6, person_class_id=0):
        """Single-pass detekce. Vrací (N, 5) [x1,y1,x2,y2,conf] pro person třídu."""
        if isinstance(image, str):
            im_pil = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            im_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            im_pil = image

        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(self.device)
        im_data = self.transforms(im_pil)[None].to(self.device)

        with torch.no_grad():
            labels, boxes, scores = self.model(im_data, orig_size)

        labels = labels[0].cpu().numpy()
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()

        mask = (labels == person_class_id) & (scores > conf_threshold)
        if mask.sum() == 0:
            return np.array([])
        return np.column_stack([boxes[mask], scores[mask]])

    def better_detection(self, img):
        """
        Multi-pass detekce: zkouší 5 preprocessingů (original, shadow_norm, CLAHE,
        hist_eq, gamma) a vrátí výsledek z té varianty s nejvyšším součtem confidencí.
        Identické chování jako v notebooku.
        """
        # Shadow suppression
        rgb_planes = cv2.split(img)
        result_norm_planes = []
        for plane in rgb_planes:
            dilated = cv2.dilate(plane, np.ones((4, 4), np.uint8))
            bg = cv2.medianBlur(dilated, 7)
            diff = 255 - cv2.absdiff(plane, bg)
            norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            result_norm_planes.append(norm)
        result_norm = cv2.merge(result_norm_planes)

        enhanced_imgs = [img, result_norm]

        # CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_imgs.append(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR))

        # Histogram equalization (na shadow_norm)
        yuv = cv2.cvtColor(result_norm, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        enhanced_imgs.append(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR))

        # Gamma correction (na shadow_norm)
        gamma = 1.3
        enhanced_imgs.append((np.power(result_norm / 255.0, gamma) * 255.0).astype(np.uint8))

        # Vyber variantu s nejvyšším součtem confidencí
        best_score = -1
        best_boxes = np.array([])
        best_idx = -1
        for i, variant in enumerate(enhanced_imgs):
            try:
                result = self.detect(variant, conf_threshold=0.3)
                if len(result) > 0:
                    score = np.sum(result[:, 4])
                    if score > best_score:
                        best_score = score
                        best_boxes = result
                        best_idx = i
            except Exception as e:
                print(f"  ⚠ Variant {i} failed: {e}")
                continue

        if len(best_boxes) == 0:
            return np.array([])

        techniques = ['original', 'shadow_norm', 'CLAHE', 'hist_eq', 'gamma']
        print(f"  best preprocessing: {techniques[best_idx]} (n={len(best_boxes)}, sum_conf={best_score:.1f})")
        return self.nms_boxes(best_boxes, iou_threshold=0.5)

    def nms_boxes(self, boxes, iou_threshold):
        """NMS pro odstranění duplicit."""
        x1, y1, x2, y2, scores = boxes.T
        indices = scores.argsort()[::-1]
        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area_rest = (x2[indices[1:]] - x1[indices[1:]]) * (y2[indices[1:]] - y1[indices[1:]])
            iou = inter / (area_i + area_rest - inter + 1e-6)
            indices = indices[1:][iou < iou_threshold]
        return boxes[keep]


class VitPoseExtractor:
    """
    Extraktor keypoints přes ViTPose-Huge (CrowdPose).
    Identický s třídou v extract_skeletons_two_stage.ipynb.
    """

    def __init__(self, config_file, checkpoint_file, device='cuda'):
        self.model = init_model(config_file, checkpoint_file, device=device)
        print(f"  ✓ ViTPose loaded on {device}")

    def extract_keypoints(self, img, boxes, visualize=False, save_path=None):
        """
        Z bounding boxů osob vrátí skeletony.
        Vrací (N, K, 3) kde K je 14 (CrowdPose) nebo 17 (COCO), poslední dim je [x, y, score].

        Args:
            img: numpy BGR
            boxes: (N, 5) [x1,y1,x2,y2,conf]
            visualize: zda vykreslit bboxy (stage 1) + skelety (stage 2) přes
                       visualization_helper.visualize_detections (používá notebook)
            save_path: cesta pro uložení vizualizace (None = neukládá)
        """
        if boxes is None or len(boxes) == 0:
            return np.array([])

        boxes_xyxy = boxes[:, :4].astype(np.float32)

        # Stage 1 vizualizace (bboxy) — volitelně
        if visualize:
            visualize_detections(
                img, boxes, skeletons=None,
                title="RT-DETR Detections (Stage 1)",
                save_path=save_path.replace('.jpg', '_bboxes.jpg') if save_path else None,
                show=True,
            )

        pose_results = inference_topdown(self.model, img, boxes_xyxy, "xyxy")

        skeletons = []
        for r in pose_results:
            if hasattr(r, 'pred_instances'):
                kp = r.pred_instances.keypoints[0]
                sc = r.pred_instances.keypoint_scores[0]
                skeletons.append(np.concatenate([kp, sc[:, None]], axis=1))

        skeletons_array = np.array(skeletons) if skeletons else np.array([])

        # Stage 2 vizualizace (skelety) — volitelně
        if visualize and len(skeletons_array) > 0:
            visualize_detections(
                img, boxes, skeletons=skeletons_array,
                title="ViTPose Results (Stage 2)",
                save_path=save_path,
                show=True,
            )

        return skeletons_array


class SuspiciousDetector:
    """
    Plná pipeline: persons (RT-DETR multi-pass) + skeletons (ViTPose) + suspicious objekty.

    Args:
        device: 'cuda' nebo 'cpu'
        skip_weapon_model: pokud True, vynechá custom weapon model (kvůli FP)
    """

    def __init__(self, device='cuda', skip_weapon_model=False):
        print(f"\n{'=' * 70}")
        print("LOADING SuspiciousDetector (full pipeline)")
        print('=' * 70)

        self.rtdetr = RTDETRDetector(_RTDETR_CONFIG, _RTDETR_MODEL, device=device)
        self.vitpose = VitPoseExtractor(_VITPOSE_CONFIG, _VITPOSE_MODEL, device=device)

        print(f"  Loading YOLO11x: {_YOLO_MODEL}")
        self.coco = YOLO(_YOLO_MODEL)

        self.weapon = None
        if not skip_weapon_model:
            print(f"  Loading weapon model: {_WEAPON_MODEL}")
            self.weapon = YOLO(_WEAPON_MODEL)

        self.device = device
        self.coco_filter = (
            list(COCO_HIGH_THREAT.keys())
            + list(COCO_MEDIUM_THREAT.keys())
            + list(COCO_CARRY.keys())
        )
        print(f"\n✓ Detector ready")
        print('=' * 70)

    @staticmethod
    def _enhance_variants(img):
        """
        Vrátí 5 image variant pro multi-pass detekci (stejné techniky jako
        RTDETRDetector.better_detection): original, shadow_norm, CLAHE, hist_eq, gamma.
        """
        # Shadow suppression
        rgb_planes = cv2.split(img)
        norm_planes = []
        for plane in rgb_planes:
            dilated = cv2.dilate(plane, np.ones((4, 4), np.uint8))
            bg = cv2.medianBlur(dilated, 7)
            diff = 255 - cv2.absdiff(plane, bg)
            norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            norm_planes.append(norm)
        shadow_norm = cv2.merge(norm_planes)

        variants = [img, shadow_norm]

        # CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
        l = clahe.apply(l)
        variants.append(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR))

        # Histogram equalization (na shadow_norm)
        yuv = cv2.cvtColor(shadow_norm, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        variants.append(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR))

        # Gamma
        gamma = 1.3
        variants.append((np.power(shadow_norm / 255.0, gamma) * 255.0).astype(np.uint8))

        return variants

    @staticmethod
    def _nms(boxes_xyxy_scores, iou_threshold=0.5):
        """NMS na (N, 5) [x1,y1,x2,y2,score]. Vrátí indexy zachovaných."""
        if len(boxes_xyxy_scores) == 0:
            return []
        x1, y1, x2, y2, scores = boxes_xyxy_scores.T
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(int(i))
            if len(order) == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area_rest = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
            iou = inter / (area_i + area_rest - inter + 1e-6)
            order = order[1:][iou < iou_threshold]
        return keep

    def better_object_detection(self, img, conf_obj=0.15, imgsz=1280):
        """
        Multi-pass YOLO11x detekce přes 5 image variant. Sloučí všechny detekce
        (UNION, ne pick-best — různé techniky najdou různé objekty: knife po CLAHE,
        batoh po hist_eq atd.) a aplikuje cross-variant NMS pro odstranění duplicit.

        Returns:
            dict {'high': [...], 'medium': [...], 'carry': [...]} per-class lists
        """
        variants = self._enhance_variants(img)
        per_class_dets = {}  # cls_id -> list of (x1,y1,x2,y2,conf)

        for variant in variants:
            try:
                results = self.coco(
                    variant,
                    conf=conf_obj,
                    classes=self.coco_filter,
                    device=self.device,
                    imgsz=imgsz,
                    verbose=False,
                )
            except Exception as e:
                print(f"  ⚠ YOLO failed on variant: {e}")
                continue
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls = int(b.cls[0])
                    xyxy = b.xyxy[0].cpu().numpy()
                    score = float(b.conf[0])
                    per_class_dets.setdefault(cls, []).append([*xyxy, score])

        # Per-class NMS + classify into threat tiers
        out = {'high': [], 'medium': [], 'carry': []}
        for cls, dets in per_class_dets.items():
            arr = np.array(dets)
            keep_idx = self._nms(arr, iou_threshold=0.5)
            for i in keep_idx:
                box = arr[i, :4]
                score = float(arr[i, 4])
                if cls in COCO_HIGH_THREAT:
                    out['high'].append((box, score, COCO_HIGH_THREAT[cls]))
                elif cls in COCO_MEDIUM_THREAT:
                    out['medium'].append((box, score, COCO_MEDIUM_THREAT[cls]))
                elif cls in COCO_CARRY:
                    out['carry'].append((box, score, COCO_CARRY[cls]))
        return out

    def detect(self, img, conf_obj=0.15, conf_weapon=0.55, imgsz=1280, use_better_obj=False):
        """
        Args:
            img: numpy BGR
            conf_obj: confidence pro YOLO11x objekty (default 0.15 - nízké pro malé pálky)
            conf_weapon: confidence pro weapon model (default 0.55)
            imgsz: image size pro YOLO11x (default 1280)
            use_better_obj: pokud True, použije multi-pass detekci pro objekty
                            (5× pomalejší, ale výrazně víc detekcí v různých světelných
                            podmínkách). Default False — pro single-frame demo.

        Returns:
            dict {
                'person_boxes': (N, 5) [x1,y1,x2,y2,conf],
                'skeletons':    (N, K, 3) nebo prázdné,
                'high':   list of (box, score, label),
                'medium': list of (box, score, label),
                'carry':  list of (box, score, label),
            }
        """
        # 1) Persons - multi-pass RT-DETR (jako v notebooku)
        person_boxes = self.rtdetr.better_detection(img)

        # 2) Skeletons - ViTPose
        skeletons = self.vitpose.extract_keypoints(img, person_boxes) if len(person_boxes) > 0 else np.array([])

        out = {
            'person_boxes': person_boxes,
            'skeletons': skeletons,
            'high': [],
            'medium': [],
            'carry': [],
        }

        # 3) Objekty - single-pass nebo multi-pass
        if use_better_obj:
            obj_out = self.better_object_detection(img, conf_obj=conf_obj, imgsz=imgsz)
            out['high'].extend(obj_out['high'])
            out['medium'].extend(obj_out['medium'])
            out['carry'].extend(obj_out['carry'])
        else:
            coco_results = self.coco(
                img,
                conf=conf_obj,
                classes=self.coco_filter,
                device=self.device,
                imgsz=imgsz,
                verbose=False,
            )
            for r in coco_results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls = int(b.cls[0])
                    xyxy = b.xyxy[0].cpu().numpy()
                    score = float(b.conf[0])
                    if cls in COCO_HIGH_THREAT:
                        out['high'].append((xyxy, score, COCO_HIGH_THREAT[cls]))
                    elif cls in COCO_MEDIUM_THREAT:
                        out['medium'].append((xyxy, score, COCO_MEDIUM_THREAT[cls]))
                    elif cls in COCO_CARRY:
                        out['carry'].append((xyxy, score, COCO_CARRY[cls]))

        # 4) Custom weapon model (volitelně)
        if self.weapon is not None:
            wres = self.weapon(img, conf=conf_weapon, device=self.device, imgsz=640, verbose=False)
            for r in wres:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    cls = int(b.cls[0])
                    xyxy = b.xyxy[0].cpu().numpy()
                    score = float(b.conf[0])
                    label = WEAPON_HIGH_THREAT.get(cls, f'class_{cls}')
                    out['high'].append((xyxy, score, label))

        return out

    def visualize(self, img, detections, save_path=None, show=False, title="Suspicious Detection"):
        """
        Matplotlib vizualizace stejného stylu jako visualize_detections (visualization_helper.py):
        rainbow barvy pro osoby, skeleton spoje + keypoints, nahoře navíc objekty s threat-color rámečkem.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=16, weight='bold')
        ax.axis('off')

        # Distinct rainbow palette pro osoby (stejně jako visualization_helper)
        person_boxes = detections['person_boxes']
        skeletons = detections['skeletons']
        n = len(person_boxes)

        if n > 0:
            distinct = [
                (1, 0, 0), (0, 0.5, 1), (0, 1, 0), (1, 0.5, 0), (0.5, 0, 1),
                (1, 1, 0), (0, 1, 1), (1, 0, 1), (0.5, 1, 0), (1, 0, 0.5),
                (0, 1, 0.5), (0.5, 0.5, 1), (1, 0.5, 0.5), (0.5, 1, 0.5),
                (1, 1, 0.5), (0.5, 0.5, 0.5), (0.75, 0.25, 0), (0, 0.5, 0.5),
                (0.5, 0, 0.5), (0.25, 0.75, 0.5),
            ]
            colors = distinct[:n] if n <= len(distinct) else [tuple(c[:3]) for c in plt.cm.rainbow(np.linspace(0, 1, n))]

            for i, box in enumerate(person_boxes):
                color = colors[i]
                x1, y1, x2, y2 = box[:4].astype(int)
                conf = box[4]
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor=color, linewidth=3, alpha=0.85))
                ax.text(x1, y1 - 6, f'Person {i}: {conf:.2f}',
                        fontsize=9, color='white', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))

                # Skeleton
                if i < len(skeletons):
                    skel = skeletons[i]
                    kp = skel[:, :2]
                    sc = skel[:, 2]
                    for c0, c1 in SKELETON_CONNECTIONS:
                        if c0 >= len(kp) or c1 >= len(kp):
                            continue
                        if sc[c0] > 0.3 and sc[c1] > 0.3:
                            ax.plot([kp[c0, 0], kp[c1, 0]], [kp[c0, 1], kp[c1, 1]],
                                    color=color, linewidth=2, alpha=0.7)
                    for j in range(len(kp)):
                        if sc[j] > 0.3:
                            ax.scatter(kp[j, 0], kp[j, 1], color=color, s=60,
                                       edgecolors='white', linewidth=1.5, alpha=0.9, zorder=10)

        # Suspicious objekty (kreslí se NAD persons, výraznější tloušťka)
        threat_colors_mpl = {
            'high':   (1.0, 0.0, 0.0),   # červená
            'medium': (1.0, 0.55, 0.0),  # oranžová
            'carry':  (1.0, 0.85, 0.0),  # žlutá
        }
        for level in ['carry', 'medium', 'high']:
            color = threat_colors_mpl[level]
            for box, score, label in detections[level]:
                x1, y1, x2, y2 = box.astype(int)
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor=color, linewidth=4, alpha=0.95))
                ax.text(x1, y2 + 18, f'{label}: {score:.2f}',
                        fontsize=11, color='white', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()

    def summary(self, detections):
        return {
            'persons': len(detections['person_boxes']),
            'skeletons': len(detections['skeletons']),
            'high':    [lbl for _, _, lbl in detections['high']],
            'medium':  [lbl for _, _, lbl in detections['medium']],
            'carry':   [lbl for _, _, lbl in detections['carry']],
        }


if __name__ == "__main__":
    detector = SuspiciousDetector()

    test_frames = [
        'frames_0.5_upscaled/001/001_0076.jpg',
        'frames_0.5_upscaled/002/002_0000.jpg',
        'frames_0.5_upscaled/004/004_0000.jpg',
        'frames_0.5_upscaled/004/004_0030.jpg',
        'frames_0.5_upscaled/004/004_0065.jpg',  # tady jsou pálky
    ]

    out_dir = 'reference_img/suspicious_detection'
    os.makedirs(out_dir, exist_ok=True)

    for frame_path in test_frames:
        if not os.path.exists(frame_path):
            print(f"\n⚠ Skipping (not found): {frame_path}")
            continue

        print(f"\n{'=' * 70}")
        print(f"TEST: {frame_path}")
        print('=' * 70)

        img = cv2.imread(frame_path)
        if img is None:
            print("  ⚠ Could not read")
            continue

        detections = detector.detect(img, conf_obj=0.15, conf_weapon=0.55)
        s = detector.summary(detections)

        print(f"  Persons / skeletons: {s['persons']} / {s['skeletons']}")
        print(f"  HIGH threat: {s['high'] if s['high'] else '(žádná)'}")
        print(f"  MEDIUM:      {s['medium'] if s['medium'] else '(žádné)'}")
        print(f"  CARRY:       {s['carry'] if s['carry'] else '(žádné)'}")

        out_name = os.path.basename(frame_path).replace('.jpg', '_suspicious.jpg')
        out_path = os.path.join(out_dir, out_name)
        detector.visualize(img, detections, save_path=out_path,
                           title=f"{os.path.basename(frame_path)}")

    print(f"\n{'=' * 70}")
    print(f"✓ Done. Visualizations: {out_dir}/")
    print('=' * 70)
