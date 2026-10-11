"""Izolovaný test (2026-10-11, viz docs/TRAINING_REPORT.md §6d) větve se zbraněmi v suspicious_detector.detect().
RT-DETR a MMPose nejsou na Macu -> zástupné moduly; zbytek (YOLO11x, model v2,
kód detect()) je skutečný a nezměněný."""
import sys, types, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.getcwd())
for name in ["src", "src.core", "mmpose", "mmpose.apis", "mmpose.registry",
             "models.vitpose", "models.vitpose.models", "models.vitpose.models.backbone",
             "models.vitpose.models.backbone.vit", "models.vitpose.models.detectors",
             "models.vitpose.models.detectors.top_down", "models.vitpose.models.head",
             "models.vitpose.models.head.topdown_heatmap_simple_head", "visualization_helper"]:
    m = types.ModuleType(name); sys.modules[name] = m
sys.modules["src.core"].YAMLConfig = object
sys.modules["mmpose.apis"].init_model = sys.modules["mmpose.apis"].inference_topdown = None
reg = types.SimpleNamespace(register_module=lambda *a, **k: (lambda c: c))
sys.modules["mmpose.registry"].MODELS = reg
sys.modules["models.vitpose.models.backbone.vit"].ViT = object
sys.modules["models.vitpose.models.detectors.top_down"].TopDown = object
sys.modules["models.vitpose.models.head.topdown_heatmap_simple_head"].TopdownHeatmapSimpleHead = object
sys.modules["visualization_helper"].visualize_detections = None

import numpy as np, cv2
import suspicious_detector as sd
from ultralytics import YOLO

print("model v pipeline:", sd._WEAPON_MODEL)
w = YOLO(sd._WEAPON_MODEL)
print("tridy modelu:      ", w.names)
print("WEAPON_HIGH_THREAT:", sd.WEAPON_HIGH_THREAT)
assert {k: v for k, v in w.names.items()} == sd.WEAPON_HIGH_THREAT, "MAPOVANI TRID NESEDI"
print("prahy:", sd.WEAPON_CONF_PER_CLASS, "COCO:", sd.COCO_CONF_PER_CLASS)

det = sd.SuspiciousDetector.__new__(sd.SuspiciousDetector)
det.rtdetr = types.SimpleNamespace(better_detection=lambda img: np.zeros((0, 5)))
det.vitpose = types.SimpleNamespace(extract_keypoints=lambda img, b: np.array([]))
det.coco = YOLO(sd._YOLO_MODEL); det.weapon = w; det.device = "mps"
det.coco_filter = list(sd.COCO_HIGH_THREAT) + list(sd.COCO_MEDIUM_THREAT) + list(sd.COCO_CARRY)

import abandoned_object_detector as aod
for path, desc in [("weapon_training/annotate_005/images/005d_00932.jpg", "bitka 005 (tyce)"),
                   ("weapon_training/annotate_neg_test/images/017_017_0081.jpg", "rvacka bez zbrani (017)"),
                   ("frames_0.5/005/005_0064.jpg", "bitka 005, NATIVNI 854x480 bez upscale")]:
    img = cv2.imread(path)
    out = det.detect(img)
    hi = out["high"]
    print(f"\n[{desc}] {img.shape[1]}x{img.shape[0]}  high={len(hi)} medium={len(out['medium'])} carry={len(out['carry'])}")
    for box, score, label in sorted(hi, key=lambda x: -x[1]):
        thr = sd.WEAPON_CONF_PER_CLASS.get(label, sd.COCO_CONF_PER_CLASS.get(label))
        ok = "ARMED-label" if label in aod.HIGH_THREAT_LABELS else "NENI v HIGH_THREAT_LABELS!"
        print(f"   {label:<13} {score:.3f}  prah={thr}  {ok}")
        if label in sd.WEAPON_CONF_PER_CLASS:
            assert score >= thr, f"detekce {label} {score} pod prahem {thr}"
        if label in sd.COCO_CONF_PER_CLASS:
            assert score >= sd.COCO_CONF_PER_CLASS[label], f"COCO {label} {score} pod prahem"
print("\nVSECHNY KONTROLY PROSLY")
