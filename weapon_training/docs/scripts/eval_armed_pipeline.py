#!/usr/bin/env python3
"""Falešné poplachy ARMED s kontrolou zápěstí: simulace logiky pipeline na celých videích.

`eval_temporal_filter.py` (§6f) počítal poplachy jen z detekcí zbraní, bez kontroly,
že zbraň drží osoba. Tady se spouští SKUTEČNÝ kód pipeline:
  * `SuspiciousDetector.detect()` — COCO YOLO11x (baseball bat / knife, conf 0,15,
    imgsz 1280) i model v2 s prahy podle tříd,
  * `PersonTracker`, `ObjectTracker`, `update_associations`, `update_armed_status`
    z `abandoned_object_detector.py` s výchozími parametry, 5 FPS, nativní rozlišení.

Jediná náhrada: RT-DETR (osoby) a ViTPose (skelety) na Macu nejdou spustit, proto
osoby i zápěstí dodává YOLO11x-pose. Jeho 17 bodů COCO se převede do 14bodového
formátu CrowdPose, který pipeline čeká (zápěstí na indexech 4 a 5).

Detekce se ukládají do `armed_eval/` (melee od conf 0,10, aby šly porovnat prahy),
varianty se pak simulují okamžitě.

Použití:
    python eval_armed_pipeline.py --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "armed_eval"
FIGHTS = ["017", "026", "018", "019", "020", "007", "009"]
NEVER_USED = ["018", "019", "020", "007", "009"]
CALM = ["021", "027", "030"]
FIGHT_005 = (840, 1008)
SAMPLE_STEP = 6
CACHE_MELEE_CONF = 0.10
# COCO i model v2 mají třídu 'knife'; v cache se třídy v2 označí příponou, aby šly odlišit
V2_LABELS = {0: "gun_v2", 1: "knife_v2", 2: "melee"}
V2_LABEL_BACK = {"gun_v2": "gun", "knife_v2": "knife"}
COCO_THREAT = {"baseball bat", "knife"}


def stub_heavy_modules() -> None:
    """RT-DETR a MMPose nejsou na Macu — zástupné moduly, jako v test_pipeline_weapon_branch.py."""
    for name in ["src", "src.core", "mmpose", "mmpose.apis", "mmpose.registry",
                 "models.vitpose", "models.vitpose.models", "models.vitpose.models.backbone",
                 "models.vitpose.models.backbone.vit", "models.vitpose.models.detectors",
                 "models.vitpose.models.detectors.top_down", "models.vitpose.models.head",
                 "models.vitpose.models.head.topdown_heatmap_simple_head", "visualization_helper"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["src.core"].YAMLConfig = object
    sys.modules["mmpose.apis"].init_model = sys.modules["mmpose.apis"].inference_topdown = None
    sys.modules["mmpose.registry"].MODELS = types.SimpleNamespace(register_module=lambda *a, **k: (lambda c: c))
    sys.modules["models.vitpose.models.backbone.vit"].ViT = object
    sys.modules["models.vitpose.models.detectors.top_down"].TopDown = object
    sys.modules["models.vitpose.models.head.topdown_heatmap_simple_head"].TopdownHeatmapSimpleHead = object
    sys.modules["visualization_helper"].visualize_detections = None


def coco17_to_crowdpose14(kp: np.ndarray) -> np.ndarray:
    """(17,3) COCO -> (14,3) CrowdPose: ramena..kotníky 1:1, hlava = nos, krk = střed ramen."""
    out = np.zeros((14, 3), dtype=np.float32)
    out[0:12] = kp[5:17]
    out[12] = kp[0]
    out[13, :2] = (kp[5, :2] + kp[6, :2]) / 2
    out[13, 2] = min(kp[5, 2], kp[6, 2])
    return out


def cache_video(vid: str, det, pose, device: str) -> dict:
    import cv2
    out = CACHE / f"video_{vid}.json"
    if out.exists():
        return json.loads(out.read_text())
    cap = cv2.VideoCapture(str(ROOT / "data/Motion_Emotion" / f"{vid}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, idx = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if idx % SAMPLE_STEP == 0:
            o = det.detect(img)                                    # skutečná detect() pipeline
            pr = pose.predict(img, conf=0.25, imgsz=1280, device=device, verbose=False)[0]
            persons, skels = [], []
            if pr.boxes is not None and len(pr.boxes):
                kps = pr.keypoints.data.cpu().numpy()
                for b, kp in zip(pr.boxes, kps):
                    persons.append([*b.xyxy[0].tolist(), float(b.conf[0])])
                    skels.append(coco17_to_crowdpose14(kp).round(2).tolist())
            frames.append({
                "src": idx,
                "objects": [[[round(float(v), 1) for v in box], round(float(s), 4), lab]
                            for tier in ("high", "medium", "carry") for box, s, lab in o[tier]],
                "persons": [[round(v, 1) for v in p[:4]] + [round(p[4], 3)] for p in persons],
                "skeletons": skels,
            })
        idx += 1
    res = {"video": vid, "fps": fps, "frames": frames}
    out.write_text(json.dumps(res))
    print(f"  {vid}: {len(frames)} framů", flush=True)
    return res


def simulate(video: dict, aod, melee_thr: float, min_frames: int, use_coco_threat: bool) -> list[set]:
    """Vrátí pro každý frame množinu ID osob s ARMED."""
    pt = aod.PersonTracker(max_dist=aod.DEFAULT_MAX_PERSON_TRACK_DIST)
    ot = aod.ObjectTracker(iou_threshold=aod.DEFAULT_OBJ_IOU_THRESHOLD, max_missed=aod.DEFAULT_MAX_OBJ_MISSED)
    armed_per_frame = []
    for i, f in enumerate(video["frames"]):
        objs = []
        for box, s, lab in f["objects"]:
            if lab == "melee" and s < melee_thr:
                continue
            if lab in COCO_THREAT and not use_coco_threat:
                continue
            objs.append((np.array(box), s, V2_LABEL_BACK.get(lab, lab)))
        persons = np.array(f["persons"], dtype=np.float32).reshape(-1, 5)
        skels = np.array(f["skeletons"], dtype=np.float32) if f["skeletons"] else None
        tracks = pt.update(i, persons, skels)
        otracks = ot.update(i, objs)
        aod.update_associations(i, tracks, otracks)
        aod.update_armed_status(tracks, otracks, min_frames=min_frames)
        armed_per_frame.append({pid for pid, p in tracks.items() if p.get("armed")})
    return armed_per_frame


def summarize(video: dict, armed: list[set]) -> dict:
    any_armed = [bool(a) for a in armed]
    events = sum(1 for i, a in enumerate(any_armed) if a and (i == 0 or not any_armed[i - 1]))
    person_events = 0
    prev: set = set()
    for a in armed:
        person_events += len(a - prev)
        prev = a
    return {"events": events, "person_events": person_events, "armed_s": sum(any_armed) / 5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    stub_heavy_modules()
    import suspicious_detector as sd
    import abandoned_object_detector as aod
    from ultralytics import YOLO

    videos = ["005", *FIGHTS, *CALM]
    if not all((CACHE / f"video_{v}.json").exists() for v in videos):
        c = sd.WEAPON_CONF_PER_CLASS
        sd.WEAPON_CONF_PER_CLASS = {"gun_v2": c["gun"], "knife_v2": c["knife"], "melee": CACHE_MELEE_CONF}
        sd.WEAPON_HIGH_THREAT = V2_LABELS
        det = sd.SuspiciousDetector.__new__(sd.SuspiciousDetector)
        det.rtdetr = types.SimpleNamespace(better_detection=lambda img: np.zeros((0, 5)))
        det.vitpose = types.SimpleNamespace(extract_keypoints=lambda img, b: np.array([]))
        det.coco = YOLO(sd._YOLO_MODEL)
        det.weapon = YOLO(sd._WEAPON_MODEL)
        det.device = args.device
        det.coco_filter = list(sd.COCO_HIGH_THREAT) + list(sd.COCO_MEDIUM_THREAT) + list(sd.COCO_CARRY)
        pose = YOLO(str(ROOT / "yolo11x-pose.pt"))
    data = {}
    for v in videos:
        if (CACHE / f"video_{v}.json").exists():
            data[v] = json.loads((CACHE / f"video_{v}.json").read_text())
        else:
            data[v] = cache_video(v, det, pose, args.device)

    variants = [
        ("dříve: melee 0,40, 1 frame", 0.40, 1, True),
        ("melee 0,20, 1 frame", 0.20, 1, True),
        ("NASAZENO: melee 0,20, 2 framy", 0.20, 2, True),
        ("melee 0,15, 2 framy", 0.15, 2, True),
        ("melee 0,10, 2 framy", 0.10, 2, True),
        ("melee 0,20, 3 framy", 0.20, 3, True),
        ("melee 0,20, 2 framy, bez COCO pálky/nože", 0.20, 2, False),
    ]
    rows = []
    for name, thr, k, coco in variants:
        r = {"varianta": name, "thr": thr, "k": k, "coco_bat": coco, "per_video": {}}
        for v in videos:
            arm = simulate(data[v], aod, thr, k, coco)
            r["per_video"][v] = summarize(data[v], arm)
            if v == "005":
                fr = data[v]["frames"]
                inside = [bool(a) for f, a in zip(fr, arm) if FIGHT_005[0] <= f["src"] <= FIGHT_005[1]]
                first = next((f["src"] for f, a in zip(fr, arm) if a and f["src"] >= FIGHT_005[0]), None)
                r["fight005_armed_frac"] = float(np.mean(inside))
                r["delay_005_s"] = None if first is None else (first - FIGHT_005[0]) / data[v]["fps"]
        for grp, vs in (("rvacky", FIGHTS), ("nepouzita", NEVER_USED), ("klid", CALM)):
            r[f"{grp}_events"] = sum(r["per_video"][v]["events"] for v in vs)
            r[f"{grp}_armed_s"] = sum(r["per_video"][v]["armed_s"] for v in vs)
        rows.append(r)

    mins = {g: sum(len(data[v]["frames"]) for v in vs) / 300 for g, vs in
            (("rvacky", FIGHTS), ("nepouzita", NEVER_USED), ("klid", CALM))}
    print(f"\nrvačky {mins['rvacky']:.1f} min (z toho nepoužité {mins['nepouzita']:.1f}), klid {mins['klid']:.1f} min")
    print(f"{'varianta':<38} | {'bitka 005':>9} {'zpoždění':>8} | {'rvačky':>6} {'nepoužité':>9} {'s':>6} | {'klid':>4}")
    for r in rows:
        d = "—" if r["delay_005_s"] is None else f"{r['delay_005_s']:.1f} s"
        print(f"{r['varianta']:<38} | {r['fight005_armed_frac']:9.0%} {d:>8} | {r['rvacky_events']:6d} "
              f"{r['nepouzita_events']:9d} {r['rvacky_armed_s']:6.1f} | {r['klid_events']:4d}")
    (CACHE / "results.json").write_text(json.dumps({"minutes": mins, "rows": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
