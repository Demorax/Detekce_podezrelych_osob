#!/usr/bin/env python3
"""Časové potvrzení poplachů: tyč musí být vidět v k po sobě jdoucích framech.

Otázka: kolik falešných poplachů odstraní požadavek, aby detekce `melee`
přetrvala k vzorkovaných framů, a kolik to stojí na skutečných tyčích?

Simuluje logiku ARMED z `abandoned_object_detector.py`:
  * vzorkování 5 FPS na nativních framech 854×480 (výchozí nastavení pipeline),
  * detekce v framu t je potvrzená, když tvoří řetěz k detekcí v po sobě
    jdoucích framech a každé dvě sousední jsou blízko sebe (stejné místo/osoba),
  * po potvrzení drží poplach ARMED_HOLD_FRAMES (8) framů — hystereze jako v pipeline.

Zjednodušení: pipeline navíc vyžaduje zápěstí u boxu zbraně (ViTPose, na Macu
nejde spustit). Simulace tento filtr vynechává, takže počty falešných poplachů
jsou HORNÍ odhad; poměr mezi variantami k je ale srovnatelný.

Data:
  * pozitiva: 48 anotovaných framů videa 005 (krok 5 zdrojových framů ≈ 6 FPS,
    blízko 5 FPS pipeline) → recall boxů; celé video 005 při 5 FPS → zpoždění poplachu,
  * rvačky bez zbraní: 017, 026 (testovací negativa §6d), 018, 019, 020, 007, 009
    (nikdy nepoužité v tréninku ani při ladění),
  * klidné scény: 021, 027, 030.

Detekce se ukládají do mezipaměti (`temporal_eval/`), simulace je pak okamžitá.

Použití:
    python eval_temporal_filter.py --device mps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent / "data/Motion_Emotion"
CACHE = HERE / "temporal_eval"
WEIGHTS = HERE.parent / "models/weapon_detection_yolo11l_med_v2/best.pt"
MELEE = 2
SAMPLE_STEP = 6            # 29,97 fps / 5 FPS
HOLD = 8                   # ARMED_HOLD_FRAMES
FIGHTS = ["017", "026", "018", "019", "020", "007", "009"]
CALM = ["021", "027", "030"]
FIGHT_005 = (840, 1008)    # zdrojové framy s tyčemi (2 FPS framy 60–72)
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
KS = [1, 2, 3]


def detect_video(model, vid: str, device: str) -> dict:
    out = CACHE / f"video_{vid}.json"
    if out.exists():
        return json.loads(out.read_text())
    cap = cv2.VideoCapture(str(VIDEOS / f"{vid}.mp4"))
    frames, idx = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if idx % SAMPLE_STEP == 0:
            r = model.predict(img, conf=0.05, imgsz=960, device=device, verbose=False)[0]
            frames.append({"src": idx, "dets": [[*b.xyxy[0].tolist(), float(b.conf[0])]
                                                for b in r.boxes if int(b.cls[0]) == MELEE]})
        idx += 1
    res = {"video": vid, "fps": cap.get(cv2.CAP_PROP_FPS), "frames": frames}
    out.write_text(json.dumps(res))
    print(f"  {vid}: {len(frames)} framů")
    return res


def detect_annotated(model, device: str) -> list[dict]:
    out = CACHE / "annotated_005.json"
    if out.exists():
        return json.loads(out.read_text())
    res = []
    for p in sorted((HERE / "annotate_005/images").glob("*.jpg")):
        r = model.predict(str(p), conf=0.05, imgsz=960, device=device, verbose=False)[0]
        h, w = r.orig_shape
        gt = []
        for line in (HERE / "annotate_005/labels" / f"{p.stem}.txt").read_text().split("\n"):
            if line.strip() and int(line.split()[0]) == MELEE:
                _, cx, cy, bw, bh = map(float, line.split())
                gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        res.append({"src": int(p.stem.split("_")[-1]), "gt": gt,
                    "dets": [[*b.xyxy[0].tolist(), float(b.conf[0])]
                             for b in r.boxes if int(b.cls[0]) == MELEE]})
    out.write_text(json.dumps(res))
    return res


def near(a, b) -> bool:
    """Dvě detekce ve vedlejších framech patří k sobě (tyč se mezi framy hýbe)."""
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    scale = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    return np.hypot(ca[0] - cb[0], ca[1] - cb[1]) < 1.5 * scale


def confirm(frames: list[dict], thr: float, k: int) -> list[list[list[float]]]:
    """Pro každý frame vrátí detekce potvrzené řetězem délky >= k."""
    prev, out = [], []
    for f in frames:
        cur = []
        for d in f["dets"]:
            if d[4] < thr:
                continue
            chain = 1 + max((c for p, c in prev if near(d, p)), default=0)
            cur.append((d, chain))
        out.append([d for d, c in cur if c >= k])
        prev = cur
    return out


def alarm_track(confirmed: list) -> list[bool]:
    armed, hold = [], 0
    for c in confirmed:
        hold = HOLD if c else max(0, hold - 1)
        armed.append(bool(c) or hold > 0)
    return armed


def events(armed: list[bool]) -> int:
    return sum(1 for i, a in enumerate(armed) if a and (i == 0 or not armed[i - 1]))


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    vids = {v: detect_video(model, v, args.device) for v in ["005", *FIGHTS, *CALM]}
    ann = detect_annotated(model, args.device)

    n_gt = sum(len(f["gt"]) for f in ann)
    fight_min = sum(len(vids[v]["frames"]) for v in FIGHTS) / 5 / 60
    calm_min = sum(len(vids[v]["frames"]) for v in CALM) / 5 / 60
    print(f"\nrvačky bez zbraní: {fight_min:.1f} min, klidné scény: {calm_min:.1f} min, "
          f"anotované tyče 005: {n_gt}")

    rows = []
    for thr in THRESHOLDS:
        for k in KS:
            conf_ann = confirm(ann, thr, k)
            tp = sum(any(iou(g, d) >= 0.5 for d in c) for f, c in zip(ann, conf_ann) for g in f["gt"])
            fp_ann = sum(not any(iou(g, d) >= 0.5 for g in f["gt"]) for f, c in zip(ann, conf_ann) for d in c)

            fr005 = vids["005"]["frames"]
            arm005 = alarm_track(confirm(fr005, thr, k))
            in_fight = [a for f, a in zip(fr005, arm005) if FIGHT_005[0] <= f["src"] <= FIGHT_005[1]]
            first = next((f["src"] for f, a in zip(fr005, arm005) if a and f["src"] >= FIGHT_005[0]), None)
            delay = None if first is None else (first - FIGHT_005[0]) / vids["005"]["fps"]

            per = {}
            for v in FIGHTS + CALM:
                arm = alarm_track(confirm(vids[v]["frames"], thr, k))
                per[v] = {"events": events(arm), "armed_s": sum(arm) / 5}
            rows.append({
                "thr": thr, "k": k, "recall_005": tp / n_gt, "fp_005": fp_ann,
                "fight005_armed_frac": float(np.mean(in_fight)), "delay_005_s": delay,
                "fight_events": sum(per[v]["events"] for v in FIGHTS),
                "fight_armed_s": sum(per[v]["armed_s"] for v in FIGHTS),
                "calm_events": sum(per[v]["events"] for v in CALM),
                "per_video": per,
            })

    print(f"\n{'práh':>5} {'k':>2} | {'recall 005':>10} {'FP 005':>6} {'005 v poplachu':>14} "
          f"{'zpoždění':>8} | {'poplachy rvačky':>15} {'s v poplachu':>12} | {'klid':>4}")
    for r in rows:
        d = "—" if r["delay_005_s"] is None else f"{r['delay_005_s']:.1f} s"
        print(f"{r['thr']:5.2f} {r['k']:2d} | {r['recall_005']:10.2f} {r['fp_005']:6d} "
              f"{r['fight005_armed_frac']:14.0%} {d:>8} | {r['fight_events']:15d} "
              f"{r['fight_armed_s']:12.1f} | {r['calm_events']:4d}")
    (CACHE / "results.json").write_text(json.dumps(
        {"fight_minutes": fight_min, "calm_minutes": calm_min, "n_gt_005": n_gt, "rows": rows},
        indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
