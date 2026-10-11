#!/usr/bin/env python3
"""Těžká negativa pro v3: framy z videí BEZ zbraní, kde model v2 hlásí tyč.

Náhodné negativní framy model většinou už zvládá; učí se hlavně z těch, na
kterých chybuje. Skript projde videa při 5 FPS (jako pipeline), najde framy
s falešnou detekcí `melee` nad prahem a uloží je jako trénovací negativa
(prázdné labels) ve stejném formátu jako ostatní anotace (4× bikubicky).

Videa jsou vybrána tak, aby se NEPŘEKRÝVALA s žádnou testovací sadou:
  * test negativ §6d: 017, 026; časové potvrzení §6f: 007, 009, 018–021, 027, 030
  * trénink v2: 012, 014, 016, 025; zbraně: 004, 005; MED validace: 001, 002
Že ve vybraných videích nejsou zbraně, ověřil průchod všemi videi (MED_VIDEO_SCAN.md).

Aby se do tréninku nedostalo mnoho skoro stejných framů, bere se z každého
videa nejvýš --per-video framů, od sebe aspoň --min-gap sekund.

Použití:
    python mine_hard_negatives.py --device mps --out annotate_neg_hard
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent / "data/Motion_Emotion"
WEIGHTS = HERE.parent / "models/weapon_detection_yolo11l_med_v2/best.pt"
SOURCE = ["003", "006", "008", "010", "011", "013", "015", "022", "023", "024", "028", "029", "031"]
MELEE = 2
SAMPLE_STEP = 6
UPSCALE = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="annotate_neg_hard")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--per-video", type=int, default=8)
    ap.add_argument("--min-gap", type=float, default=2.0)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    out = HERE / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    manifest = []
    for vid in SOURCE:
        cap = cv2.VideoCapture(str(VIDEOS / f"{vid}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cands, idx = [], 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if idx % SAMPLE_STEP == 0:
                r = model.predict(img, conf=args.conf, imgsz=960, device=args.device, verbose=False)[0]
                c = [float(b.conf[0]) for b in r.boxes if int(b.cls[0]) == MELEE]
                if c:
                    cands.append((max(c), idx, img))
            idx += 1
        # nejjistější chyby první, s minimálním odstupem v čase
        picked = []
        for conf, i, img in sorted(cands, key=lambda x: -x[0]):
            if len(picked) >= args.per_video:
                break
            if all(abs(i - j) / fps >= args.min_gap for _, j, _ in picked):
                picked.append((conf, i, img))
        for conf, i, img in sorted(picked, key=lambda x: x[1]):
            stem = f"{vid}h_{i:05d}"
            big = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(out / "images" / f"{stem}.jpg"), big, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / f"{stem}.txt").write_text("")
            manifest.append({"stem": stem, "video": vid, "src_frame": i, "v2_max_conf": round(conf, 3)})
        print(f"  {vid}: {len(cands)} framů s falešnou detekcí, vybráno {len(picked)}")

    (out / "classes.txt").write_text("gun\nknife\nmelee\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"celkem {len(manifest)} těžkých negativ -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
