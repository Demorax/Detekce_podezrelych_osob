#!/usr/bin/env python3
"""Kalibrace prahu `melee` na časové polovině testu z videa 005 (TRAINING_REPORT §6e).

Proč časová, ne náhodná polovina: hustě vzorkované framy jsou od sebe 0,17 s,
takže sousední framy jsou skoro stejné. Náhodné rozdělení by do obou polovin
dalo téměř identické obrázky a ověření by nic neměřilo.

  kalibrační  : zdrojový frame <= 902
  mezera      : 907 (vynechán)
  ověřovací   : zdrojový frame >= 912

Pro každý kandidátní práh vypíše P/R/F1 na obou polovinách a počet falešných
poplachů na framech bez zbraní (testovací negativa z §6d + klidná scéna 001/002).

Použití:
    python calibrate_melee_threshold.py --device mps
"""
from __future__ import annotations

import argparse
import glob
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MELEE = 2
THRESHOLDS = [0.027, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def frame_idx(p: Path) -> int:
    return int(p.stem.split("_")[-1])          # 005d_00912 -> 912


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE.parent / "models/weapon_detection_yolo11l_med_v2/best.pt"))
    ap.add_argument("--test-dir", default=str(HERE / "annotate_005"))
    ap.add_argument("--neg-dir", default=str(HERE / "annotate_neg_test/images"))
    ap.add_argument("--calm-glob", default=str(HERE / "datasets/med_finetune_v2/val/images") + "/med_00[12]*.jpg")
    ap.add_argument("--cal-max", type=int, default=902)
    ap.add_argument("--hold-min", type=int, default=912)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    imgs = sorted((Path(args.test_dir) / "images").glob("*.jpg"))
    halves = {
        "kalibrační": [p for p in imgs if frame_idx(p) <= args.cal_max],
        "ověřovací": [p for p in imgs if frame_idx(p) >= args.hold_min],
    }

    with tempfile.TemporaryDirectory() as td:
        for name, files in halves.items():
            lst = Path(td) / f"{name}.txt"
            lst.write_text("\n".join(str(p) for p in files) + "\n")
            yml = Path(td) / f"{name}.yaml"
            yml.write_text(f"train: {lst}\nval: {lst}\nnc: 3\nnames: [gun, knife, melee]\n")
            r = model.val(data=str(yml), split="val", imgsz=args.imgsz, batch=4, conf=0.001,
                          device=args.device, plots=False, verbose=False)
            i = list(r.box.ap_class_index).index(MELEE)
            px = r.box.px
            print(f"\n{name}: {len(files)} framů, AP50 melee {r.box.ap50[i]:.3f}")
            if name == "kalibrační":
                k = int(np.argmax(r.box.f1_curve[i]))
                print(f"  argmax F1 = {r.box.f1_curve[i][k]:.3f} při conf {px[k]:.3f}")
            for t in THRESHOLDS:
                k = int(np.searchsorted(px, t))
                print(f"  conf {t:5.3f}: P {r.box.p_curve[i][k]:.2f}  R {r.box.r_curve[i][k]:.2f}"
                      f"  F1 {r.box.f1_curve[i][k]:.2f}")

    neg_sets = {
        "rvačky bez zbraní": sorted(Path(args.neg_dir).glob("*.jpg")),
        "klidná scéna": sorted(Path(p) for p in glob.glob(args.calm_glob))  # Path("/").glob na Windows neumí absolutní vzor,
    }
    print("\nFalešné poplachy melee (boxy / framy s poplachem):")
    for name, files in neg_sets.items():
        confs = []                               # max. conf melee boxů pro každý frame
        for p in files:
            b = model.predict(str(p), conf=min(THRESHOLDS), imgsz=args.imgsz,
                              device=args.device, verbose=False)[0].boxes
            confs.append([float(c) for c, k in zip(b.conf, b.cls) if int(k) == MELEE])
        row = []
        for t in THRESHOLDS:
            n_box = sum(sum(c >= t for c in fc) for fc in confs)
            n_fr = sum(any(c >= t for c in fc) for fc in confs)
            row.append(f"{t}: {n_box}/{n_fr}")
        print(f"  {name} ({len(files)} framů): " + "  ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
