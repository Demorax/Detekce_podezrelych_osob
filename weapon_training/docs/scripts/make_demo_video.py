#!/usr/bin/env python3
"""Ukázkové video a snímky pro obhajobu: větev ARMED na bitce 005 a na rvačce bez zbraní.

Vykresluje z mezipaměti `eval_armed_pipeline.py` (armed_eval/video_*.json) — žádná
nová inference. Stav osob počítá skutečný kód pipeline (trackery, update_armed_status)
s nasazeným nastavením: melee ≥ 0,20, COCO pálka ≥ 0,50, 2 snímky po sobě.
Osoby a zápěstí jsou z YOLO11x-pose (náhrada RT-DETR + ViTPose, viz §6g).

Výstup:
    docs/demo/armed_005.mp4, armed_019.mp4         5 FPS, nativní rozlišení (mimo git)
    docs/figures/fig8_armed_ukazka.png/.pdf        tři snímky vedle sebe pro text práce

Použití:
    python make_demo_video.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "docs/demo"
MELEE_THR, BAT_THR, MIN_FRAMES = 0.20, 0.50, 2
CLIPS = {"005": (720, 1100), "019": (None, None)}   # zdrojové framy (None = celé video)

RED, YELLOW, ORANGE, GRAY, WHITE = (40, 40, 230), (0, 200, 240), (0, 130, 255), (170, 170, 170), (255, 255, 255)


def run(vid: str, aod) -> list[dict]:
    """Projde video stejně jako pipeline a pro každý frame vrátí, co vykreslit."""
    d = json.loads((HERE / "armed_eval" / f"video_{vid}.json").read_text())
    pt = aod.PersonTracker(max_dist=aod.DEFAULT_MAX_PERSON_TRACK_DIST)
    ot = aod.ObjectTracker(iou_threshold=aod.DEFAULT_OBJ_IOU_THRESHOLD, max_missed=aod.DEFAULT_MAX_OBJ_MISSED)
    states = []
    for i, f in enumerate(d["frames"]):
        objs = []
        for box, s, lab in f["objects"]:
            if lab == "melee" and s < MELEE_THR:
                continue
            if lab == "baseball bat" and s < BAT_THR:
                continue
            objs.append((np.array(box), s, {"gun_v2": "gun", "knife_v2": "knife"}.get(lab, lab)))
        persons = np.array(f["persons"], dtype=np.float32).reshape(-1, 5)
        skels = np.array(f["skeletons"], dtype=np.float32) if f["skeletons"] else None
        tracks = pt.update(i, persons, skels)
        otracks = ot.update(i, objs)
        aod.update_associations(i, tracks, otracks)
        aod.update_armed_status(tracks, otracks, min_frames=MIN_FRAMES)
        states.append({
            "src": f["src"],
            "persons": [(p["box"].tolist(), bool(p.get("armed")), p.get("weapon_label"), pid,
                         aod._wrist_points(p.get("skeleton"))) for pid, p in tracks.items()],
            "weapons": [(o[0].tolist(), o[1], o[2]) for o in objs if o[2] in aod.HIGH_THREAT_LABELS],
        })
    return states


def draw(img: np.ndarray, st: dict, fps: float) -> np.ndarray:
    out = img.copy()
    for box, armed, wl, pid, wrists in st["persons"]:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), RED if armed else GRAY, 2 if armed else 1)
        if armed:
            cv2.putText(out, f"ARMED: {wl}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, RED, 2)
            for w in wrists:
                cv2.circle(out, (int(w[0]), int(w[1])), 3, RED, -1)
    for box, s, lab in st["weapons"]:
        x1, y1, x2, y2 = map(int, box)
        col = YELLOW if lab == "melee" else ORANGE
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{lab} {s:.2f}", (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
    n_armed = sum(p[1] for p in st["persons"])
    bar = (0, 0, 150) if n_armed else (40, 40, 40)
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), bar, -1)
    # OpenCV neumí diakritiku, text je proto bez háčků a čárek
    txt = f"t = {st['src'] / fps:5.1f} s   " + (f"POPLACH: ozbrojene osoby {n_armed}" if n_armed else "bez poplachu")
    cv2.putText(out, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    return out


def main() -> int:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
    from eval_armed_pipeline import stub_heavy_modules
    stub_heavy_modules()
    import abandoned_object_detector as aod

    OUT.mkdir(parents=True, exist_ok=True)
    stills = {}
    for vid, (lo, hi) in CLIPS.items():
        states = run(vid, aod)
        cap = cv2.VideoCapture(str(ROOT / "data/Motion_Emotion" / f"{vid}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(str(OUT / f"armed_{vid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
        by_src = {s["src"]: s for s in states}
        idx, n = 0, 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if idx in by_src and (lo is None or lo <= idx <= hi):
                frame = draw(img, by_src[idx], fps)
                vw.write(frame); n += 1
                stills[(vid, idx)] = (frame, by_src[idx])
            idx += 1
        vw.release()
        print(f"  armed_{vid}.mp4: {n} snímků")

    # snímky do práce: vrchol bitky s nejvíc ozbrojenými, začátek poplachu, rvačka bez zbraní
    s005 = [(k, v) for k, v in stills.items() if k[0] == "005"]
    peak = max(s005, key=lambda kv: (sum(p[1] for p in kv[1][1]["persons"]), -kv[0][1]))
    first = min((kv for kv in s005 if any(p[1] for p in kv[1][1]["persons"])), key=lambda kv: kv[0][1])
    s019 = [(k, v) for k, v in stills.items() if k[0] == "019"]
    calm = next(kv for kv in s019 if kv[0][1] >= 600 and not any(p[1] for p in kv[1][1]["persons"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def crop(frame, st, zoom):
        rows = np.where(frame[30:].mean(axis=(1, 2)) > 8)[0] + 30   # bez černých pruhů (stavový řádek nechat)
        y_lo, y_hi = rows.min(), rows.max()
        if not zoom:
            return np.vstack([frame[:22], frame[y_lo:y_hi]])
        boxes = [p[0] for p in st["persons"] if p[1]] + [w[0] for w in st["weapons"]]
        x_lo = min(b[0] for b in boxes); x_hi = max(b[2] for b in boxes)
        cx = (x_lo + x_hi) / 2; half = max(170, (x_hi - x_lo) / 2 + 30)
        x0 = int(max(0, min(cx - half, frame.shape[1] - 2 * half)))
        return frame[y_lo:y_hi, x0:int(x0 + 2 * half)]   # stavový řádek by byl uříznutý, poplach říká nadpis

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, (k, (frame, st)), title, zoom in zip(axes, (first, peak, calm),
                                                 ("a) první poplach v bitce 005", "b) vrchol bitky 005",
                                                  "c) rvačka bez zbraní (019)"), (True, True, False)):
        frame = crop(frame, st, zoom)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); ax.axis("off")
        ax.set_title(title, loc="left", fontsize=8)
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / "docs/figures" / f"fig8_armed_ukazka.{ext}", dpi=220, bbox_inches="tight")
    print(f"  snímky: 005 f{first[0][1]}, 005 f{peak[0][1]}, 019 f{calm[0][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
