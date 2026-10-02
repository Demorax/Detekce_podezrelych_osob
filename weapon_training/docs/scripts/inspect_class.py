#!/usr/bin/env python3
"""Vyřízne vzorek bounding boxů zvolené třídy a složí je do jednoho náhledu.

Slouží k rozhodnutí, co je nejednoznačná třída (typicky 'weapon') skutečně za
objekt — jestli jsou to pistole, nože, nebo mix. Na tom závisí, jestli se dá
dataset použít, nebo se musí zahodit.

Použití:
    python inspect_class.py --ds datasets/raw/weapondetection-mqgm5__weapon-detection-cabwp \\
                            --cls weapon --n 48 --out weapon_sample.jpg
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

SPLITS = ("train", "valid", "val", "test")


def load_classes(ds: Path) -> list[str]:
    with (ds / "data.yaml").open() as f:
        d = yaml.safe_load(f)
    names = d.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names or [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, help="adresář datasetu (s data.yaml)")
    ap.add_argument("--cls", required=True, help="název třídy k prohlédnutí")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--tile", type=int, default=128, help="velikost dlaždice v px")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--pad", type=float, default=0.12, help="okraj okolo boxu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="class_sample.jpg")
    args = ap.parse_args()

    ds = Path(args.ds).expanduser().resolve()
    classes = load_classes(ds)
    target = args.cls.strip().lower()
    matches = [i for i, c in enumerate(classes) if str(c).strip().lower() == target]
    if not matches:
        raise SystemExit(f"Třída {args.cls!r} není v {classes}")
    cid = matches[0]
    print(f"třída {args.cls!r} = index {cid} z {classes}")

    # posbírat (obrázek, box) páry pro danou třídu
    found: list[tuple[Path, tuple[float, float, float, float]]] = []
    for split in SPLITS:
        lbl_d, img_d = ds / split / "labels", ds / split / "images"
        if not lbl_d.is_dir():
            continue
        for lbl in lbl_d.iterdir():
            if lbl.suffix != ".txt":
                continue
            try:
                lines = lbl.read_text().splitlines()
            except OSError:
                continue
            boxes = []
            for line in lines:
                p = line.split()
                if len(p) < 5:
                    continue
                try:
                    if int(float(p[0])) != cid:
                        continue
                    boxes.append(tuple(float(v) for v in p[1:5]))
                except ValueError:
                    continue
            if not boxes:
                continue
            img = next((img_d / (lbl.stem + e) for e in (".jpg", ".jpeg", ".png")
                        if (img_d / (lbl.stem + e)).is_file()), None)
            if img is None:
                continue
            for b in boxes:
                found.append((img, b))

    print(f"nalezeno {len(found)} boxů třídy {args.cls!r}")
    if not found:
        raise SystemExit("Žádné boxy — nic k prohlédnutí.")

    random.seed(args.seed)
    sample = random.sample(found, min(args.n, len(found)))

    t, cols = args.tile, args.cols
    rows = (len(sample) + cols - 1) // cols
    canvas = np.full((rows * t, cols * t, 3), 30, np.uint8)

    for i, (img_p, (cx, cy, bw, bh)) in enumerate(sample):
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        H, W = im.shape[:2]
        pw, ph = bw * args.pad, bh * args.pad
        x1 = int(max(0, (cx - bw / 2 - pw) * W))
        y1 = int(max(0, (cy - bh / 2 - ph) * H))
        x2 = int(min(W, (cx + bw / 2 + pw) * W))
        y2 = int(min(H, (cy + bh / 2 + ph) * H))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        crop = im[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        s = min(t / w, t / h)
        crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
        ch, cw = crop.shape[:2]
        r, c = divmod(i, cols)
        oy, ox = r * t + (t - ch) // 2, c * t + (t - cw) // 2
        canvas[oy:oy + ch, ox:ox + cw] = crop

    out = Path(args.out).expanduser().resolve()
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"náhled: {out}  ({rows}x{cols} dlaždic, {len(sample)} vzorků)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
