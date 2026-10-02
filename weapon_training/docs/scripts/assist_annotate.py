#!/usr/bin/env python3
"""Pomůcka pro anotaci s asistencí modelu jazyka (coarse-to-fine).

Proč to takhle: bez pomoci dokáže model jazyka odhadnout box s IoU ~0,70,
což je na trénink volné. S PŘIBLÍŽENÝM VÝŘEZEM A SOUŘADNICOVOU MŘÍŽKOU
lze souřadnice odečíst s přesností jednotek pixelů, protože se nehádá
poloha, ale čte se z mřížky.

Postup:
    1. `overview`  — celý frame s mřížkou → najít objekty, hrubé souřadnice
    2. `crop`      — přiblížený výřez s mřížkou → odečíst přesný box
    3. `write`     — zapsat boxy do YOLO labelu

Použití:
    python assist_annotate.py overview --dir annotate_med --stem 004_004_0065
    python assist_annotate.py crop --dir annotate_med --stem 004_004_0065 \\
                              --at 2246,662 --size 300
    python assist_annotate.py write --dir annotate_med --stem 004_004_0065 \\
                              --boxes "2:2226,628,2266,696" "0:862,1115,933,1171"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
GRID_COL = (0, 255, 255)
TXT_X = (255, 255, 0)
TXT_Y = (255, 0, 255)
CLASS_COL = [(255, 60, 60), (255, 200, 0), (60, 255, 60)]


def find_img(work: Path, stem: str) -> Path:
    for e in IMG_EXT:
        p = work / "images" / (stem + e)
        if p.is_file():
            return p
    raise SystemExit(f"Obrázek pro {stem!r} nenalezen")


def load_boxes(work: Path, stem: str, w: int, h: int):
    """-> [(cls, x1, y1, x2, y2)] v pixelech."""
    p = work / "labels" / (stem + ".txt")
    out = []
    if p.is_file():
        for line in p.read_text().splitlines():
            q = line.split()
            if len(q) != 5:
                continue
            c = int(float(q[0])); cx, cy, bw, bh = (float(v) for v in q[1:])
            out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def draw_grid(d: ImageDraw.ImageDraw, ox: int, oy: int, cw: int, ch: int,
              scale: float, step: int):
    """Mřížka s popisky v souřadnicích ORIGINÁLU."""
    x = (ox // step) * step
    while x <= ox + cw / scale:
        px = (x - ox) * scale
        if 0 <= px <= cw:
            d.line([(px, 0), (px, ch)], fill=GRID_COL, width=1)
            d.text((px + 3, 3), str(int(x)), fill=TXT_X)
        x += step
    y = (oy // step) * step
    while y <= oy + ch / scale:
        py = (y - oy) * scale
        if 0 <= py <= ch:
            d.line([(0, py), (cw, py)], fill=GRID_COL, width=1)
            d.text((3, py + 3), str(int(y)), fill=TXT_Y)
        y += step


def cmd_overview(work: Path, stem: str, args):
    img = find_img(work, stem)
    im = Image.open(img).convert("RGB")
    scale = min(args.width / im.width, 1.0)
    view = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(view)
    draw_grid(d, 0, 0, view.width, view.height, scale, args.step)
    if args.show_prelabels:
        for c, x1, y1, x2, y2 in load_boxes(work, stem, im.width, im.height):
            col = CLASS_COL[c % len(CLASS_COL)]
            d.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                        outline=col, width=2)
    out = Path(args.out)
    view.save(out)
    print(f"{out}  |  orig {im.width}x{im.height}, zobrazeno {view.width}x{view.height} "
          f"(scale {scale:.3f}), mřížka {args.step} px originálu")


def stretch_contrast(im: Image.Image, lo_pct: float = 2.0, hi_pct: float = 98.0):
    """Percentilové roztažení kontrastu.

    Na tmavém asfaltu a ve stínech je rozdíl mezi tyčí a předloktím v surovém
    výřezu často pod rozlišovací schopností oka. Roztažení histogramu podle
    percentilů ho zvýrazní, aniž by přibarvovalo — několikrát to rozhodlo
    případy, kde samotný zoom nestačil.
    """
    import numpy as np
    a = np.asarray(im, dtype=np.float32)
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        ch = a[..., c]
        lo, hi = np.percentile(ch, lo_pct), np.percentile(ch, hi_pct)
        if hi - lo < 1e-3:
            out[..., c] = ch
        else:
            out[..., c] = np.clip((ch - lo) * 255.0 / (hi - lo), 0, 255)
    return Image.fromarray(out.astype("uint8"))


def cmd_crop(work: Path, stem: str, args):
    img = find_img(work, stem)
    im = Image.open(img).convert("RGB")
    cx, cy = (int(v) for v in args.at.split(","))
    s = args.size
    x0, y0 = max(0, cx - s // 2), max(0, cy - s // 2)
    x1, y1 = min(im.width, x0 + s), min(im.height, y0 + s)
    crop = im.crop((x0, y0, x1, y1))
    if args.enhance:
        crop = stretch_contrast(crop)
    zoom = args.zoom
    view = crop.resize((int(crop.width * zoom), int(crop.height * zoom)),
                       Image.LANCZOS if zoom <= 4 else Image.NEAREST)
    d = ImageDraw.Draw(view)
    draw_grid(d, x0, y0, view.width, view.height, zoom, args.step)
    out = Path(args.out)
    view.save(out)
    print(f"{out}  |  výřez ({x0},{y0})-({x1},{y1}) zvětšeno {zoom}×, "
          f"mřížka {args.step} px originálu")


def cmd_write(work: Path, stem: str, args):
    img = find_img(work, stem)
    im = Image.open(img)
    W, H = im.width, im.height
    lines = []
    for spec in args.boxes:
        try:
            cls_s, rest = spec.split(":")
            x1, y1, x2, y2 = (float(v) for v in rest.split(","))
            c = int(cls_s)
        except ValueError:
            raise SystemExit(f"Špatný formát boxu: {spec!r} (čekám 'cls:x1,y1,x2,y2')")
        x1, x2 = sorted((max(0.0, x1), min(float(W), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(H), y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise SystemExit(f"Degenerovaný box: {spec!r}")
        lines.append(f"{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                     f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
    lp = work / "labels" / (stem + ".txt")
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(("\n".join(lines) + "\n") if lines else "")

    # označit frame jako zkontrolovaný — stejný formát jako annotate_boxes.py
    pp = work / "progress.json"
    prog = {"reviewed": [], "total": 0}
    if pp.is_file():
        try:
            prog = json.loads(pp.read_text())
        except ValueError:
            pass
    rev = set(prog.get("reviewed", []))
    rev.add(stem)
    n_img = len([p for p in (work / "images").iterdir() if p.suffix.lower() in IMG_EXT])
    pp.write_text(json.dumps({"reviewed": sorted(rev), "total": n_img}, indent=1))
    print(f"{stem}: zapsáno {len(lines)} boxů, zkontrolováno {len(rev)}/{n_img}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["overview", "crop", "write"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", default="/tmp/annot_view.png")
    ap.add_argument("--step", type=int, default=0, help="krok mřížky v px originálu")
    ap.add_argument("--width", type=int, default=1600, help="overview: šířka náhledu")
    ap.add_argument("--show-prelabels", action="store_true")
    ap.add_argument("--at", help="crop: střed 'x,y' v souřadnicích originálu")
    ap.add_argument("--size", type=int, default=300, help="crop: velikost výřezu v px")
    ap.add_argument("--zoom", type=float, default=3.0)
    ap.add_argument("--enhance", action="store_true",
                    help="crop: percentilové roztažení kontrastu — pomáhá odlišit "
                         "tyč od předloktí na tmavém asfaltu a ve stínech")
    ap.add_argument("--boxes", nargs="*", default=[],
                    help="write: 'cls:x1,y1,x2,y2' ... (prázdné = frame bez objektů)")
    args = ap.parse_args()

    work = Path(args.dir).expanduser().resolve()
    if args.mode == "overview":
        args.step = args.step or 200
        cmd_overview(work, args.stem, args)
    elif args.mode == "crop":
        if not args.at:
            raise SystemExit("crop vyžaduje --at x,y")
        args.step = args.step or 50
        cmd_crop(work, args.stem, args)
    else:
        cmd_write(work, args.stem, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
