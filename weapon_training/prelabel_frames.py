#!/usr/bin/env python3
"""Předanotace framů natrénovaným modelem — vstup pro ruční korekci.

Smysl: anotovat 200-300 framů od nuly je den práce. Když model předkreslí
kandidátní boxy s NÍZKÝM confidence, zbývá je jen opravit — smazat špatné
a dokreslit ty, které model minul. To je řádově rychlejší.

Záměrně se pouští s conf ~0.10, tedy mnohem níž než v produkci. Cílem je
VYSOKÝ RECALL, ne přesnost: každý zachycený kandidát ušetří kreslení,
falešný se smaže jedním klávesovým stiskem.

Volba framů: `--stride` vybírá každý N-tý, protože sousední framy z 2 FPS
videa jsou si velmi podobné a anotovat je oba nemá hodnotu.

Výstup je připraven pro `annotate_boxes.py`:
    <out>/images/<video>_<frame>.jpg
    <out>/labels/<video>_<frame>.txt      (YOLO formát, prázdný = nic nenalezeno)
    <out>/prelabel_stats.json

Použití (na stroji s GPU a ultralytics):
    python prelabel_frames.py --weights best.pt --frames frames_0.5_upscaled \\
                              --out annotate_med --conf 0.10 --stride 1
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frames", required=True, help="adresář s framy (rekurzivně)")
    ap.add_argument("--out", default="annotate_med")
    ap.add_argument("--conf", type=float, default=0.10,
                    help="nízký práh = vysoký recall (default 0.10)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--stride", type=int, default=1,
                    help="brát každý N-tý frame (sousední framy jsou podobné)")
    ap.add_argument("--videos", default="", help="jen tato videa, oddělená čárkou (např. 004,001)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    root = Path(args.frames)
    frames = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    if args.videos:
        want = {v.strip() for v in args.videos.split(",") if v.strip()}
        frames = [f for f in frames if f.parent.name in want]
    frames = frames[:: max(1, args.stride)]
    if args.max_frames:
        frames = frames[: args.max_frames]
    if not frames:
        raise SystemExit(f"Žádné framy v {root} (videos={args.videos!r})")

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    names = model.names

    per_class: Counter[str] = Counter()
    with_det = 0
    manifest = []

    for i, fp in enumerate(frames):
        stem = f"{fp.parent.name}_{fp.stem}"
        dst_img = out / "images" / f"{stem}{fp.suffix.lower()}"
        if not dst_img.exists():
            shutil.copy2(fp, dst_img)

        r = model.predict(str(fp), conf=args.conf, imgsz=args.imgsz,
                          device=args.device, verbose=False)[0]

        lines = []
        if r.boxes is not None and len(r.boxes):
            with_det += 1
            for b in r.boxes:
                cid = int(b.cls[0])
                cx, cy, w, h = b.xywhn[0].tolist()
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                per_class[names[cid]] += 1
        (out / "labels" / f"{stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "")

        manifest.append({"stem": stem, "src": str(fp), "prelabels": len(lines)})
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(frames)}", flush=True)

    # classes.txt pro anotační nástroj
    ordered = [names[k] for k in sorted(names)]
    (out / "classes.txt").write_text("\n".join(ordered) + "\n")

    stats = {
        "weights": str(args.weights),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "stride": args.stride,
        "frames": len(frames),
        "frames_with_prelabels": with_det,
        "prelabels_per_class": dict(per_class),
        "prelabels_total": sum(per_class.values()),
        "classes": ordered,
        "manifest": manifest,
    }
    (out / "prelabel_stats.json").write_text(json.dumps(stats, indent=1))

    print(f"\n=== předanotace hotova ===")
    print(f"framů:                {len(frames)}")
    print(f"s předanotací:        {with_det} ({100*with_det/len(frames):.1f}%)")
    print(f"kandidátních boxů:    {sum(per_class.values())}  {dict(per_class)}")
    print(f"\nvýstup: {out}/")
    print(f"  → zkopíruj adresář na stroj s GUI a spusť:")
    print(f"    python annotate_boxes.py --dir {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
