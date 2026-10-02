#!/usr/bin/env python3
"""Test natrénovaného detektoru na reálných MED framech (domain-gap test).

Účel: trénovací data jsou web/sportovní fotky, cílová doména je CCTV shora.
Žádná metrika z validačního splitu na tuhle otázku odpovědět neumí, protože
val má stejnou distribuci jako train. Tenhle skript pouští model na skutečné
MED framy a ukládá anotované výstupy + statistiku detekcí podle confidence.

Použití:
    python test_on_med.py --weights best.pt --frames frames_0.5_upscaled \\
                          --out med_test_out --conf 0.25
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frames", required=True, help="adresář s framy (rekurzivně)")
    ap.add_argument("--out", default="med_test_out")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = vše")
    args = ap.parse_args()

    from ultralytics import YOLO

    frames = sorted(p for p in Path(args.frames).rglob("*") if p.suffix.lower() in IMG_EXT)
    if args.max_frames:
        frames = frames[: args.max_frames]
    if not frames:
        raise SystemExit(f"Žádné framy v {args.frames}")

    out = Path(args.out)
    (out / "annotated").mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    names = model.names

    # statistika po třídách a po confidence pásmech
    per_class: Counter[str] = Counter()
    per_class_conf: dict[str, list[float]] = defaultdict(list)
    bands = {"0.25-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    per_video: Counter[str] = Counter()
    frames_with_det = 0
    detail = []

    for i, fp in enumerate(frames):
        r = model.predict(str(fp), conf=args.conf, imgsz=args.imgsz,
                          device=args.device, verbose=False)[0]
        n = 0 if r.boxes is None else len(r.boxes)
        if n:
            frames_with_det += 1
            vid = fp.parent.name
            per_video[vid] += n
            rec = {"frame": str(fp.relative_to(args.frames)), "dets": []}
            for b in r.boxes:
                cls = names[int(b.cls[0])]
                sc = float(b.conf[0])
                per_class[cls] += 1
                per_class_conf[cls].append(sc)
                for k, (lo, hi) in zip(bands, [(.25,.4),(.4,.6),(.6,.8),(.8,1.01)]):
                    if lo <= sc < hi:
                        bands[k] += 1
                        break
                rec["dets"].append({"cls": cls, "conf": round(sc, 3),
                                    "xyxy": [round(v, 1) for v in b.xyxy[0].tolist()]})
            detail.append(rec)
            # anotovaný obrázek jen když je co kreslit
            import cv2
            cv2.imwrite(str(out / "annotated" / f"{fp.parent.name}_{fp.stem}.jpg"), r.plot())
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(frames)}", flush=True)

    total = sum(per_class.values())
    lines = [
        "=== MED domain-gap test ===",
        f"váhy:      {args.weights}",
        f"framů:     {len(frames)}  (conf>={args.conf}, imgsz={args.imgsz})",
        f"framů s detekcí: {frames_with_det} ({100*frames_with_det/len(frames):.1f}%)",
        f"detekcí celkem:  {total}",
        "",
        "--- podle třídy ---",
    ]
    for c, n in per_class.most_common():
        cs = per_class_conf[c]
        lines.append(f"  {c:<7} n={n:<5} conf prum={sum(cs)/len(cs):.3f} "
                     f"min={min(cs):.3f} max={max(cs):.3f}")
    lines += ["", "--- podle confidence ---"]
    for k, v in bands.items():
        lines.append(f"  {k}: {v}")
    lines += ["", "--- podle videa ---"]
    for v, n in per_video.most_common():
        lines.append(f"  {v}: {n}")
    lines += [
        "",
        "POZOR K INTERPRETACI: MED videa téměř neobsahují zbraně, takže většina",
        "detekcí jsou pravděpodobně FALSE POSITIVES. Vysoký počet detekcí = špatná",
        "zpráva, ne dobrá. Klíčové je podívat se na anotované obrázky.",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    (out / "report.txt").write_text(report + "\n")
    (out / "detections.json").write_text(json.dumps(detail, indent=1))
    print(f"\nvýstup: {out}/  (annotated/, report.txt, detections.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
