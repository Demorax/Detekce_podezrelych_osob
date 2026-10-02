#!/usr/bin/env python3
"""Najde detekce, které se opakují na stejném místě napříč framy.

Statický objekt v surveillance záběru NEMŮŽE být zbraň v ruce — lidé se hýbou.
Opakovaná detekce na identických souřadnicích je téměř jistě artefakt pozadí
(namalovaná linie na asfaltu, praskliny, skvrny).

Nalezeno empiricky: model opakovaně detekoval jako 'melee' světlou
diagonální značku na asfaltu okolo (3100-3210, 730-920) ve videu 004.

Použití:
    python find_static_fp.py --dir annotate_med --video 004 --min-repeats 3
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

W, H = None, None

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    return i / (ua + ub - i)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="annotate_med")
    ap.add_argument("--video", default="")
    ap.add_argument("--min-repeats", type=int, default=3)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()
    work = Path(a.dir)
    from PIL import Image
    names = ["gun", "knife", "melee"]

    dets = []
    for t in sorted((work / "labels").glob(f"{a.video}*.txt" if a.video else "*.txt")):
        img = next((work/"images"/(t.stem+e) for e in (".jpg",".png")
                    if (work/"images"/(t.stem+e)).is_file()), None)
        if not img: continue
        with Image.open(img) as im: w, h = im.width, im.height
        for l in t.read_text().splitlines():
            q = l.split()
            if len(q) != 5: continue
            c = int(float(q[0])); cx, cy, bw, bh = map(float, q[1:])
            dets.append((t.stem, c, [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]))

    # klastruj podle prostorove polohy
    clusters = []
    for stem, c, box in dets:
        for cl in clusters:
            if cl["cls"] == c and iou(cl["box"], box) >= a.iou:
                cl["members"].append((stem, box)); break
        else:
            clusters.append({"cls": c, "box": box, "members": [(stem, box)]})

    # POZOR: pocitat RUZNE framy, ne clenu klastru — dve prekryvajici se
    # detekce v JEDNOM framu nejsou staticky artefakt.
    for cl in clusters:
        cl["frames"] = sorted({s for s, _ in cl["members"]})
    susp = [cl for cl in clusters if len(cl["frames"]) >= a.min_repeats]
    print(f"detekci: {len(dets)}, klastru: {len(clusters)}, "
          f"podezrelych (>={a.min_repeats} RUZNYCH framu na stejnem miste): {len(susp)}")
    out = []
    for cl in sorted(susp, key=lambda c: -len(c["members"])):
        b = [round(v) for v in cl["box"]]
        stems = cl["frames"]
        print(f"\n  {names[cl['cls']]:<6} {b}  {len(stems)}x")
        print(f"    framy: {', '.join(stems[:8])}{' ...' if len(stems)>8 else ''}")
        out.append({"cls": names[cl["cls"]], "box": b, "count": len(stems), "frames": stems})
    if out:
        Path("static_fp_candidates.json").write_text(json.dumps(out, indent=1))
        print(f"\n-> static_fp_candidates.json")
        print("   POZOR: staticka detekce != automaticky FP. Zbran polozena na zemi")
        print("   je take staticka. Zkontroluj vizualne, jestli tam objekt je.")
    return 0

raise SystemExit(main())
