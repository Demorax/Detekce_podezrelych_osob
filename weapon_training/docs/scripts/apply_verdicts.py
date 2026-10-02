#!/usr/bin/env python3
"""Sloučí JSON verdikty od anotátorů a zapíše je do YOLO labelů.

Vstup: soubory /tmp/result_*.json ve formátu z docs/ANNOTATION_SPEC.md.
Výstup: přepsané labels/*.txt + progress.json + přehled sporných případů.

Pravidla:
  * `accept`     → box zůstává, jak byl
  * `retighten`  → použije se `new_box`
  * `reject`     → box se zahodí
  * `missed`     → přidá se, ale JEN pokud confidence != "low"
  * `ambiguous`  → NEZAPISUJE se, jen se vypíše do worklistu pro člověka

Bez `--write` jen reportuje a nic nemění (dry-run).
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from PIL import Image

NAME2ID = {"gun": 0, "knife": 1, "melee": 2}
MIN_SIDE = 12


def to_yolo(box, w, h) -> str | None:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, float(x1)), min(float(w), float(x2))))
    y1, y2 = sorted((max(0.0, float(y1)), min(float(h), float(y2))))
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        return None
    return (f"{(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
            f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="annotate_med")
    ap.add_argument("--results", default="/tmp/result_*.json")
    ap.add_argument("--write", action="store_true", help="bez tohoto jen dry-run")
    ap.add_argument("--min-conf", default="medium",
                    choices=["high", "medium", "low"],
                    help="minimální confidence pro zapsání 'missed' boxu")
    ap.add_argument("--exclude", default="exclusions.json",
                    help="JSON se seznamem boxů zamítnutých při kontrole; "
                         "formát [{\"stem\":..., \"box\":[x1,y1,x2,y2], \"why\":...}]")
    args = ap.parse_args()

    # Boxy zamítnuté při vizuální kontrole kontaktního listu. Přesouvají se
    # do worklistu pro člověka, nezapisují se.
    excl = []
    ep = Path(args.exclude)
    if ep.is_file():
        try:
            excl = json.loads(ep.read_text())
        except ValueError as e:
            raise SystemExit(f"{ep}: nevalidní JSON ({e})")

    def is_excluded(stem, box):
        if not box:
            return None
        for e in excl:
            if e.get("stem") != stem:
                continue
            b = e.get("box") or []
            if len(b) == 4 and all(abs(float(b[i]) - float(box[i])) <= 2 for i in range(4)):
                return e.get("why", "vyloučeno při kontrole")
        return None

    work = Path(args.dir).expanduser().resolve()
    files = sorted(glob.glob(args.results))
    if not files:
        raise SystemExit(f"Žádné výsledky: {args.results}")

    rank = {"high": 3, "medium": 2, "low": 1}
    need = rank[args.min_conf]

    rows = []
    seen_stems = set()
    dup = []
    for f in files:
        try:
            data = json.loads(Path(f).read_text())
        except ValueError as e:
            print(f"[!] {f}: nevalidní JSON ({e}) — přeskakuji")
            continue
        if isinstance(data, dict):
            data = [data]
        for r in data:
            s = r.get("stem")
            if not s:
                continue
            if s in seen_stems:
                dup.append(s)
                continue
            seen_stems.add(s)
            rows.append((Path(f).name, r))

    st = Counter()
    per_class = Counter()
    worklist = []
    plans = []

    for src, r in rows:
        stem = r["stem"]
        img = next((work / "images" / (stem + e)
                    for e in (".jpg", ".jpeg", ".png")
                    if (work / "images" / (stem + e)).is_file()), None)
        if img is None:
            print(f"[!] {stem}: obrázek nenalezen — přeskakuji")
            st["chybi_obrazek"] += 1
            continue
        with Image.open(img) as im:
            W, H = im.width, im.height

        lines = []
        for v in r.get("prelabel_verdicts", []) or []:
            verdict = (v.get("verdict") or "").lower()
            nm = (v.get("name") or "").lower()
            cid = NAME2ID.get(nm)
            st[f"verdict_{verdict or 'chybi'}"] += 1
            if cid is None:
                st["neznama_trida"] += 1
                continue
            if verdict == "accept":
                box = v.get("box")
            elif verdict == "retighten":
                box = v.get("new_box") or v.get("box")
            else:
                continue
            why = is_excluded(stem, box)
            if why:
                st["vyloucen_pri_kontrole"] += 1
                worklist.append((stem, "vyloucen_kontrolou", box, why))
                continue
            y = to_yolo(box, W, H) if box else None
            if y:
                lines.append(f"{cid} {y}")
                per_class[nm] += 1
            elif box:
                st["prilis_maly_box"] += 1

        for m in r.get("missed", []) or []:
            nm = (m.get("name") or "").lower()
            cid = NAME2ID.get(nm)
            conf = (m.get("confidence") or "medium").lower()
            st[f"missed_{conf}"] += 1
            if cid is None:
                st["neznama_trida"] += 1
                continue
            if rank.get(conf, 0) < need:
                worklist.append((stem, "missed_low_conf", m.get("approx_box"),
                                 m.get("note", "")))
                continue
            why = is_excluded(stem, m.get("approx_box"))
            if why:
                st["vyloucen_pri_kontrole"] += 1
                worklist.append((stem, "vyloucen_kontrolou", m.get("approx_box"), why))
                continue
            y = to_yolo(m.get("approx_box"), W, H)
            if y:
                lines.append(f"{cid} {y}")
                per_class[nm] += 1
            else:
                st["prilis_maly_box"] += 1

        for a in r.get("ambiguous", []) or []:
            st["ambiguous"] += 1
            worklist.append((stem, "ambiguous", a.get("approx_box"), a.get("note", "")))

        if r.get("empty"):
            st["prazdny_frame"] += 1
        plans.append((stem, lines))
        st["framu"] += 1

    print("=== SOUHRN ===")
    for k in sorted(st):
        print(f"  {k:<22} {st[k]}")
    print(f"\n  boxů k zapsání celkem: {sum(len(l) for _, l in plans)}  {dict(per_class)}")
    if dup:
        print(f"\n[!] duplicitní framy ve výsledcích (použit první): {sorted(set(dup))}")

    if worklist:
        print(f"\n=== PRO RUČNÍ ROZHODNUTÍ ({len(worklist)}) ===")
        for stem, kind, box, note in worklist:
            print(f"  {stem:<18} {kind:<14} {box}  {note[:70]}")
        Path("worklist_human.json").write_text(json.dumps(
            [{"stem": s, "kind": k, "approx_box": b, "note": n}
             for s, k, b, n in worklist], indent=1, ensure_ascii=False))
        print("  → uloženo do worklist_human.json")

    if not args.write:
        print("\n(dry-run — nic nezapsáno; přidej --write)")
        return 0

    for stem, lines in plans:
        p = work / "labels" / (stem + ".txt")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(("\n".join(lines) + "\n") if lines else "")

    pp = work / "progress.json"
    rev = set()
    if pp.is_file():
        try:
            rev = set(json.loads(pp.read_text()).get("reviewed", []))
        except ValueError:
            pass
    rev |= {s for s, _ in plans}
    n_img = len([p for p in (work / "images").iterdir()
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
    pp.write_text(json.dumps({"reviewed": sorted(rev), "total": n_img}, indent=1))
    print(f"\nZAPSÁNO: {len(plans)} labelů, zkontrolováno {len(rev)}/{n_img} framů")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
