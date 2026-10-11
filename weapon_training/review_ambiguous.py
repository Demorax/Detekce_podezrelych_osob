#!/usr/bin/env python3
"""Ruční rozhodnutí sporných případů z asistované anotace (tyč vs. paže / stín).

Agenti nejisté případy nezapsali, poslali je do worklistu. Takové místo teď
v trénovacích datech NENÍ označené: je-li to skutečná tyč, model se učí, že tyč
je pozadí. Rozhodnutí člověka to opraví.

Krok 1 — podklady:
    python review_ambiguous.py sheets --device mps
  → review_ambiguous/list_01.png …  očíslované výřezy (červeně sporné místo,
    žlutě detekce modelu v2 s confidence) + poznámka agenta
  → review_ambiguous/rozhodnuti.csv  sloupec `rozhodnuti` vyplnit:
        T = je to tyč/zbraň     N = není (paže, stín, …)     ? = nevím

Krok 2 — zápis:
    python review_ambiguous.py apply
  T → box se přidá do labels (pokud se s místem překrývá detekce modelu, vezme
      se její přesnější box, jinak přibližný box agenta),
  N → beze změny (místo už je v datech správně jako pozadí),
  ? → frame se zapíše do annotation_results/exclusions_ambiguous.txt a
      build_finetune_dataset.py --exclude-file ho z tréninku vynechá.

Test 005 (worklist_005.json) se tu ZÁMĚRNĚ nezpracovává: změna testovací sady
podle pohledu na výsledky by ji znehodnotila.
"""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
WORKLISTS = ["annotation_results/worklist_human.json", "annotation_results/worklist_dense.json"]
OUT = HERE / "review_ambiguous"
CSV = OUT / "rozhodnuti.csv"
EXCL = HERE / "annotation_results/exclusions_ambiguous.txt"
WEIGHTS = HERE.parent / "models/weapon_detection_yolo11l_med_v2/best.pt"
MELEE = 2
TILE, COLS, ROWS = 420, 4, 3


def ann_dir(stem: str) -> Path:
    return HERE / ("annotate_dense" if stem.startswith("004d_") else "annotate_med")


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def load_items() -> list[dict]:
    items = []
    for wl in WORKLISTS:
        for x in json.loads((HERE / wl).read_text()):
            if x.get("approx_box"):
                items.append(x)
    return items


def font(size: int):
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def sheets(device: str) -> None:
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    OUT.mkdir(exist_ok=True)
    items = load_items()
    f_big, f_small = font(22), font(15)
    rows, tiles = [], []
    for n, it in enumerate(items, 1):
        img_path = ann_dir(it["stem"]) / "images" / f"{it['stem']}.jpg"
        im = Image.open(img_path).convert("RGB")
        b = it["approx_box"]
        r = model.predict(str(img_path), conf=0.05, imgsz=960, device=device, verbose=False)[0]
        dets = [(d.xyxy[0].tolist(), float(d.conf[0])) for d in r.boxes if int(d.cls[0]) == MELEE]
        best = max(((q, c) for q, c in dets if iou(q, b) > 0.1), key=lambda x: x[1], default=None)

        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        half = max(b[2] - b[0], b[3] - b[1], 260) * 0.9
        x0 = int(min(max(0, cx - half), im.width - 2 * half)); y0 = int(min(max(0, cy - half), im.height - 2 * half))
        crop = im.crop((x0, y0, int(x0 + 2 * half), int(y0 + 2 * half))).resize((TILE, TILE))
        s = TILE / (2 * half)
        d = ImageDraw.Draw(crop)
        d.rectangle([(b[0] - x0) * s, (b[1] - y0) * s, (b[2] - x0) * s, (b[3] - y0) * s],
                    outline=(230, 40, 40), width=3)
        for q, c in dets:
            d.rectangle([(q[0] - x0) * s, (q[1] - y0) * s, (q[2] - x0) * s, (q[3] - y0) * s],
                        outline=(240, 170, 0), width=2)
            d.text(((q[0] - x0) * s + 2, (q[1] - y0) * s - 18), f"{c:.2f}".replace(".", ","),
                   fill=(240, 170, 0), font=f_small)
        tile = Image.new("RGB", (TILE, TILE + 92), "white")
        tile.paste(crop, (0, 0))
        td = ImageDraw.Draw(tile)
        td.text((6, TILE + 4), f"#{n}  {it['stem']}", fill="black", font=f_big)
        for i, line in enumerate(textwrap.wrap(it.get("note", ""), 52)[:3]):
            td.text((6, TILE + 32 + i * 18), line, fill=(80, 80, 80), font=f_small)
        tiles.append(tile)
        rows.append({"id": n, "stem": it["stem"], "approx_box": json.dumps([round(v) for v in b]),
                     "model_v2_conf": f"{best[1]:.2f}" if best else "",
                     "model_v2_box": json.dumps([round(v) for v in best[0]]) if best else "",
                     "poznamka_agenta": it.get("note", ""), "rozhodnuti": ""})

    per = COLS * ROWS
    for k in range(0, len(tiles), per):
        sheet = Image.new("RGB", (COLS * (TILE + 10) + 10, ROWS * (TILE + 102) + 10), (235, 235, 235))
        for i, t in enumerate(tiles[k:k + per]):
            sheet.paste(t, (10 + (i % COLS) * (TILE + 10), 10 + (i // COLS) * (TILE + 102)))
        sheet.save(OUT / f"list_{k // per + 1:02d}.png")

    if CSV.exists():   # nepřepsat už vyplněná rozhodnutí
        old = {r["id"]: r["rozhodnuti"] for r in csv.DictReader(open(CSV, encoding="utf-8"))}
        for r in rows:
            r["rozhodnuti"] = old.get(str(r["id"]), "")
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(items)} případů, {len(tiles) // per + (len(tiles) % per > 0)} listů -> {OUT}")


def apply() -> None:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    todo = [r for r in rows if not r["rozhodnuti"].strip()]
    if todo:
        raise SystemExit(f"Nevyplněno {len(todo)} případů (např. #{todo[0]['id']}). "
                         "Vyplň T / N / ? a spusť znovu.")
    added, excluded = 0, []
    for r in rows:
        dec = r["rozhodnuti"].strip().upper()
        if dec == "T":
            box = json.loads(r["model_v2_box"]) if r["model_v2_box"] else json.loads(r["approx_box"])
            d = ann_dir(r["stem"])
            with Image.open(d / "images" / f"{r['stem']}.jpg") as im:
                w, h = im.size
            lbl = d / "labels" / f"{r['stem']}.txt"
            lines = [l for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
            dup = False
            for l in lines:
                c, cx, cy, bw, bh = map(float, l.split())
                old = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                dup |= int(c) == MELEE and iou(old, box) > 0.5
            if not dup:
                lines.append(f"{MELEE} {(box[0] + box[2]) / 2 / w:.6f} {(box[1] + box[3]) / 2 / h:.6f} "
                             f"{(box[2] - box[0]) / w:.6f} {(box[3] - box[1]) / h:.6f}")
                lbl.write_text("\n".join(lines) + "\n")
                added += 1
        elif dec == "?":
            excluded.append(r["stem"])
        elif dec != "N":
            raise SystemExit(f"#{r['id']}: neznámé rozhodnutí {r['rozhodnuti']!r} (čekám T / N / ?)")
    EXCL.write_text("\n".join(sorted(set(excluded))) + ("\n" if excluded else ""))
    n_no = sum(r["rozhodnuti"].strip().upper() == "N" for r in rows)
    print(f"přidáno boxů: {added}, beze změny (N): {n_no}, "
          f"vyloučených framů (?): {len(set(excluded))} -> {EXCL.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheets"); s.add_argument("--device", default="0")
    sub.add_parser("apply")
    a = ap.parse_args()
    sheets(a.device) if a.cmd == "sheets" else apply()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
