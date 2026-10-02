#!/usr/bin/env python3
"""Sestaví dataset pro domain adaptation: anotované MED framy + vzorek původních dat.

Rozdělení train/val je ČASOVÉ, ne náhodné. Husté framy z videa 004 jsou od sebe
0,17 s, takže náhodné rozdělení by poslalo téměř totožné snímky do tréninku
i validace a validační čísla by byla nafouknutá.

  * Video 004 (bitka): validace = souvislý blok zdrojových framů VAL_LO..VAL_HI,
    kolem něj ochranná mezera GUARD framů, která se nepoužije vůbec.
  * Video 001 (klidná scéna, jen negativa): poslední čtvrtina framů → val.
  * Video 002 (6 framů, jen negativa): celé → val.

Časová osa: řídký frame 2 FPS číslo N = zdrojový frame N*14; hustý frame
004d_XXXXX = zdrojový frame XXXXX (ověřeno porovnáním pixelů).

Přimíchání původních dat (MIX_ORIG obrázků z weapons_merged/train) brání tomu,
aby model při dotrénování na malém jednodoménovém datasetu zapomněl `gun`
a `knife` — v MED framech jsou jen `melee`.

DŮLEŽITÉ OMEZENÍ: všechny pozitivní příklady jsou z jediné bitky o délce ~6 s,
jedné kamery a stejných lidí. Validace proto měří interpolaci uvnitř téže
události, NE zobecnění na novou scénu. Nezávislý test by vyžadoval jiné video
se zbraněmi.

Použití (na Sparku):
    python build_finetune_dataset.py --out datasets/med_finetune
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

VAL_LO, VAL_HI = 931, 946   # zdrojové framy videa 004 ve validaci (10 boxů, 18 %)
GUARD = 5                   # ±zdrojových framů kolem validačního bloku se zahodí
MIX_ORIG = 800
SEED = 0
NAMES = ["gun", "knife", "melee"]


def src_index(stem: str) -> tuple[str, int]:
    """'004_004_0065' -> ('004', 910);  '004d_00871' -> ('004', 871);  '001_001_0012' -> ('001', 168)."""
    if stem.startswith("004d_"):
        return "004", int(stem.split("_")[1])
    parts = stem.split("_")
    return parts[0], int(parts[-1]) * 14


def assign(stem: str, n001_cut: int) -> str | None:
    vid, s = src_index(stem)
    if vid == "004":
        if VAL_LO <= s <= VAL_HI:
            return "val"
        if VAL_LO - GUARD <= s < VAL_LO or VAL_HI < s <= VAL_HI + GUARD:
            return None          # ochranná mezera
        return "train"
    if vid == "002":
        return "val"
    if vid == "001":
        return "val" if s >= n001_cut else "train"
    return "train"


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/med_finetune")
    ap.add_argument("--orig", default="datasets/weapons_merged")
    ap.add_argument("--mix", type=int, default=MIX_ORIG)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    med = []
    for d in ("annotate_med", "annotate_dense"):
        for lbl in sorted(Path(d, "labels").glob("*.txt")):
            img = Path(d, "images", lbl.stem + ".jpg")
            if img.is_file():
                med.append((lbl.stem, img, lbl))

    s001 = sorted(src_index(s)[1] for s, _, _ in med if s.startswith("001"))
    n001_cut = s001[int(len(s001) * 0.75)] if s001 else 10**9

    st = Counter()
    for stem, img, lbl in med:
        split = assign(stem, n001_cut)
        if split is None:
            st["guard_zahozeno"] += 1
            continue
        n = sum(1 for l in lbl.read_text().splitlines() if l.strip())
        link(img, out / split / "images" / f"med_{stem}.jpg")
        link(lbl, out / split / "labels" / f"med_{stem}.txt")
        st[f"{split}_img"] += 1
        st[f"{split}_box"] += n
        st[f"{split}_{'pos' if n else 'neg'}"] += 1

    # vzorek původních dat → jen train (validace na původních datech se dělá zvlášť
    # na kompletním weapons_merged/val, aby šla porovnat s výchozím modelem)
    orig_imgs = sorted((Path(args.orig) / "train" / "images").iterdir())
    random.Random(SEED).shuffle(orig_imgs)
    oc = Counter()
    for img in orig_imgs[: args.mix]:
        lbl = Path(args.orig) / "train" / "labels" / (img.stem + ".txt")
        if not lbl.is_file():
            continue
        link(img, out / "train" / "images" / f"orig_{img.name}")
        link(lbl, out / "train" / "labels" / f"orig_{img.stem}.txt")
        for l in lbl.read_text().splitlines():
            if l.strip():
                oc[NAMES[int(l.split()[0])]] += 1
        st["train_orig_img"] += 1

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\n"
        "nc: 3\nnames:\n- gun\n- knife\n- melee\n")

    print("=== dataset pro domain adaptation ===")
    print(f"  video 001: val od zdrojového framu {n001_cut}")
    print(f"  video 004: val blok {VAL_LO}–{VAL_HI}, mezera ±{GUARD}")
    print(f"  MED train: {st['train_img']} framů ({st['train_pos']} s boxy, "
          f"{st['train_neg']} negativ), {st['train_box']} boxů")
    print(f"  MED val:   {st['val_img']} framů ({st['val_pos']} s boxy, "
          f"{st['val_neg']} negativ), {st['val_box']} boxů")
    print(f"  zahozeno v ochranné mezeře: {st['guard_zahozeno']}")
    print(f"  přimícháno původních: {st['train_orig_img']} obrázků, instance {dict(oc)}")
    print(f"  -> {out.resolve()}/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
