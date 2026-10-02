#!/usr/bin/env python3
"""Vytvori variantu datasetu s oversamplovanou tridou 'knife' v TRAIN splitu.

Duvod: knife ma 3 208 instanci proti gun 14 218 (1:4.4) a projevuje se to
recallem 0.587 pri precision 0.884 - model noz oznaci spolehlive, ale
prehledne ~41 %. Oversampling train splitu je nejlevnejsi zpusob, jak
poměr vyrovnat bez novych dat.

val a test se NEOVERSAMPLUJI - jinak by metriky nebyly srovnatelne.
Obrazky se hardlinkuji, takze varianta nezere misto.
"""
import shutil, sys
from collections import Counter
from pathlib import Path

src = Path("datasets/weapons_merged")
dst = Path("datasets/weapons_knife_boost")
FACTOR = 3  # kolikrat celkem se knife obrazek objevi v train splitu

if dst.exists():
    shutil.rmtree(dst)

stats = Counter()
for split in ("train", "val", "test"):
    (dst / split / "images").mkdir(parents=True, exist_ok=True)
    (dst / split / "labels").mkdir(parents=True, exist_ok=True)
    for lbl in sorted((src / split / "labels").iterdir()):
        img = next((p for p in (src / split / "images").glob(lbl.stem + ".*")), None)
        if img is None:
            continue
        classes = {int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}
        reps = FACTOR if (split == "train" and 1 in classes) else 1
        for i in range(reps):
            sfx = "" if i == 0 else f"_ov{i}"
            di = dst / split / "images" / f"{img.stem}{sfx}{img.suffix}"
            dl = dst / split / "labels" / f"{lbl.stem}{sfx}.txt"
            try:
                di.hardlink_to(img)
            except OSError:
                shutil.copy2(img, di)
            dl.write_text(lbl.read_text())
            stats[split] += 1
            for c in [int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()]:
                stats[f"{split}_cls{c}"] += 1

(dst / "data.yaml").write_text(
    f"path: {dst.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
    "nc: 3\nnames:\n- gun\n- knife\n- melee\n")

names = ["gun", "knife", "melee"]
for split in ("train", "val", "test"):
    inst = {names[c]: stats.get(f"{split}_cls{c}", 0) for c in range(3)}
    print(f"  {split}: {stats[split]} obrazku, instance {inst}")
