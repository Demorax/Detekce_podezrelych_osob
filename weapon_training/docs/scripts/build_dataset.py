#!/usr/bin/env python3
"""Sloučení stažených Roboflow datasetů do jednoho YOLO datasetu o 3 třídách.

Cílové třídy:
    0 = gun    (pistole, revolvery, pušky, brokovnice)
    1 = knife  (nože, řezáky, sekáčky)
    2 = melee  (baseballové pálky, sekery, meče, kopí)

backpack/handbag/suitcase se netrénují — na ty zůstává COCO YOLO11x.

Dvě věci, které tenhle skript řeší a na kterých se dá snadno uklouznout:

1. NEBEZPEČNÉ ZAHOZENÍ LABELU.
   Když z obrázku odstraním label zbraňové třídy (např. nejednoznačné 'weapon')
   a obrázek nechám v datasetu, stane se z něj *pozadí* — a model se naučí,
   že zbraň je pozadí. Proto se rozlišuje:
     - SAFE_DROP  (person, Ball, Blood, ...) -> label zmizí, obrázek zůstane
     - jinak (neznámá/nejednoznačná třída) -> zahodí se CELÝ obrázek
   Chování pro nejednoznačné třídy jde přepnout přes --ambiguous.

2. DUPLICITY MEZI DATASETY.
   Tyhle weapon datasety jsou navzájem forky, takže stejný obrázek je často
   ve víc z nich — a klidně v jednom v trainu a v jiném ve validaci.
   Bez deduplikace by validační mAP byla nadhodnocená (leakage).
   Deduplikuje se podle SHA-1 obsahu souboru; první výskyt vyhrává,
   přičemž train má nižší prioritu než val/test (aby se val nekontaminovala).

Použití:
    python build_dataset.py --raw datasets/raw --out datasets/weapons_merged
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

TARGETS = ["gun", "knife", "melee"]
TARGET_ID = {name: i for i, name in enumerate(TARGETS)}

# Třídy, jejichž odstranění NEVADÍ — nejsou to zbraně, takže z obrázku
# může zůstat platné pozadí.
SAFE_DROP = {
    "person", "people", "hand", "ball", "blood", "victim", "aggressor",
    "violence", "stabbing", "baseball", "helmet", "glove",
}

# Mapování názvů zdrojových tříd -> cílová třída.
# Klíče se porovnávají na lowercase + strip.
CLASS_MAP = {
    # --- guns ---
    "gun": "gun", "guns": "gun", "handgun": "gun", "hand gun": "gun",
    "pistol": "gun", "pistols": "gun", "revolver": "gun",
    "rifle": "gun", "rifles": "gun", "rifles.": "gun",
    "shotgun": "gun", "ak": "gun", "m16": "gun",
    "semi automatic": "gun", "heavy gun": "gun", "heavyweapon": "gun",
    # m7qso v16: ověřeno inspect_class.py -> výhradně dlouhé palné zbraně
    # (pušky, AK, brokovnice, samopaly), žádné melee ani nože
    "heavy-weapon": "gun", "heavy weapon": "gun",
    "long guns": "gun", "guns perspective": "gun", "larga": "gun",
    # --- knives ---
    "knife": "knife", "knives": "knife", "knife_weapon": "knife",
    "knife_deploy": "knife", "cutter": "knife", "cleaver": "knife",
    "karambit": "knife",
    # 'short sword' = dýky, vizuálně nerozeznatelné od velkých nožů (ověřeno
    # náhledem merged_melee.jpg). Jako melee si konkurovaly s třídou knife;
    # sem patří lépe a navíc posilují nejslabší třídu. 'long sword' zůstává melee.
    "short sword": "knife", "dagger": "knife",
    # --- melee ---
    "baseball-bat": "melee", "baseball bat": "melee", "baseball_bat": "melee",
    "bat": "melee", "bats": "melee", "baseball-bats": "melee",
    "stick": "melee", "club": "melee", "crowbar": "melee",
    "ax": "melee", "axe": "melee", "hammer": "melee",
    "long sword": "melee", "sword": "melee",
    "katana": "melee", "machete": "melee", "machette": "melee",
    "spear": "melee",
    # dataset-pizme/baseball-bat exportuje jedinou nepojmenovanou třídu "0"
    "0": "melee",
}

# Třídy, u kterých není jasné, co znamenají. Default = zahodit celý obrázek.
AMBIGUOUS = {"weapon", "weapons", "eto", "al", "object", "threat"}

SPLIT_ALIASES = {
    "train": "train",
    "valid": "val", "val": "val", "validation": "val",
    "test": "test",
    # m7qso má navíc nestandardní adresář 'export' s dalšími 4228 obrázky.
    # Bereme ho jako train — pokud jde o kopie obrázků z valid/test,
    # deduplikace je zahodí ve prospěch val/test (viz SPLIT_PRIORITY).
    "export": "train",
}
# Nižší číslo = vyšší priorita při deduplikaci (val/test nesmí přijít o obrázek
# ve prospěch trainu, jinak vznikne leakage).
SPLIT_PRIORITY = {"val": 0, "test": 1, "train": 2}


def norm(name: str) -> str:
    return str(name).strip().lower()


def load_source_classes(ds_dir: Path) -> list[str] | None:
    y = ds_dir / "data.yaml"
    if not y.is_file():
        return None
    with y.open() as f:
        d = yaml.safe_load(f)
    names = d.get("names")
    if isinstance(names, dict):  # {0: 'a', 1: 'b'}
        return [names[k] for k in sorted(names, key=int)]
    if isinstance(names, list):
        return names
    return None


def find_split_dirs(ds_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Vrátí {split: (images_dir, labels_dir)} pro existující splity."""
    out = {}
    for child in sorted(ds_dir.iterdir()):
        if not child.is_dir():
            continue
        split = SPLIT_ALIASES.get(norm(child.name))
        if not split:
            continue
        img_d, lbl_d = child / "images", child / "labels"
        if img_d.is_dir() and lbl_d.is_dir():
            out[split] = (img_d, lbl_d)
    return out


def sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def parse_label(text: str) -> list[tuple[int, str]]:
    """-> [(class_id, 'cx cy w h')] ; ignoruje prázdné/nevalidní řádky.

    Exporty rhackathonu a baseball-v1 míchají bboxy s POLYGONY
    ('class x1 y1 x2 y2 ...', 4-87 bodů). Polygon se převádí na opsaný
    bounding box — anotace se tím zachová. Naivní vzetí prvních 4 čísel
    by dalo úplně jiný box.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue

        if len(vals) == 4:
            cx, cy, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        else:
            continue  # nesmyslný počet hodnot

        # clamp do [0,1] a zahození degenerovaných boxů
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
        if w <= 1e-6 or h <= 1e-6:
            continue
        rows.append((cid, f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="datasets/raw")
    ap.add_argument("--out", default="datasets/weapons_merged")
    ap.add_argument(
        "--ambiguous", choices=["drop-image", "as-gun", "keep-none"],
        default="drop-image",
        help="co s nejednoznačnou třídou 'weapon': zahodit obrázek (default), "
             "mapovat na gun, nebo jen smazat label a obrázek nechat (RIZIKO)",
    )
    ap.add_argument("--link", action="store_true",
                    help="hardlinkovat obrázky místo kopírování (rychlé, šetří místo)")
    args = ap.parse_args()

    raw = Path(args.raw).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not raw.is_dir():
        sys.exit(f"Neexistuje --raw: {raw}")

    ds_dirs = [d for d in sorted(raw.iterdir()) if d.is_dir()]
    if not ds_dirs:
        sys.exit(f"Žádné datasety v {raw}")

    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (out / split / kind).mkdir(parents=True, exist_ok=True)

    # --- 1) posbírat kandidáty (bez zapisování), ať jde dedup vyřešit globálně
    #        s ohledem na prioritu splitů
    candidates = []  # (split, img_path, lbl_path, ds_name, src_classes)
    unknown_classes: Counter[str] = Counter()

    for ds in ds_dirs:
        src_classes = load_source_classes(ds)
        if src_classes is None:
            print(f"[warn] {ds.name}: chybí data.yaml, přeskakuji", file=sys.stderr)
            continue
        splits = find_split_dirs(ds)
        if not splits:
            print(f"[warn] {ds.name}: nenalezeny train/valid/test, přeskakuji", file=sys.stderr)
            continue
        n = 0
        for split, (img_d, lbl_d) in splits.items():
            for img in sorted(img_d.iterdir()):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                lbl = lbl_d / (img.stem + ".txt")
                candidates.append((split, img, lbl, ds.name, src_classes))
                n += 1
        print(f"[scan] {ds.name}: {n} obrázků, třídy={src_classes}")

    # --- 2) dedup podle obsahu, val/test má přednost před trainem
    candidates.sort(key=lambda c: SPLIT_PRIORITY[c[0]])
    seen: dict[str, str] = {}
    dup_count = 0
    kept = []
    for cand in candidates:
        digest = sha1(cand[1])
        if digest in seen:
            dup_count += 1
            continue
        seen[digest] = cand[0]
        kept.append((digest, *cand))

    print(f"\n[dedup] {len(candidates)} kandidátů -> {len(kept)} unikátních "
          f"({dup_count} duplicit odstraněno)")

    # --- 3) remap labelů + zápis
    stats_inst: Counter[str] = Counter()
    stats_split: Counter[str] = Counter()
    dropped_img = Counter()
    per_ds_kept = Counter()
    name_clashes = 0

    for digest, split, img, lbl, ds_name, src_classes in kept:
        rows = parse_label(lbl.read_text()) if lbl.is_file() else []

        out_rows = []
        drop_whole = False
        for cid, rest in rows:
            if not (0 <= cid < len(src_classes)):
                drop_whole = True
                unknown_classes[f"{ds_name}:id{cid}"] += 1
                break
            sname = norm(src_classes[cid])
            target = CLASS_MAP.get(sname)
            if target:
                out_rows.append(f"{TARGET_ID[target]} {rest}")
                continue
            if sname in SAFE_DROP:
                continue  # label pryč, obrázek smí zůstat
            if sname in AMBIGUOUS:
                if args.ambiguous == "as-gun":
                    out_rows.append(f"{TARGET_ID['gun']} {rest}")
                    continue
                if args.ambiguous == "keep-none":
                    continue
                drop_whole = True
                break
            # úplně neznámá třída -> radši zahodit obrázek
            unknown_classes[f"{ds_name}:{sname}"] += 1
            drop_whole = True
            break

        if drop_whole:
            dropped_img[ds_name] += 1
            continue
        if not out_rows:
            # čisté pozadí — necháváme jen minimum, ať nezředí dataset
            dropped_img[f"{ds_name} (prázdné)"] += 1
            continue

        stem = f"{ds_name}__{img.stem}"
        dst_img = out / split / "images" / f"{stem}{img.suffix.lower()}"
        if dst_img.exists():
            name_clashes += 1
            dst_img = out / split / "images" / f"{stem}_{digest[:8]}{img.suffix.lower()}"
        dst_lbl = out / split / "labels" / f"{dst_img.stem}.txt"

        if args.link:
            try:
                dst_img.hardlink_to(img)
            except OSError:
                shutil.copy2(img, dst_img)
        else:
            shutil.copy2(img, dst_img)
        dst_lbl.write_text("\n".join(out_rows) + "\n")

        stats_split[split] += 1
        per_ds_kept[ds_name] += 1
        for r in out_rows:
            stats_inst[TARGETS[int(r.split()[0])]] += 1

    # --- 4) data.yaml
    data_yaml = {
        "path": str(out),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(TARGETS),
        "names": TARGETS,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False,
                                                  allow_unicode=True))

    # --- 5) report
    print("\n=== VÝSLEDEK ===")
    print(f"out: {out}")
    print(f"splity:    {dict(stats_split)}  (celkem {sum(stats_split.values())} obrázků)")
    print(f"instance:  {dict(stats_inst)}  (celkem {sum(stats_inst.values())})")
    print(f"\nzachováno po datasetech: {dict(per_ds_kept)}")
    print(f"zahozené obrázky:        {dict(dropped_img)}")
    if name_clashes:
        print(f"kolize jmen vyřešena hashem: {name_clashes}")
    if unknown_classes:
        print("\n[!] Neznámé / nezmapované třídy (obrázky zahozeny):")
        for k, v in unknown_classes.most_common(30):
            print(f"      {k}: {v}")
        print("    Doplň je do CLASS_MAP nebo SAFE_DROP a spusť znovu.")

    if sum(stats_split.values()) == 0:
        print("\nCHYBA: nezůstal žádný obrázek.", file=sys.stderr)
        return 1
    if stats_split.get("val", 0) == 0:
        print("\n[!] Validační split je prázdný — mAP nebude co měřit.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
