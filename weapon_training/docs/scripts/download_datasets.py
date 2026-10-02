#!/usr/bin/env python3
"""Stažení Roboflow datasetů pro detektor podezřelých předmětů.

Klíč se NEUKLÁDÁ do repa — čte se z ENV ROBOFLOW_API_KEY nebo z ~/.config/roboflow/key.

Cílové třídy (viz build_dataset.py pro mapování):
    0 = gun, 1 = knife, 2 = melee

backpack/handbag/suitcase se ZÁMĚRNĚ netrénují — na ty zůstává COCO YOLO11x,
který je má z 118k obrázků a překonal by fine-tune z ~800 obrázků.

Použití:
    python download_datasets.py [--out datasets/raw]
"""
import argparse
import os
import sys
from pathlib import Path

# (workspace, project, version, popis) — verze fixované, ať je běh reprodukovatelný
DATASETS = [
    ("ntut-zy5y0", "weapon-detection-yfvuq", 3, "NTUT: baseball-bat/knife/pistol, CC BY 4.0, nejčistší"),
    # POZOR: cabwp export má jedinou třídu 'weapon' (ne tři, jak tvrdí API metadata)
    # a inspect_class.py ukázal, že je to MIX pistolí, pušek A NOŽŮ. Nerozdělitelné
    # bez relabelingu -> build_dataset.py ho zahodí (třída je v AMBIGUOUS).
    # Ponecháno ve seznamu jen pro dokumentaci nálezu; klidně zakomentuj.
    ("weapondetection-mqgm5", "weapon-detection-cabwp", 3, "NEPOUŽITELNÝ: 1 třída = mix gun+knife"),
    ("rhackathon", "weapon-detection-aoxpz", 5, "bohaté melee: ax/sword/spear + guns"),
    ("fypit2", "weapon-detection-mhdza", 2, "pouze guns"),
    ("yolov7test-u13vc", "weapon-detection-m7qso", 16, "28 tříd, hodně šumu k zahození"),
    ("dataset-pizme", "baseball-bat", 1, "baseball-bat, třída '0'"),
    ("baseball-v1", "baseball-and-bat", 2, "Bat 3955 instancí (sport)"),
]


def read_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key.strip()
    keyfile = Path.home() / ".config" / "roboflow" / "key"
    if keyfile.is_file():
        return keyfile.read_text().strip()
    sys.exit(
        "Chybí API klíč. Nastav ROBOFLOW_API_KEY nebo ulož klíč do "
        "~/.config/roboflow/key (chmod 600)."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/raw", help="kam stahovat")
    ap.add_argument("--format", default="yolov11", help="Roboflow export format")
    args = ap.parse_args()

    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("Chybí roboflow SDK: pip install roboflow")

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=read_api_key())

    ok, failed = [], []
    for ws, proj, ver, note in DATASETS:
        dest = out_root / f"{ws}__{proj}"
        if (dest / "data.yaml").is_file():
            print(f"[skip] {ws}/{proj} už stažen -> {dest}")
            ok.append(str(dest))
            continue
        print(f"[get ] {ws}/{proj} v{ver}  ({note})")
        try:
            version = rf.workspace(ws).project(proj).version(ver)
            version.download(args.format, location=str(dest), overwrite=True)
            ok.append(str(dest))
            print(f"[ok  ] -> {dest}")
        except Exception as e:  # noqa: BLE001 - chceme pokračovat na dalších
            failed.append((f"{ws}/{proj}", f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {ws}/{proj}: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"\nHotovo: {len(ok)} OK, {len(failed)} selhalo")
    for name, err in failed:
        print(f"  FAIL {name}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
