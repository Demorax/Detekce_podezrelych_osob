#!/usr/bin/env python3
"""Vyhodnocení detektorů na NEZÁVISLÉM testovacím setu z MED videa 005.

Video 005 je stejný dvůr a kamera jako 004, ale jiná bitka. Na 005 se nikdy
netrénovalo, takže měří zobecnění mimo jednu událost (což MED validace z videa
004 neumí). 48 framů, 67 ověřených boxů `melee`, viz docs/TRAINING_REPORT.md §6c.

POZOR: neladit podle tohoto setu confidence thresholdy — test by přestal
být nezávislý.

Použití (stačí CPU/MPS, 48 obrázků):
    python eval_test005.py
"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODELS = {
    "vychozi": ROOT.parent / "models/weapon_detection_yolo11l/best.pt",
    "ft_vse": ROOT / "runs/detect/runs/weapons/med_domain_adapt/weights/last.pt",
    "ft_zmrazeny": ROOT.parent / "models/weapon_detection_yolo11l_med/best.pt",
}


def main():
    yaml = ROOT / "annotate_005" / "test005.yaml"
    yaml.write_text(f"path: {ROOT / 'annotate_005'}\ntrain: images\nval: images\n"
                    "nc: 3\nnames: [gun, knife, melee]\n")
    for tag, w in MODELS.items():
        if not w.is_file():
            print(f"{tag}: váhy nenalezeny ({w})")
            continue
        m = YOLO(str(w))
        for conf in (0.001, 0.40):
            r = m.val(data=str(yaml), split="val", imgsz=960, batch=8,
                      verbose=False, plots=False, workers=0, conf=conf)
            idx = list(r.box.ap_class_index)
            if 2 not in idx:
                print(f"{tag} conf={conf}: melee nedetekováno")
                continue
            i = idx.index(2)
            print(f"{tag:<12} conf={conf:<5} melee P={r.box.p[i]:.3f} R={r.box.r[i]:.3f} "
                  f"AP50={r.box.ap50[i]:.3f} AP50-95={r.box.ap[i]:.3f}")


if __name__ == "__main__":
    main()
