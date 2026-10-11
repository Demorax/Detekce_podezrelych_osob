#!/usr/bin/env python3
"""Fine-tune YOLO11 na detekci zbraní a melee předmětů (3 třídy).

Volba modelu a rozlišení (proč právě takhle):
  - YOLO11l @ imgsz=960 místo YOLO11x @ 640. U zbraní v CCTV je limitem
    VELIKOST OBJEKTU, ne kapacita backbone — pálka na 40 px je malý objekt.
    Vyšší rozlišení tady vynáší víc než hlubší síť při srovnatelném čase.
  - Fine-tune předtrénovaných COCO vah, ne trénink od nuly: ~30k obrázků
    je na from-scratch málo (RT-DETR from scratch by trval 15-60 h a prohrál).

Doména: trénovací data jsou web/sportovní fotky, cílová doména je CCTV shora.
Ten domain gap imgsz=960 jen zmírňuje, neodstraňuje — na produkční kvalitu
bude potřeba dolabelovat ~200-300 vlastních MED framů (viz weapon_detectors.md).

Použití:
    python train_weapon_yolo.py --data datasets/weapons_merged/data.yaml
    python train_weapon_yolo.py --model yolo11x.pt --imgsz 1280   # větší varianta
    python train_weapon_yolo.py --resume runs/.../weights/last.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/weapons_merged/data.yaml")
    ap.add_argument("--model", default="yolo11l.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32,
                    help="-1 = auto podle VRAM")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="weapons_yolo11l_960")
    ap.add_argument("--project", default="runs/weapons")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    # POZOR: optimizer='auto' na ultralytics 8.4.115 vybere MuSGD(lr=0.01),
    # což na tomhle datasetu DIVERGUJE — val cls_loss 2.47 -> 6.12 a mAP50
    # 0.30 -> 0.10 během 3 epoch, protože warmup vyžene lr na 0.03.
    # Proto se optimizer i lr0 nastavují explicitně a NE na 'auto'.
    ap.add_argument("--optimizer", default="SGD",
                    choices=["SGD", "AdamW", "Adam", "auto"])
    ap.add_argument("--lr0", type=float, default=0.005)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    # Zmrazení prvních N vrstev. U YOLO11 je backbone vrstvy 0-10, takže
    # --freeze 11 trénuje jen neck + hlavu. Smysl: při dotrénování na malém
    # jednodoménovém datasetu omezit zapomínání (catastrophic forgetting).
    ap.add_argument("--freeze", type=int, default=0)
    # 'ram' = dekódovat a zmenšit obrázky jednou a držet je v paměti. Šetří CPU
    # (dekódování 4K JPEGů je nejdražší část načítání) — vhodné na sdíleném stroji
    # a pro malé datasety. Na výsledky tréninku vliv nemá.
    ap.add_argument("--cache", default="false", choices=["false", "ram", "disk"])
    ap.add_argument("--resume", default=None,
                    help="cesta k last.pt pro pokračování")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        data = Path(args.data).expanduser().resolve()
        if not data.is_file():
            raise SystemExit(f"Chybí data.yaml: {data}")

        model = YOLO(args.model)
        results = model.train(
            data=str(data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=args.project,
            name=args.name,
            seed=args.seed,
            deterministic=False,   # rychlejší, u detekce nepotřebujeme bitovou reprodukovatelnost
            patience=args.patience,
            amp=True,
            cache=False if args.cache == "false" else args.cache,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            warmup_epochs=args.warmup_epochs,
            freeze=args.freeze if args.freeze > 0 else None,
            cos_lr=True,
            # Augmentace mířené na domain gap web-foto -> CCTV shora:
            degrees=10.0,          # kamera shora = jiná rotace objektu
            scale=0.6,             # velké variace velikosti (objekt daleko/blízko)
            translate=0.15,
            fliplr=0.5,
            mosaic=1.0,
            close_mosaic=10,       # posledních 10 epoch bez mosaic = čistší konvergence
            mixup=0.1,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # různé osvětlení/noc
            plots=True,
            val=True,
        )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
    print(f"\n=== trénink dokončen ===")
    if save_dir:
        print(f"výstup: {save_dir}")
        print(f"váhy:   {save_dir / 'weights' / 'best.pt'}")

    # Finální validace na test splitu (train() validuje na val splitu)
    best = (save_dir / "weights" / "best.pt") if save_dir else None
    import yaml
    has_test = "test" in (yaml.safe_load(Path(args.data).read_text()) or {})
    if best and best.is_file() and has_test:
        print("\n=== test split ===")
        YOLO(str(best)).val(data=args.data, split="test", imgsz=args.imgsz,
                            device=args.device, plots=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
