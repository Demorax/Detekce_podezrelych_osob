# Detektory zbraní/tyčí/pálek — rešerše (2026-06-26)

> Sestaveno z paralelní rešerše (Roboflow Universe, HuggingFace, GitHub, akademické zdroje).
> **Kontext:** detekce MELEE zbraní (tyče, pálky, hole, baseball bat) v CCTV surveillance. Pipeline RT-DETR + ViTPose + YOLO11x.

## Hlavní závěr

**Hotový free pretrained melee detektor v produkční kvalitě NEEXISTUJE.** Drtivá většina "weapon detection" modelů cílí na střelné zbraně (pistol/rifle) a nože. Holá tyč/hůl/pálka jako samostatná třída je doložená mezera napříč všemi zdroji (Roboflow, HF, GitHub, MDPI/arXiv). Realistická cesta = **fine-tune vlastní YOLO11 na existujících datasetech** (nemusíš labelovat od nuly).

## Stažitelné pretrained modely (kandidáti na test)

| Model | Třídy | Formát | Pozn. |
|---|---|---|---|
| **CallMeGovos/WeaponDetection** (GitHub) | `stick` (1 třída) | YOLO11s best.pt ~19 MB, v repu | Nejbližší k melee. Web-trained, ne CCTV. **Vyzkoušet jako první.** |
| Roboflow **weapons-085lm** | Bat, Club, Crowbar, Hammer, Axe, Machete | YOLOv8 export + API, stažitelné váhy | Bohatý melee, ale web obrázky. |
| HF Subh775/Threat-Detection-YOLOv8n | threat (gun/knife) | YOLOv8n .pt | Hlavně guns/knife. |
| HF jparedesDS/cs2-yolo12m-weapon-detection | guns (CS2 hra!) | YOLO12m | Herní data, nevhodné. |
| YOLO11x COCO (current) | baseball bat (+79) | — | Co už máš. Slabé na tyče (OOD). |

## Datasety pro vlastní fine-tuning (DOPORUČENO)

| Dataset | Třídy | Velikost | Licence | Pozn. |
|---|---|---|---|---|
| **Gun-Knife-Stick** (violence-detection-2) | Knife, Pistol, Rifle, **Stick** | ~9007 | YOLO export | Top pick — má Stick. |
| **NTUT weapon-detection-yfvuq** | pistol, knife, **baseball-bat** | ~3000 | **CC BY 4.0** | Nejčistší zdroj bat dat. |
| **yolov7test/m7qso** | knife, pistol, **stick** | ~9263 | YOLO export | Velký, hodně forků (ověř že fork má stick). |
| dataset-pizme/baseball-bat | baseball-bat | ~1128 | YOLO export | Single-class augmentace. |
| ahrir/cctv-security | Sword, knife, Gun... | ~8997 | YOLO export | **CCTV doména** (Sword/knife melee). |

## Klíčový caveat — domain gap

Všechny tyto datasety jsou **web/sportovní fotky, ne CCTV shora**. Na záběry typu MED (dav, pohled shora, pohybové rozmazání, malé objekty) bude domain gap. Pro slušný výsledek pravděpodobně nutné přidat **~200-300 vlastních anotovaných CCTV framů** (z MED videí) pro domain adaptaci.

## Doporučený postup

1. **Rychlý test:** stáhnout CallMeGovos `best.pt`, pustit na MED 005 → funguje web-trained stick detektor na CCTV?
2. **Fine-tune (pokud test selže):** YOLO11 na Gun-Knife-Stick + NTUT bat (~12k obr.), ~2-4h na RTX 3090, + ~200-300 vlastních MED anotací.
3. Roboflow váhy za paywallem, ale **datasety export zdarma** v YOLO formátu (potřeba Roboflow účet + API key).

## Pozn. k integraci

Nový detektor by nahradil/doplnil současný YOLO11x v `suspicious_detector.py` (`COCO_HIGH_THREAT` baseball bat). ARMED logika (`abandoned_object_detector.py::update_armed_status`, skeleton-based) zůstává — jen dostane lepší vstupní detekce.
