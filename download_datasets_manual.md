# Návod na stažení datasetů pro RT-DETR

## 🔑 Krok 1: Získání Roboflow API klíče (ZDARMA)

1. Jdi na https://roboflow.com/
2. Zaregistruj se (zdarma)
3. Jdi na Account → API → Private API Key
4. Zkopíruj API klíč

Pak spusť:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="TVŮJ_API_KLÍČ_ZDE")
```

---

## 📥 Krok 2: Stažení datasetů (automaticky po získání API)

Jakmile máš API klíč, přidej ho do `download_datasets.py` na řádek 17:
```python
rf = Roboflow(api_key="TVŮJ_API_KLÍČ")  # místo rf = Roboflow()
```

A spusť:
```bash
python download_datasets.py
```

---

## 🖱️ ALTERNATIVA: Ruční stažení (bez API klíče)

Pokud nechceš registraci, můžeš datasety stáhnout ručně:

### 1. Weapon Detection (8,855 obrázků)
- URL: https://universe.roboflow.com/weapondetection-mqgm5/weapon-detection-cabwp
- Klikni "Download This Dataset" → vyberte "COCO" format
- Stáhni do: `datasets/suspicious_objects/weapon-detection-1/`

### 2. RHackathon Weapons (2,827 obrázků)
- URL: https://universe.roboflow.com/rhackathon/weapon-detection-aoxpz
- Klikni "Download" → "COCO"
- Stáhni do: `datasets/suspicious_objects/weapon-detection-2/`

### 3. FYPit2 Weapon Detection (7,900 obrázků)
- URL: https://universe.roboflow.com/fypit2/weapon-detection-mhdza
- Klikni "Download" → "COCO"
- Stáhni do: `datasets/suspicious_objects/weapon-detection-3/`

### 4. Backpack Dataset (819 obrázků)
- URL: https://universe.roboflow.com/mahakworkspace/backpack-yn8it
- Klikni "Download" → "COCO"
- Stáhni do: `datasets/suspicious_objects/backpack/`

### 5. yolov7test Weapons (9,672 obrázků)
- URL: https://universe.roboflow.com/yolov7test-u13vc/weapon-detection-m7qso
- Klikni "Download" → "COCO"
- Stáhni do: `datasets/suspicious_objects/weapon-detection-4/`

---

## 📊 Celkem datasetů:

| Dataset | Obrázky | Třídy | Velikost |
|---------|---------|-------|----------|
| weapondetection | 8,855 | gun, knife, rifle | ~2GB |
| RHackathon | 2,827 | weapon | ~500MB |
| FYPit2 | 7,900 | weapon | ~1.5GB |
| Backpack | 819 | backpack | ~200MB |
| yolov7test | 9,672 | weapon | ~2GB |
| **CELKEM** | **29,273** | - | **~6.2GB** |

---

## ⚡ Rychlá cesta (DOPORUČUJI):

1. Zaregistruj se na Roboflow (2 minuty)
2. Získej API klíč
3. Spusť `python download_datasets.py`
4. ☕ Počkej ~30 minut (stahování 6GB dat)

Pak ti ukážu, jak datasety spojit a natrénovat RT-DETR!
