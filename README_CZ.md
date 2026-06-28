# Detekce a klasifikace lidského chování

Systém pro detekci a klasifikaci lidského chování pomocí odhadu pózy z videozáznamů z dohledu. Zpracovává Motion Emotion Dataset pro detekci osob, extrakci klíčových bodů skeletu, sledování jednotlivců napříč snímky a klasifikaci chování do tří kategorií: chůze/normální, podezřelé a běh/panika.

## Přehled projektu

### Čtyřstupňová pipeline

#### 1. Extrakce snímků (`extract_frames.py`)
- Extrahuje snímky z videí rychlostí 2 FPS
- Vstup: `data/Motion_Emotion/*.mp4` (31 videí)
- Výstup: `frames_0.5/`

#### 2. Vylepšení rozlišení (Super-Resolution)
- ESPCN 4× zvětšení pomocí `models/super_resolution/ESPCN_x4.pb`
- Dodatečné předzpracování: CLAHE, ekvalizace histogramu, gamma korekce (γ=0.8)
- Výstup: `frames_0.5_upscaled/`

#### 3. Dvoustupňová detekční pipeline (`extract_skeletons_two_stage.ipynb`)

**Stupeň 1 - YOLO detekce osob:**
- Model: YOLO11x (`models/yolo/yolo11x.pt`)
- Vícenásobné průchody detekcí s různými konfiguracemi

**Postup vylepšování detekce:**

1. **Původní konfigurace** (nízká confidence → mnoho falešných detekcí)
   ```python
    configs = [
        {'conf': 0.25, 'imgsz': 1280, 'iou': 0.45},
        {'conf': 0.20, 'imgsz': 1024, 'iou': 0.4},
        {'conf': 0.30, 'imgsz': 1536, 'iou': 0.5},
        {'conf': 0.35, 'imgsz': 1792, 'iou': 0.55},
        {'conf': 0.15, 'imgsz': 2048, 'iou': 0.35},
    ]
   ```

2. **Zvýšení confidence** (snížení false positives)
   ```python
   configs = [
       {'conf': 0.65, 'imgsz': 1280, 'iou': 0.6},
       {'conf': 0.60, 'imgsz': 1024, 'iou': 0.6},
       {'conf': 0.60, 'imgsz': 1536, 'iou': 0.6},
       {'conf': 0.55, 'imgsz': 1792, 'iou': 0.7},
       {'conf': 0.40, 'imgsz': 2048, 'iou': 0.6},
   ]
   ```
   - Výsledek: `reference_img/yolo_vitpose/normalni/` - problémy s detekcí vzdálených osob

3. **NMS (Non-Maximum Suppression)** - odstranění duplicit
   - Výsledek: `reference_img/yolo_vitpose/nms/` - problém přetrvával

4. **Local Redetect** - lokální předetekce v problematických oblastech
   - Výsledek: `reference_img/local_redetect/` - vzdálené osoby stále nedetekované

5. **RT-DETR** - transformer-based detektor jako řešení
   - Model: RT-DETRv2-S, confidence 0.6
   - Výsledek: `reference_img/yolo_vs_rtdetr/` - **úspěch!**
   - RT-DETR: 13 osob, YOLO: 9 osob (+44% detekcí)

**Stupeň 2 - ViTPose extrakce skeletu:**
- Model: ViTPose-Huge na CrowdPose datasetu
- Checkpoint: `models/vitpose/vitpose-h-multi-crowdpose.pth` (2.4GB)
- Konfigurace: `models/vitpose/configs/ViTPose_huge_crowdpose_256x192_without_training.py`
- Vlastní MMPose komponenty v `models/vitpose/models/`
- Výstup: 14 nebo 17 klíčových bodů na osobu
- Výstupní soubory: `.npy` v `skeletons_yolo_11_upscaled_2/` s tvarem (num_people, 17, 2)

#### 4. Sledování a značkování (`label_skeletons.py`)
- Interaktivní GUI pro sledování jedné osoby napříč snímky
- Automatické sledování založené na centroidech s možností manuální korekce
- Anotace chování: chůze/normální (0), podezřelé (1), běh/panika (2)
- Výstup: `labeled_behaviors/*.json` (metadata sledování) a `*.npz` (sekvence skeletu + chování)

### Modely klasifikace chování

- **MLP Frame Classifier** (`clean_detekce_osob.ipynb`): Jednoduchá neuronová síť, 67% přesnost, uloženo jako `frame_classifier_mlp.keras`
- **LSTM Sequence Classifier** (`detekce_novy_pristub.ipynb`): 2vrstvý LSTM s maskováním pro temporální analýzu, uloženo jako `best_lstm_behavior_model.h5` nebo `best_lstm_behavior_model/`

## Spuštění pipeline

```bash
# 1. Extrakce snímků z videí
python extract_frames.py

# 2. Extrakce skeletů (dvoustupňový přístup - DOPORUČENO)
jupyter notebook extract_skeletons_two_stage.ipynb
# Spusťte všechny buňky pro zpracování snímků s super-resolution + dvoustupňovou detekcí

# 3. Interaktivní sledování a značkování chování
python label_skeletons.py

# 4. Trénování klasifikátoru chování
jupyter notebook detekce_novy_pristub.ipynb

# 5. Vizualizace výsledků detekce skeletu
python test_skeletons.py

# 6. Test a porovnání RT-DETR vs YOLO (NOVÉ)
python test_rtdetr.py --compare
```

## Klíčové technické detaily

### RT-DETR - Alternativní detektor

**Model:** RT-DETRv2-S (ResNet-18), 48.1 AP
**Umístění:** `models/rtdetr/`

**Výhody oproti YOLO** (conf=0.6):
- +44% detekcí (13 vs 9 osob)
- +32% rychlejší (0.340s vs 0.503s)
- +11% vyšší spolehlivost (0.743 vs 0.670)
- 100% pokrytí YOLO + navíc vzdálené osoby

**Použití:**
```python
from test_rtdetr import RTDETRDetector

detector = RTDETRDetector(
    'models/rtdetr/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
    'models/rtdetr/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth'
)
boxes = detector.detect(image_path, conf_threshold=0.6)
```

**Test:** `python test_rtdetr.py --compare` (vytvoří statistiky + grafy)

### ViTPose vlastní integrace

- Vlastní MMPose registry komponenty v `models/vitpose/models/`:
  - ViT backbone: `backbone/vit.py`
  - TopDown detektor: `detectors/top_down.py`
  - Heatmap head: `head/topdown_heatmap_simple_head.py`
  - Builder: `builder.py`
- **Důležitá úprava v `vit.py`**: Řádek s `super().init_weights(pretrained, patch_padding=self.patch_padding)` je zakomentován, aby se předešlo závislostem trénovací pipeline
- Konfigurační soubory upraveny k vyloučení transformací specifických pro trénování

### Podpora OpenCV CUDA

Tento projekt vyžaduje OpenCV sestavené s podporou CUDA pro GPU-akcelerované super-resolution. Instalační kroky jsou zdokumentovány v `materiály.md`:

1. Odstranit CPU-only OpenCV wheels
2. Zkopírovat custom-built `cv2.pyd` do Python site-packages
3. Zkopírovat OpenCV DLLs, CUDA 12.2 DLLs a cuDNN 8.9.7 DLLs do site-packages
4. Ověřit podporu CUDA: `cv2.cuda.getCudaEnabledDeviceCount()` by mělo vrátit > 0

### Tok dat

```
MP4 videa → extract_frames.py → JPEG snímky (2 FPS)
           → Super-Resolution (ESPCN 4×) → Vylepšené snímky
           → YOLO11x / RT-DETR (detekce osob) → Bounding boxy
           → ViTPose-Huge (odhad pózy) → Klíčové body skeletu (.npy)
           → label_skeletons.py (sledování + značkování) → Značené sekvence (.json + .npz)
           → Trénování LSTM → Klasifikátor chování (.h5)
```

## Známé problémy a omezení

1. **Výrazná nevyváženost tříd**: Trénovací data jsou ~96% chůze, ~4% běhu, ~0% podezřelého chování
2. **Malý značkovaný dataset**: Zatím pouze 4 sekvence osob (196 celkových snímků)
3. **Omezení paměti**: Velké modely (6GB+ pro ViTPose) mohou způsobit OOM chyby na strojích s omezenou VRAM
4. **Hardcodované cesty**: Mnoho skriptů používá relativní cesty, které může být potřeba upravit
5. **Chybějící soubor requirements**: Závislosti musí být instalovány manuálně
6. **Detekce vzdálených osob**: YOLO11x má problémy s detekcí malých/vzdálených osob - zvažte použití RT-DETR

## Závislosti

Hlavní knihovny (requirements.txt neexistuje - instalujte manuálně):
- PyTorch (s podporou CUDA)
- TensorFlow 2.10+
- MMPose, MMEngine
- ultralytics (YOLO)
- OpenCV (custom CUDA build)
- NumPy, SciPy, scikit-learn
- Matplotlib, Seaborn
- tkinter (pro GUI)
- Pillow, tqdm
- PyYAML (pro RT-DETR)
- onnx (pro RT-DETR)

Hardware: GPU s podporou CUDA je vyžadováno pro efektivní zpracování.

## Struktura modelu

```
models/
├── vitpose/          # ViTPose pose estimation
│   ├── configs/
│   ├── models/
│   └── vitpose-h-multi-crowdpose.pth (2.4GB)
├── yolo/             # YOLO object detection
│   └── yolo11x.pt
├── super_resolution/ # ESPCN upscaling
│   └── ESPCN_x4.pb
└── rtdetr/          # RT-DETR detection (NOVÉ)
    ├── configs/
    ├── src/
    └── rtdetrv2_r18vd_120e_coco_rerun_48.1.pth (77.4MB)
```

## Git poznámky

- Aktuálně na větvi `master` (není nakonfigurována main větev)
- Mnoho generovaných souborů (.npy, .jpg, váhy modelů) je sledováno v gitu
- Nesledované adresáře obsahují výstupní data: `annotation/`, `skeletons_yolo_11_upscaled_2/`, atd.
- Nedávné commity ukazují probíhající práci na dvoustupňové detekční pipeline a optimalizaci paměti

## Alternativní přístupy

**Single-stage YOLO-pose** (`extract_skeletons.py`): Starší přístup s přímou YOLO detekci pózy. Zahrnuje odstranění duplikátů pomocí centroidů a podobnosti skeletu. Preferovaný je dvoustupňový přístup.

**RT-DETR vs YOLO** (`reference_img/yolo_vs_rtdetr`): RT-DETR převyšuje YOLO11x - více detekcí (+44%), rychlejší (+32%), spolehlivější (+11%). **Doporučeno pro scény s více lidmi a vzdálenými osobami.**

## Vizualizace a testování

### Test detekcí
```bash
# Základní test RT-DETR
python test_rtdetr.py

# Porovnání RT-DETR vs YOLO s grafy (confidence 0.6)
python test_rtdetr.py --compare
```

### Výsledky testování (confidence: 0.6)

| Metrika | RT-DETR | YOLO11x | Rozdíl |
|---------|---------|---------|--------|
| Detekce | 13 osob | 9 osob | +44% |
| Čas | 0.340s | 0.503s | -32% |
| FPS | 2.9 | 2.0 | +45% |
| Spolehlivost | 0.743 | 0.670 | +11% |

**Shoda:** 9 spárováno (IoU: 0.891), 4 pouze RT-DETR, 0 pouze YOLO

➡️ RT-DETR detekuje všechny YOLO osoby + 4 další, rychleji a s vyšší spolehlivostí.

### Výstupní soubory
- `rtdetr_test_result.jpg` - vizualizace RT-DETR detekcí
- `rtdetr_vs_yolo_comparison.jpg` - side-by-side porovnání (také v `reference_img/yolo_vs_rtdetr/`)
- `rtdetr_vs_yolo_graphs.jpg` - statistické grafy (také v `reference_img/yolo_vs_rtdetr/`):
  1. Počet detekcí
  2. Čas inference
  3. FPS
  4. Rozdělení spolehlivosti
  5. Průměrná spolehlivost
  6. Rozdělení ploch bbox
  7. Shoda detekcí (koláčový graf)

## Referenční obrázky

- `reference_img/yolo_vitpose/normalni/` - původní YOLO detekce (problémy s levým horním rohem)
- `reference_img/yolo_vitpose/nms/` - pokus s NMS
- `reference_img/local_redetect/` - pokus s lokální redetekci
- `reference_img/yolo_vs_rtdetr/` - porovnání YOLO vs RT-DETR (RT-DETR vyhrává!)
