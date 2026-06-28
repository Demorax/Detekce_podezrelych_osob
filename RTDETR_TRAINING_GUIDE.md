# RT-DETR Trénování od nuly - Kompletní průvodce

## 📋 Co už máš připravené

✅ Dataset: 67,572 obrázků, 71,835 anotací v COCO formátu
✅ 7 tříd: gun, knife, rifle, weapon, backpack, bag, suspicious_object
✅ Train/Val/Test splits
✅ data.yaml konfigurace
✅ RTX 3090 (24GB VRAM)

---

## 🎯 Dvě možnosti trénování

### **Možnost 1: Ultralytics (DOPORUČUJI)**
- ✅ Jednoduchá instalace a použití
- ✅ Už máš nainstalované
- ✅ Dobře dokumentované
- ⚠️ Méně kontroly nad detaily

### **Možnost 2: Oficiální RT-DETR repo**
- ✅ Plná kontrola nad tréninkem
- ✅ Původní implementace autorů
- ⚠️ Složitější setup
- ⚠️ Potřeba PyTorch, CUDA atd.

---

## 🚀 MOŽNOST 1: Ultralytics (START ZDE)

### Krok 1: Kontrola datasetu

```bash
cd C:\Users\tomas\PycharmProjects\CustomDetekceWebcamera
python -c "
import yaml
with open('datasets/suspicious_objects_merged/data.yaml') as f:
    data = yaml.safe_load(f)
    print(f'Classes: {data[\"nc\"]}')
    print(f'Names: {data[\"names\"]}')
    print(f'Train images: {data[\"total_images\"][\"train\"]}')
"
```

### Krok 2: Výběr velikosti modelu

RT-DETR má 4 varianty:

| Model | Parametry | COCO mAP | Rychlost (FPS) | VRAM |
|-------|-----------|----------|----------------|------|
| rtdetr-l | 32M | 53.0% | 114 | ~8GB |
| rtdetr-x | 67M | 54.8% | 74 | ~12GB |

**Doporučení:** Začni s `rtdetr-l` (rychlejší trénování)

### Krok 3: Spustit trénování

```bash
# Spustit připravený skript
python train_rtdetr_from_scratch.py
```

Nebo upravit pro menší/větší model:

```python
# Pro rtdetr-x (větší, přesnější)
model = RTDETR('rtdetr-x.yaml')  # Místo rtdetr-l.yaml

# Pro rtdetr-l (menší, rychlejší)
model = RTDETR('rtdetr-l.yaml')
```

### Krok 4: Monitorování trénování

Během trénování sleduj:

```bash
# V jiném terminálu - sleduj loss
tensorboard --logdir runs/rtdetr_from_scratch
```

Nebo sleduj soubory:
- `runs/rtdetr_from_scratch/suspicious_objects_300ep/results.csv` - loss hodnoty
- `runs/rtdetr_from_scratch/suspicious_objects_300ep/results.png` - grafy

### ⚠️ DŮLEŽITÉ PARAMETRY PRO RT-DETR

RT-DETR **NENÍ** YOLO! Potřebuje jiné nastavení:

| Parametr | YOLO hodnota | RT-DETR hodnota | Proč? |
|----------|--------------|-----------------|-------|
| `lr0` | 0.01 | **0.0001** | Transformer potřebuje nižší LR |
| `batch` | 16 | **4-8** | Transformer má větší paměťovou náročnost |
| `epochs` | 100 | **300** | Transformer konverguje pomaleji |
| `warmup_epochs` | 3 | **5** | Delší warmup pro stabilitu |
| `optimizer` | SGD | **AdamW** | AdamW je lepší pro transformery |
| `amp` | True | **False** | Mixed precision může způsobit NaN |

### 🐛 Troubleshooting

#### Problem 1: NaN losses
```
Epoch 1: loss=nan, box_loss=nan
```

**Řešení:**
```python
training_params['lr0'] = 0.00005  # Poloviční LR
training_params['amp'] = False    # Vypnout mixed precision
training_params['batch'] = 2      # Menší batch
```

#### Problem 2: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Řešení:**
```python
training_params['batch'] = 2      # Menší batch
training_params['workers'] = 4    # Méně workers
training_params['cache'] = False  # Vypnout cache
```

#### Problem 3: Velmi pomalý trénink
```
1 epoch trvá 2+ hodiny
```

**To je NORMÁLNÍ!** RT-DETR je transformer, není to YOLO.

Očekávaný čas na RTX 3090:
- rtdetr-l: ~30-40 hodin (300 epoch)
- rtdetr-x: ~50-70 hodin (300 epoch)

**Tipy pro zrychlení:**
- Začni s 100 epochs místo 300
- Použij rtdetr-l místo rtdetr-x
- Sniž image size na 512 (místo 640)

---

## 🔬 MOŽNOST 2: Oficiální RT-DETR repo (Pokročilé)

### Proč použít oficiální repo?

- Potřebuješ custom úpravy architektury
- Chceš experimentovat s různými backbones
- Potřebuješ publikovat research

### Setup

```bash
# 1. Clone official repo
cd C:\Users\tomas\PycharmProjects
git clone https://github.com/lyuwenyu/RT-DETR.git
cd RT-DETR

# 2. Instalace
cd rtdetr_pytorch
pip install -r requirements.txt

# 3. Install jako package
pip install -e .
```

### Příprava datasetu pro oficiální repo

Oficiální repo očekává COCO formát (už ho máš!):

```
datasets/suspicious_objects_merged/
├── annotations/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── images/
    ├── train/
    ├── val/
    └── test/
```

✅ Tohle už máš připravené!

### Vytvoř config soubor

```yaml
# configs/rtdetr/rtdetr_r50_suspicious.yml
task: detection

dataset:
  train_dataloader:
    dataset:
      type: CocoDetection
      img_folder: C:/Users/tomas/PycharmProjects/CustomDetekceWebcamera/datasets/suspicious_objects_merged/images/train
      ann_file: C:/Users/tomas/PycharmProjects/CustomDetekceWebcamera/datasets/suspicious_objects_merged/annotations/train.json
    batch_size: 4
    num_workers: 4

  val_dataloader:
    dataset:
      type: CocoDetection
      img_folder: C:/Users/tomas/PycharmProjects/CustomDetekceWebcamera/datasets/suspicious_objects_merged/images/val
      ann_file: C:/Users/tomas/PycharmProjects/CustomDetekceWebcamera/datasets/suspicious_objects_merged/annotations/val.json
    batch_size: 4

model:
  type: RTDETR
  num_classes: 7
  backbone:
    type: ResNet
    depth: 50

optimizer:
  type: AdamW
  lr: 0.0001
  weight_decay: 0.0001

training:
  epochs: 300
  warmup_epochs: 5
  eval_spatial_size: [640, 640]
```

### Trénování s official repo

```bash
cd rtdetr_pytorch
python tools/train.py -c configs/rtdetr/rtdetr_r50_suspicious.yml --use-amp
```

---

## 📊 Porovnání přístupů

| Feature | Ultralytics | Official Repo |
|---------|-------------|---------------|
| **Setup čas** | 5 minut | 1-2 hodiny |
| **Jednoduchość** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Kontrola** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Dokumentace** | Výborná | Střední |
| **Customizace** | Omezená | Plná |
| **Doporučení** | ✅ Pro produkci | ✅ Pro research |

---

## 🎯 Moje doporučení pro tebe

### Fáze 1: Proof of Concept (TEĎ)
```bash
python train_rtdetr_from_scratch.py
```

- Použij Ultralytics
- Model: rtdetr-l
- Epochs: 100 (místo 300)
- Cíl: Zjistit, jestli dataset funguje

### Fáze 2: Full Training (PO TESTU)
Pokud fáze 1 funguje:
- Zvýšit na 300 epochs
- Nebo zkusit rtdetr-x
- Nebo přejít na official repo pro více kontroly

---

## 📈 Očekávané výsledky

### Metriky během trénování

**Dobrý signál:**
```
Epoch 50: box_loss=2.5, cls_loss=1.8, mAP50=0.45
Epoch 100: box_loss=1.2, cls_loss=0.9, mAP50=0.65
Epoch 300: box_loss=0.5, cls_loss=0.4, mAP50=0.80+
```

**Špatný signál:**
```
Epoch 10: box_loss=nan  → Sniž LR!
Epoch 50: box_loss=10.0, cls_loss=8.0  → Dataset problém
Epoch 100: mAP50=0.10  → Špatné hyperparametry
```

### Konvergence

RT-DETR konverguje **POMALEJI** než YOLO:
- YOLO: ~50-100 epochs
- RT-DETR: ~200-300 epochs

**Nestresuj se**, pokud po 50 epochách mAP není super!

---

## 🔗 Užitečné odkazy

**Ultralytics:**
- [RT-DETR dokumentace](https://docs.ultralytics.com/models/rtdetr/)
- [Training mode docs](https://docs.ultralytics.com/modes/train/)
- [GitHub Issue #8933](https://github.com/ultralytics/ultralytics/issues/8933) - Training from scratch diskuze

**Oficiální RT-DETR:**
- [GitHub repo](https://github.com/lyuwenyu/RT-DETR)
- [RT-DETR paper (CVPR 2024)](https://arxiv.org/abs/2304.08069)
- [PaddleDetection configs](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr)

**Další tutoriály:**
- [RT-DETR HuggingFace tutorial](https://blog.roboflow.com/train-rt-detr-custom-dataset-transformers/)
- [DETR debugging guide](https://debuggercafe.com/rt-detr/)

---

## ⏱️ Timeline estimate

### Ultralytics rtdetr-l (DOPORUČUJI ZAČÍT TADY)

```
Setup: 5 minut
Trénování (100 epoch): ~15 hodin
Evaluace: 30 minut
CELKEM: ~16 hodin
```

### Ultralytics rtdetr-x (Pokud chceš přesnější model)

```
Setup: 5 minut
Trénování (300 epoch): ~60 hodin
Evaluace: 1 hodina
CELKEM: ~61 hodin (2.5 dne)
```

### Official repo (Pokud potřebuješ research-level kontrolu)

```
Setup + learning: 2-4 hodiny
Trénování: Similar k Ultralytics
Debug time: +2-5 hodin
CELKEM: +4-9 hodin oproti Ultralytics
```

---

## 🚦 Start checklist

Před spuštěním zkontroluj:

```bash
# 1. CUDA funguje
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 2. Dataset existuje
ls datasets/suspicious_objects_merged/images/train | head

# 3. Annotations jsou validní
python -c "
import json
with open('datasets/suspicious_objects_merged/annotations/train.json') as f:
    data = json.load(f)
    print(f'Images: {len(data[\"images\"])}')
    print(f'Annotations: {len(data[\"annotations\"])}')
"

# 4. Ultralytics je aktuální
pip install --upgrade ultralytics

# 5. Volné místo na disku (model checkpoints ~2GB každý)
```

---

## ✅ Ready to train!

```bash
# Jednoduchý start - Ultralytics
python train_rtdetr_from_scratch.py

# Monitoruj progress
watch -n 30 'tail runs/rtdetr_from_scratch/suspicious_objects_300ep/results.csv'
```

**Hodně štěstí! 🚀**
