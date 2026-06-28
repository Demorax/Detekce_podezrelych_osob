# Plán: Pre-incident detection podezřelého chování

> **Sestaveno:** 2026-05-09
> **Pivot z:** binary fight/non-fight LSTM (reactive) **→** abandoned object + loitering + person-object interaction (proactive)
> **Motivace:** cíl projektu je detekovat podezřelé osoby _předtím_ než něco udělají (např. odložení batohu a odchod, podezřelé zdržování), ne až ve chvíli incidentu. Současný LSTM trénovaný na RWF-2000 (5s clips s binary fight label) tohle ze své podstaty nezachytí.

---

## Proč pivot

| Aktuální stav | Cílový stav |
|---|---|
| RWF binary classifier: "v posledních 5 sekundách probíhá rvačka?" | "Tato osoba se chová podezřele a může za chvíli něco udělat" |
| Reactive: alert AŽ když incident probíhá | Proactive: alert na **precursors** (varovné signály) |
| Skeleton-only | Skeleton + objects + interactions + temporal patterns |
| Krátké okno (15 framů ≈ 3s) | Dlouhý kontext (30+ sekund tracking) |

## Tři reálné pre-incident úlohy

| Úloha | Konkrétní příklad | Detection logic |
|---|---|---|
| **Abandoned object** | Osoba má batoh, položí ho, odejde. Batoh zůstane | Track objektu + osoby; když objekt > N sekund bez nejbližší osoby v radiu R → alert |
| **Loitering** | Osoba se nehne z místa > 60 sekund, ohlíží se | Track osoby; když průměrná pozice nemění víc než D pixelů za T sekund → alert |
| **Suspicious carrying** | Osoba nese zbraň (nůž, pálka, kladivo) | Per-frame detekce + temporální konsistence (objekt detekován v ≥ N framech v ruce osoby) |

---

## Architektura

```
                ┌─────────────────────────┐
                │ Vstupní video (CCTV)    │
                └─────────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
       ┌────────────┐  ┌────────────┐  ┌──────────────┐
       │  RT-DETR   │  │  YOLO11x   │  │ Custom YOLO  │
       │ (osoby)    │  │ (COCO obj) │  │ (gun/knife)  │
       └─────┬──────┘  └─────┬──────┘  └──────┬───────┘
             │ ✓ MÁME        │ ✓ MÁME         │ ✓ MÁME
             │               │                │
             ▼               ▼                ▼
       ┌─────────────────────────────────────────┐
       │ ViTPose (skeletony pro detekované osoby)│ ✓ MÁME
       └────────┬────────────────────────────────┘
                │
                ▼
       ┌─────────────────────────────────────────┐
       │ Person tracker  +  Object tracker (NOVÉ)│
       │ (greedy centroid + IoU)                 │
       └────────┬────────────────────────────────┘
                │
                ▼
       ┌─────────────────────────────────────────┐
       │ Person-Object Association (NOVÉ)         │
       │ - kdo nese co (overlap bbox / wrist)    │
       │ - když přestane nést → "drop event"     │
       └────────┬────────────────────────────────┘
                │
                ▼
       ┌─────────────────────────────────────────┐
       │ Rule-based classifier (NOVÉ)             │
       │ - Abandoned object rule                  │
       │ - Loitering rule                         │
       │ - Suspicious carrying rule               │
       └────────┬────────────────────────────────┘
                │
                ▼
       ┌─────────────────────────────────────────┐
       │ Alert + vizualizace                      │
       └──────────────────────────────────────────┘

       (Opционálně paralelně:)
       ┌─────────────────────────────────────────┐
       │ LSTM fight classifier (z V2)             │ ✓ MÁME
       │ → fallback pro fight-in-progress         │
       └──────────────────────────────────────────┘
```

---

## Doporučené datasety pro pre-incident detection

### ⭐ Top picks pro tento problém

#### 1. **ABODA** (Abandoned Object Dataset)

- **Autor:** Kevin Lin (kevinlin311tw), IIS Sinica
- **Velikost:** 11 video sekvencí (krátké, ~minuty každá)
- **Co obsahuje:** real CCTV scény s odloženými předměty, různá obtížnost (crowded vs sparse, různé světlo)
- **Annotace:** ground truth bounding boxy + labels pro abandoned events
- **Stažení:** [github.com/kevinlin311tw/ABODA](https://github.com/kevinlin311tw/ABODA), [oficiální stránka](http://imp.iis.sinica.edu.tw/ABODA/index.html)
- **Licence:** research, free
- **Použití pro nás:** primární eval set pro abandonment detection rule

#### 2. **PETS 2006** (Performance Evaluation of Tracking and Surveillance)

- **Velikost:** 7 multi-sensor sekvencí (každá 4 kamery)
- **Co obsahuje:** **left luggage scenarios** s rostoucí složitostí
- **Annotace:** XML s ground truth events
- **Stažení:** [cvg.reading.ac.uk/PETS2006/data.html](http://www.cvg.reading.ac.uk/PETS2006/data.html), FTP `ftp.pets.rdg.ac.uk` (anonymous)
- **Licence:** free for academic research (UK ICO approved), copyright ISCAPS consortium
- **Použití pro nás:** klasický benchmark — důležité pro BP citaci ("evaluated on PETS 2006")

#### 3. **PETS 2007**

- **Velikost:** 9 sekvencí, 4 kamery
- **Co obsahuje:** rozšíření PETS 2006 o **loitering scenarios** (osoba zůstává > 60 sekund) + theft of left luggage
- **Stažení:** [cvg.reading.ac.uk/PETS2007/data.html](http://www.cvg.reading.ac.uk/PETS2007/data.html)
- **Použití pro nás:** **dual purpose — left luggage + loitering** v jednom datasetu

#### 4. **AVSS 2007 (i-LIDS subset)**

- **Velikost:** 15 sekvencí ze 3 stanic londýnského metra
- **Co obsahuje:** abandoned baggage scénáře, **3 obtížnostní úrovně** (easy / medium / hard)
- **Charakteristika:** crowded subway, occlusion, perspektivní distorze
- **Stažení:** [eecs.qmul.ac.uk/~andrea/avss2007_d.html](https://www.eecs.qmul.ac.uk/~andrea/avss2007_d.html)
- **Licence:** research only, vyžaduje citaci
- **Použití pro nás:** challenging eval — opravdu husté CCTV scény

### Komplementární datasety

#### 5. **UCF-Crime** — částečně relevantní (už máme jako reference)

- Subsety **Stealing**, **Shoplifting**, **Burglary** obsahují **precursor moments** (loitering, surveillance, carry suspicious item) protože videa jsou _untrimmed_
- 128 hodin total, ale můžeme vzít jen relevantní subset (~10-20 GB)
- Test set má frame-level temporal annotations začátku/konce anomálie

#### 6. **NWPU Campus** (Cao et al. 2023)

- 547 videí, 28 anomaly tříd
- Frame-level annotations
- **Anomaly anticipation** task (předpověď budoucích anomálií) — perfektní akademická vazba
- [github.com/ChangsongCao/NWPU-Campus-Dataset](https://github.com/ChangsongCao/NWPU-Campus-Dataset)

#### 7. **VIRAT Video Dataset** (DARPA)

- 46 activity types, 7 object types
- **Activity "abandoning object"** přímo v annotacích
- [viratdata.org](https://viratdata.org/)
- Velký, komplexní outdoor surveillance

#### 8. **CCTV-KD** (Korzo + Düsseldorf, 2024)

- 570 obrázků, 11 104 anotací (7890 person + 3214 baggage)
- Real public-space scenarios
- Annotated for object detection training

#### 9. **Long-term Thermal Drift Dataset**

- Thermal CCTV s anotovaným loitering chováním
- Pro nás méně relevantní (thermal ≠ RGB), ale dobré reference

---

## Implementační fáze

### Fáze 1 — Rule-based detector (1-2 dny práce)

**Co postavit:**
1. **Object tracker** — analogický person trackeru, ale pro suspicious objekty z `suspicious_detector.py`
   - Greedy IoU tracking (objekty se moc nehýbou, IoU > 0.5)
   - Krátký max_missed (objekt nezmizí na dlouho)
2. **Person-object association**
   - Pro každou osobu: nejbližší objekt v radiu R (např. 2× šířka bbox osoby)
   - Track ownership over time: dict[object_id → owner_person_id]
   - Detect "drop event": ownership → None v rámci 1-3 framů
3. **Abandonment rule**
   - Po drop event: monitor objekt
   - Pokud > N sekund (typ. 30s) bez nového ownera v radiu → **alert: abandoned**
4. **Loitering rule**
   - Pro každou osobu: variance pozice za posledních T sekund
   - Pokud variance < threshold → **alert: loitering**
5. **Suspicious carrying rule** (už částečně máme)
   - Když osoba nese gun/knife/baseball bat ve > N framech → **alert: weapon**

**Output:** funkční prototyp na 1-2 testovacích videích, vizuální alert overlay.

### Fáze 2 — Eval na labeled datasets (1-2 dny)

**Postup:**
1. Stáhnout ABODA + PETS 2006/2007
2. Procesovat přes pipeline (extract frames → detector → tracker → rules)
3. Spočítat metriky:
   - **TP**: správně detekoval abandonment / loitering podle ground truth
   - **FP**: alarm když ground truth neříká abandonment
   - **FN**: ground truth říká abandonment, ale detektor nedetekoval
   - **Latency**: jak rychle po události (ground truth) přišel alert
4. Per-dataset confusion matrix + summary tabulka

**Output:** tabulka výsledků pro BP (něco jako "Recall 85%, Precision 78%, mean latency 8.5s")

### Fáze 3 — ML augmentation (volitelně, podle Fáze 2 výsledků)

Pokud rule-based dosáhne >80% accuracy → spokojit se.
Pokud <70% → přidat ML:
- Trénovat malý classifier nad rule-based features (was object alone for X sec, was loitering, etc.)
- Nebo full anomaly detection (autoencoder na skeleton sekvencích)

---

## Eval strategie pro BP

**Pro každý dataset reportovat:**

| Dataset | Task | Total events | TP | FP | FN | Precision | Recall | F1 | Mean latency |
|---|---|---|---|---|---|---|---|---|---|
| ABODA | Abandoned object | ~22 | ? | ? | ? | ? | ? | ? | ? |
| PETS 2006 | Left luggage | 7 | ? | ? | ? | ? | ? | ? | ? |
| PETS 2007 | Loitering + LL | ~14 | ? | ? | ? | ? | ? | ? | ? |
| AVSS 2007 | Abandoned baggage | 15 | ? | ? | ? | ? | ? | ? | ? |

**+ Cross-dataset eval:** trénovat thresholdy na PETS, testovat na ABODA, atd.

---

## Konkrétní další kroky

1. **Stáhnout ABODA** (~MB, GitHub clone) → ověřit, že frames + GT lze parsovat
2. **Stáhnout PETS 2006** (~GB, FTP) → strukturovat do `data/PETS2006/`
3. **Postavit `object_tracker.py`** — na vzoru `track_persons` v `run_rwf_batch.py`
4. **Postavit `person_object_association.py`** — overlap-based ownership matching
5. **Postavit `pre_incident_detector.py`** — orchestrátor s 3 rules (abandonment, loitering, weapon)
6. **Test na 1-2 videích z ABODA** — quick sanity
7. **Full eval Fáze 2**
8. **Iterace** podle výsledků

---

## Knihovny / metodiky relevantní

- **DeepSORT** — proven object tracker, použít místo simple greedy IoU pro robustnost
- **ByteTrack** — modernější tracker, lepší při occlusion
- **Norfair** — lightweight Python tracker, snadná integrace

Pro Phase 1 stačí simple IoU tracker (jako u person tracking). Phase 3 případně upgrade na DeepSORT/ByteTrack.

---

## Vztah k existujícímu LSTM modelu

LSTM V2 (best_lstm_pytorch.pt) zůstává jako **fallback / second opinion**:
- Pre-incident detector spustí na full track historie
- LSTM dává binary score na poslední 15 framů (3 sec)
- **Combined alert:** `pre_incident_alert OR lstm_fight_score > 0.7`

Takhle si nezahazuješ předchozí práci, ale máš dva nezávislé signály.

---

## 🎯 Výsledky implementace (2026-05-09)

### Implementované komponenty

- ✅ `abandoned_object_detector.py` — full pipeline: PersonTracker (centroid) + ObjectTracker (IoU) + person-object association (proximity) + abandonment rule + visualization
- ✅ `suspicious_detector.py::better_object_detection()` — multi-pass YOLO11x detekce (5 image variant: original, shadow_norm, CLAHE, hist_eq, gamma) s UNION+NMS strategií
- ✅ `eval_caviar_rules.py` — eval skript co krmí GT bboxy z CAVIAR XML do rule logiky (decoupled detection quality from rule correctness)
- ✅ ESPCN_x4 / INTER_CUBIC upscale support (`--upscale` flag)

### Eval výsledky

#### A) CAVIAR — rule logic isolated test (GT bboxes as input)

**18 scénářů:** 5 LeftBag* + 4 Fight* + 3 Walk + 4 Browse + 2 Meet

| Metric | Value |
|---|---|
| Precision | **100%** (0 false positives) |
| Recall | **75%** (3 ze 4 GT abandonment events) |
| Mean TP latency | **8.7 s** (range 6.9 – 11.5 s) |
| TN baselines | 14/14 ✓ |

Per-scenario:
- LeftBag (GT drop frame 945) → Alert frame 1117, latency +6.9s ✓ TP
- LeftBag_AtChair (GT 455) → Alert 650, +7.8s ✓ TP
- LeftBox (GT 571) → Alert 858, +11.5s ✓ TP
- LeftBag_PickedUp (GT 503) → no alert ✗ FN (semantically correct: bag was picked up before threshold)
- All Walk/Browse/Fight/Meet baselines: 0 alerts ✓ TN

Settings: `abandon_sec=5, radius_factor=2.0, min_ownership_frames=2, max_obj_missed=15`.

#### B) ABODA — full pipeline test (RT-DETR + YOLO11x detection + rule)

**video1.avi** (640×480 → 2560×1920 pomocí 4× upscale):
- Detection: YOLO11x detekoval backpack 0.55, suitcase 0.20, handbag 0.16
- 2 abandonment alerts at frame 1710 (57.1s, handbag) + 1998 (66.7s, suitcase)
- Visuální ověření: oba alerty jsou correct — viditelný batoh na podlaze v zorném poli kamery
- Processing time: 8.6 min na 73-sec video při 4× upscale + multi-pass

**video2-11**: batch processing v běhu (~2 hodiny total compute).

### Limity zjištěné

1. **CAVIAR low-res (384×288) je out-of-distribution pro YOLO11x** — i s 4× upscale YOLO nedetekuje batohy na patterned floor pod fisheye úhlem. Důvod: COCO training data jsou close-up fotky, ne CCTV scény.
2. **ESPCN_x4 dnn_superres není dostupné** v aktuálním OpenCV buildu (chybí `cv2.dnn_superres`). Funguje INTER_CUBIC fallback (matematicky stejný počet pixelů, ale bez ESPCN learned upsampling).
3. **ABODA nemá public GT annotations** → kvalitativní (ne kvantitativní) eval.

### HD CCTV datasety k integraci (kandidáti pro fázi 3)

| Dataset | Resolution | Velikost | Status | Vhodnost |
|---|---|---|---|---|
| **NWPU Campus** (CVPR 2023) | **1920×1080** + 1280×960 + jiné | 76.6 GB | Public, Google Drive / Baidu | ⭐⭐⭐⭐⭐ — modern, 28 anomaly classes, frame-level GT |
| **MSAD** (NeurIPS 2024) | **1920×1080** | unknown | Application form required | ⭐⭐⭐⭐ — newest, 14 scenarios, 11 anomaly types |
| **VIRAT Ground 2.0** | HD outdoor | velký | Public + Protection Agreement | ⭐⭐⭐ — explicit "abandoning object" activity |
| **DCSASS** | varies | | Kaggle | ⭐⭐ — UCF Crime subset |

**Pozn. k disku:** aktuálně 50 GB volné. NWPU full (76 GB) se nevejde — potřeba (a) freeing space, (b) externí disk, (c) stáhnout jen test split.

### Status pre-incident detection práce

- ✅ Phase 1 (rule-based detector): **HOTOVO** (`abandoned_object_detector.py` + `eval_caviar_rules.py`)
- ✅ Phase 2 (eval na labeled datasets): **HOTOVO** pro CAVIAR (quantitative) + ABODA (qualitative)
- 🚧 Phase 2 extension: HD dataset eval (NWPU Campus / MSAD — vyžaduje dodatečné stažení)
- ⏳ Phase 3 (ML augmentation): pending — pouze pokud rule-based potřebuje boost
