# Detektor podezřelých předmětů — dokumentace

Dokumentace tréninku YOLO11l detektoru zbraní a melee předmětů pro pipeline
analýzy pohybu osob. Implementuje bod 4 z hlavního README („Detekce podezřelých
předmětů"), který byl dosud označen jako neimplementovaný.

**Trénink proveden 2026-08-02 až 2026-08-04** na NVIDIA DGX Spark.

---

## Výsledek v jednom odstavci

Natrénován **YOLO11l** detektor **3 tříd** (`gun`, `knife`, `melee`) fine-tunem
předtrénovaných COCO vah na **24 340 obrázcích** sloučených ze 6 veřejných
Roboflow datasetů. Trénink 90 epoch / 43,4 h, ukončen early stoppingem.
Na validačním splitu **mAP50 = 0,775**, mAP50-95 = 0,523
(`gun` 0,884 / `melee` 0,830 / `knife` 0,611).

Test na reálných MED framech potvrdil, že **domain transfer částečně funguje**
(model detekoval předmět v ruce osoby v davu při pohledu shora), ale také že
**model přehlédne dlouhé tyče a hole** — což je přímý důsledek toho, že taková
třída v žádném dostupném veřejném datasetu neexistuje.

Následný experiment s oversamplingem třídy `knife` (poměr srovnán z 4,4 : 1 na
1,6 : 1) **recall nezvýšil** (0,587 → 0,585) a model celkově mírně zhoršil.
Tím je vyloučeno, že slabina nožů je problémem nevyváženosti — jde o
nedostatečnou rozmanitost dat. Viz [`TRAINING_REPORT.md`](TRAINING_REPORT.md) §6a.

---

**Domain adaptation (2026-10-02).** Po ruční a asistované anotaci 252 MED framů
(56 boxů, 220 negativ) byl model dotrénován se zmrazeným backbone. Na CCTV záběrech
shora vzrostl recall `melee` z **0,10 na 0,50** (mAP50 0,138 → 0,623), na původních
datech klesl mAP50 z 0,777 na 0,755. Tento model je nasazen v pipeline. Na **nezávislé
bitce z videa 005** (67 boxů, na kterých se netrénovalo) vzrostla AP50 `melee` z 0,22 na 0,60. Viz [`TRAINING_REPORT.md`](TRAINING_REPORT.md) §6b.

## Kde co najít

| Dokument | Obsah |
|---|---|
| **[`TRAINING_REPORT.md`](TRAINING_REPORT.md)** | Hlavní zpráva: volba modelu, hardware, průběh tréninku, divergence, výsledky, domain-gap test, omezení |
| **[`DATASETS.md`](DATASETS.md)** | Všechny zkoumané datasety, licence, mapování tříd, co bylo odmítnuto a proč, konstrukce datasetu |
| **[`MED_VIDEO_SCAN.md`](MED_VIDEO_SCAN.md)** | Průchod všemi 31 MED videi: kde jsou zbraně, rvačky bez zbraní a další užitečná negativa |
| **[`ANNOTATION_SPEC.md`](ANNOTATION_SPEC.md)** | Závazná anotační pravidla: tyč vs. paže, tyč vs. stín, statické artefakty, těsnost boxů |
| **[`ANNOTATION.md`](ANNOTATION.md)** | Návod na anotaci framů pro domain adaptation — ovládání nástroje, na co si dát pozor, jak pak dotrénovat |
| [`scripts/`](scripts/) | Použité skripty (kopie) |
| [`artifacts/`](artifacts/) | Metriky, grafy, confusion matice, logy — včetně divergujícího běhu |
| [`med_test/`](med_test/) | Test na reálných MED framech: anotované snímky, JSON detekcí |

---

## Skripty a jejich role

| Skript | Účel |
|---|---|
| `download_datasets.py` | Stažení 7 Roboflow datasetů (fixované verze) |
| `inspect_class.py` | Vizuální kontrola obsahu třídy — vyřízne vzorek boxů do náhledu |
| `build_dataset.py` | Sloučení, SHA-1 deduplikace, remapování tříd, převod polygonů na boxy |
| `train_weapon_yolo.py` | Trénink / fine-tune |
| `test_on_med.py` | Domain-gap test na reálných MED framech |
| `sampler.sh` | Vzorkování tempa tréninku a paměti (diagnostika) |

Aktuální verze skriptů jsou v nadřazené složce `weapon_training/`.

---

## Pět zjištění, která ovlivnila výsledek

Pro text práce jsou tato zjištění zajímavější než samotná čísla, protože žádné
z nich nebylo patrné z dokumentace datasetů ani z metadat.

1. **Roboflow API vrací u některých datasetů jiné třídy než skutečný export.**
   Třídy je nutné čtete z exportovaného `data.yaml`.
   ([`DATASETS.md`](DATASETS.md) §1)

2. **Největší dataset (15 981 obrázků) byl nepoužitelný.** Jeho jediná třída
   `weapon` je směs pistolí, pušek a nožů — zjištěno až vizuální kontrolou
   vzorku boxů. ([`DATASETS.md`](DATASETS.md) §2.1)

3. **`optimizer='auto'` v ultralytics 8.4.115 způsobí divergenci.** Vybere
   `MuSGD(lr=0.01)`, warmup ho vyžene na 0,03 a za 3 epochy val cls_loss
   vzroste 2,5× a mAP50 spadne na třetinu. Nutné nastavit `SGD` explicitně.
   ([`TRAINING_REPORT.md`](TRAINING_REPORT.md) §4.2)

4. **Veřejné weapon datasety jsou navzájem forky** — deduplikace odstranila
   5 049 duplicit (10,9 %). Bez ní by validační mAP byla nadhodnocená.
   ([`DATASETS.md`](DATASETS.md) §5.2)

5. **Chybějící třída „tyč/hůl" má měřitelný důsledek.** Model nenašel jasně
   viditelnou dlouhou tyč v ruce osoby v MED framu, protože `melee` je
   natrénováno z 72,7 % na baseballových pálkách.
   ([`TRAINING_REPORT.md`](TRAINING_REPORT.md) §6.2)

6. **Oversampling nevyvážené třídy nepomohl** (negativní výsledek). Duplikování
   existujících obrázků nepřidává informaci — recall `knife` se nezměnil
   (0,587 → 0,585). Slabinu nelze řešit převážením, jen novými daty.
   ([`TRAINING_REPORT.md`](TRAINING_REPORT.md) §6a)

---

## Integrace do pipeline

Váhy: `models/weapon_detection_yolo11l/best.pt` (49 MB)

Provedené změny:

| Soubor | Změna |
|---|---|
| `suspicious_detector.py` | `_WEAPON_MODEL` → `models/weapon_detection_yolo11l_med/best.pt` (domain-adapted, 2026-10-02; původní model v `models/weapon_detection_yolo11l/`); `WEAPON_HIGH_THREAT` rozšířeno o `2: 'melee'`; přidány `WEAPON_CONF_PER_CLASS` |
| `abandoned_object_detector.py` | `HIGH_THREAT_LABELS` rozšířeno o `'melee'` |

`WEAPON_CONF_PER_CLASS` nastavuje **nižší práh pro `knife` (0,35)** než pro
`gun` (0,55) a `melee` (0,50). Důvod: `knife` má recall 0,587 při precision
0,884 — model nůž označí spolehlivě, ale přehlédne ~41 % z nich. U detektoru
hrozeb je false negative horší chyba než false positive, a vysoká precision
dává prostor ji obětovat za recall.

`'baseball bat'` zůstává v `HIGH_THREAT_LABELS`, protože ho produkuje COCO
YOLO11x, který je v pipeline dál kvůli zavazadlům (`backpack`, `handbag`,
`suitcase`) a na custom modelu je nezávislý.

---

## Co dál

Domain adaptation je hotová (§6b). Nejdůležitější otevřený bod je **ověřit zobecnění
na dalším videu se zbraněmi** — současné zlepšení je změřeno jen na jedné bitce.
Dál: rozhodnout 64 sporných anotací a omezit zapomínání. Podrobně
[`TRAINING_REPORT.md`](TRAINING_REPORT.md) §8.
