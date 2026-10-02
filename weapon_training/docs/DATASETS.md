# Datasety pro detektor podezřelých předmětů — kompletní rozpis

Dokumentace všech datasetů, které byly pro trénink zkoumány, s odůvodněním
přijetí nebo odmítnutí. Stav k 2026-08-04.

Zdroj: [Roboflow Universe](https://universe.roboflow.com). Přístup přes Roboflow
API (klíč uložen mimo repozitář v `~/.config/roboflow/key`).

---

## 1. Metodické varování: API metadata nesouhlasí s exportem

**Klíčové zjištění, které ovlivnilo celý výběr.** Roboflow API endpoint
`https://api.roboflow.com/{workspace}/{project}` vrací seznam tříd projektu,
který se u některých datasetů **liší od tříd ve skutečně exportovaném
`data.yaml`**. Rozdíly nalezené v této práci:

| Dataset | Třídy podle API | Třídy v exportu (YOLOv11) |
|---|---|---|
| `weapon-detection-cabwp` | `weapon`, `Handgun`, `person` (3) | `weapon` (1) |
| `weapon-detection-m7qso` | 28 tříd včetně `Blood`, `Victim`, `Aggressor`, `violence` | `gun`, `heavy-weapon`, `knife` (3) |

U `m7qso` byl rozdíl ve prospěch exportu (28 zašuměných tříd → 3 čisté),
u `cabwp` naopak (3 rozlišitelné → 1 nerozlišitelná).

**Důsledek pro metodiku:** třídy je nutné čtete z exportovaného `data.yaml`,
nikdy z API metadat ani z popisu datasetu na webu. Skript `build_dataset.py`
proto načítá třídy výhradně z `data.yaml` každého stáhnutého datasetu.

---

## 2. Použité datasety (7 stáhnuto, 6 použito)

Verze jsou fixované pro reprodukovatelnost běhu.

| # | Dataset | Verze | Obrázků<br>(export) | Třídy v exportu | Licence | Použito |
|---|---|---|---|---|---|---|
| 1 | [`ntut-zy5y0/weapon-detection-yfvuq`](https://universe.roboflow.com/ntut-zy5y0/weapon-detection-yfvuq) | 3 | 3 000 | `baseball-bat`, `knife`, `pistol` | CC BY 4.0 | ✅ |
| 2 | [`rhackathon/weapon-detection-aoxpz`](https://universe.roboflow.com/rhackathon/weapon-detection-aoxpz) | 5 | 6 751 | 14 tříd (viz níže) | — | ✅ částečně |
| 3 | [`yolov7test-u13vc/weapon-detection-m7qso`](https://universe.roboflow.com/yolov7test-u13vc/weapon-detection-m7qso) | 16 | 16 619 | `gun`, `heavy-weapon`, `knife` | — | ✅ |
| 4 | [`baseball-v1/baseball-and-bat`](https://universe.roboflow.com/baseball-v1/baseball-and-bat) | 2 | 5 837 | `Ball`, `Bat` | CC BY 4.0 | ✅ jen `Bat` |
| 5 | [`dataset-pizme/baseball-bat`](https://universe.roboflow.com/dataset-pizme/baseball-bat) | 1 | 1 128 | `0` (bez názvu) | — | ✅ |
| 6 | [`fypit2/weapon-detection-mhdza`](https://universe.roboflow.com/fypit2/weapon-detection-mhdza) | 2 | 1 085 | `Gun`, `Guns`, `Handgun` | — | ✅ |
| 7 | [`weapondetection-mqgm5/weapon-detection-cabwp`](https://universe.roboflow.com/weapondetection-mqgm5/weapon-detection-cabwp) | 3 | 15 981 | `weapon` | — | ❌ **odmítnut** |

Nejkvalitnější zdroj je **NTUT** (#1): čisté, vyvážené třídy
(`baseball-bat` 1 240 / `knife` 1 012 / `pistol` 1 144 instancí), permisivní
licence CC BY 4.0 a přesně ty tři kategorie, které jsou cílem.

Třídy datasetu #2 (rhackathon), rovnoměrně zastoupené po ~500 instancích:
`ak` 572, `ax` 534, `cleaver` 483, `cutter` 505, `eto` 561, `knife` 478,
`long sword` 542, `m16` 526, `revolver` 493, `rifle` 530, `semi automatic` 496,
`short sword` 501, `shotgun` 506, `spear` 616.

### 2.1 Proč byl `weapon-detection-cabwp` odmítnut

Největší jediný dataset (15 981 obrázků) exportuje **jedinou třídu `weapon`**
se 16 966 boxy. Vizuální kontrola náhodného vzorku 48 boxů (skript
`inspect_class.py`, výstup v `artifacts/`) prokázala, že třída obsahuje
**směs krátkých palných zbraní, dlouhých palných zbraní a nožů**.

Namapování na jednu cílovou třídu by tedy nutně zavedlo šum: `weapon → gun`
by vložilo nože do třídy `gun`. Rozdělit třídu nelze bez kompletního
přeanotování. Dataset byl proto vyřazen celý.

Sekundární problém: export obsahuje agresivní augmentaci (sůl-a-pepř šum,
konverze do grayscale, rozmazání), po které je část výřezů na hranici
rozpoznatelnosti i pro člověka.

---

## 3. Zkoumané a odmítnuté datasety

### 3.1 Odmítnuté zdroje melee zbraní

Rešerše `weapon_detectors.md` (2026-06-26) doporučovala tři zdroje melee
zbraní. Ověření přes API prokázalo, že **žádný z nich není použitelný**:

| Dataset | Slíbené třídy | Skutečnost | Verdikt |
|---|---|---|---|
[`weapon-detection-ohetg/weapons-085lm`](https://universe.roboflow.com/weapon-detection-ohetg/weapons-085lm) | Bat, Club, Crowbar, Hammer, Axe, Machete | Názvy tříd jsou **zbytky README textu**: `'=============================='`, `'Roboflow is an end-to-end computer vision platform that helps you'`, `'This dataset was exported via roboflow.com on June 3, 2024...'` | ❌ labely rozbité |
| [`weapons-dataset/weapons-dataset-os1ki`](https://universe.roboflow.com/weapons-dataset/weapons-dataset-os1ki) | axe, bat, club, crowbar, katana, machette | `Knife` 904, `Gun` 1 049, `Pistol` 918, `Grenade` 1 080 — **žádné melee**. Navíc `versions=[]`, tj. bez exportované verze → nelze stáhnout | ❌ |
| [`weapon-detection-rfsqg/weapons-c6acf`](https://universe.roboflow.com/weapon-detection-rfsqg/weapons-c6acf) | (nalezeno při hledání melee) | `rifle` 7, `rifles` 24, `rifles.` 3 904 — tři varianty téže třídy včetně tečky, pouze pušky | ❌ |

**Závěr:** holá tyč, hůl nebo obušek jako samostatná anotovaná třída
v dostupných veřejných datasetech **neexistuje**. Nezávisle to potvrzuje
původní závěr rešerše. Melee je proto v této práci pokryto jen baseballovými
pálkami, sekerami, meči a kopími.

### 3.2 Odmítnuté zdroje zavazadel (`handbag`, `suitcase`, `backpack`)

Původní plán počítal se 6 třídami včetně zavazadel. Zkoumané zdroje:

| Dataset | Obrázků | Relevantní instance |
|---|---|---|
| [`playground-v6gip/person-and-luggage`](https://universe.roboflow.com/playground-v6gip/person-and-luggage) | 1 204 | `suitcase` 1 763, `handbag` 100, `backpack` 42 |
| [`suitcaseimages/suitcases-curf2`](https://universe.roboflow.com/suitcaseimages/suitcases-curf2) | 400 | `suitcase` 513, `handbag` 10 |
| [`suitcaseimages/suitcases-2`](https://universe.roboflow.com/suitcaseimages/suitcases-2) | 250 | `suitcase` 366 |
| [`pilot-test/baggage-detection`](https://universe.roboflow.com/pilot-test/baggage-detection) | 138 | `handbag` 85, `suitcase` 58, `backpack` 66 |
| [`datacluster-labs-agryi/suitcase-luggage-dataset-indoor-object-image`](https://universe.roboflow.com/datacluster-labs-agryi/suitcase-luggage-dataset-indoor-object-image) | 100 | `suitcase` 19, `handbag` 9 |
| [`jessel/bags-wugc6`](https://universe.roboflow.com/jessel/bags-wugc6) | 32 | 25 tříd typů kabelek, jednotky instancí |
| [`mahakworkspace/backpack-yn8it`](https://universe.roboflow.com/mahakworkspace/backpack-yn8it) | 819 | `backpack` |

Dohromady by šlo posbírat ~2 700 instancí `suitcase`, ale jen ~200 `handbag`.

**Rozhodnutí: zavazadla se netrénují.** Důvod není nedostatek dat u `suitcase`,
ale to, že **COCO je má lépe**. Předtrénovaný YOLO11x, který už je v pipeline
(`suspicious_detector.py`), detekuje `backpack` (třída 24), `handbag` (26)
a `suitcase` (28) z celého COCO datasetu (118 000 obrázků). Fine-tune z 2 700,
resp. 200 instancí by byl **horší, ne lepší**.

Vlastní model tak dělá výhradně to, co COCO neumí — zbraně a melee předměty.
Zavazadla zůstávají na COCO detektoru. Tím se z původních 6 tříd staly 3.

---

## 4. Mapování zdrojových tříd na cílové

Cílové třídy: **`0 = gun`, `1 = knife`, `2 = melee`**

| Cílová třída | Namapované zdrojové třídy |
|---|---|
| `gun` | `gun`, `guns`, `handgun`, `hand gun`, `pistol`, `pistols`, `revolver`, `rifle`, `rifles`, `rifles.`, `shotgun`, `ak`, `m16`, `semi automatic`, `heavy gun`, `heavyweapon`, **`heavy-weapon`**, `long guns`, `guns perspective`, `larga` |
| `knife` | `knife`, `knives`, `knife_weapon`, `knife_deploy`, `cutter`, `cleaver`, `karambit`, **`short sword`**, `dagger` |
| `melee` | `baseball-bat`, `baseball bat`, `baseball_bat`, `bat`, `bats`, `stick`, `club`, `crowbar`, `ax`, `axe`, `hammer`, `long sword`, `sword`, `katana`, `machete`, `machette`, `spear`, `0` |

### 4.1 Dvě mapování ověřená vizuálně

**`heavy-weapon` → `gun`.** Třída v `m7qso` s 11 598 boxy, název nejednoznačný
(mohlo jít i o granátomety či melee). Vzorek 32 boxů prokázal výhradně
**dlouhé palné zbraně** — pušky, AK, brokovnice, samopaly. Mapování na `gun`
je korektní.

**`short sword` → `knife`** (nikoli `melee`). Původně mapováno na `melee`.
Náhled sloučené třídy `melee` odhalil, že `short sword` obsahuje **dýky
vizuálně nerozeznatelné od velkých nožů**, které tak konkurovaly třídě `knife`.
Přesunutí 501 instancí do `knife` odstranilo konkurenci tříd a zároveň
posílilo nejslabší třídu (2 707 → 3 208 instancí). `long sword` (542 instancí)
zůstal v `melee`, protože meč nůž nepřipomíná.

**`0` → `melee`.** Nepojmenovaná jediná třída datasetu `dataset-pizme`.
Vzorek 32 boxů potvrdil, že jde o **baseballové pálky**.

### 4.2 Zahozené třídy

**Bezpečně zahozené** (`SAFE_DROP` — nejde o zbraně, obrázek zůstává v datasetu
jako platné pozadí): `person`, `people`, `hand`, `ball`, `blood`, `victim`,
`aggressor`, `violence`, `stabbing`, `baseball`, `helmet`, `glove`.

**Nejednoznačné** (`AMBIGUOUS` — zahodí se **celý obrázek**): `weapon`,
`weapons`, `eto`, `al`, `object`, `threat`.

Rozlišení mezi těmito dvěma kategoriemi je metodicky podstatné, viz §5.1.

---

## 5. Konstrukce sloučeného datasetu

Skript `build_dataset.py`. Tři problémy, které řeší:

### 5.1 Nebezpečné zahození labelu

Pokud se z obrázku odstraní label zbraňové třídy a obrázek zůstane v datasetu,
stane se z něj **pozadí** — a model se naučí, že zbraň je pozadí. To je horší
než obrázek nepoužít vůbec.

Proto se u každé nezmapované třídy rozhoduje:
- třída v `SAFE_DROP` → label zmizí, obrázek zůstává (např. `person` na fotce pistole)
- jinak → **zahodí se celý obrázek**

Díky tomuto pravidlu byl dataset `cabwp` vyřazen automaticky (jeho třída
`weapon` je v `AMBIGUOUS`): 14 250 obrázků zahozeno pro nejednoznačný label,
1 729 jako prázdné.

### 5.2 Deduplikace mezi datasety

Veřejné weapon datasety na Roboflow jsou **navzájem forky**, takže tentýž
obrázek se vyskytuje v několika z nich — a klidně v jednom v trénovacím a
v jiném ve validačním splitu. Bez deduplikace by validační mAP byla
**nadhodnocená** (data leakage).

Deduplikuje se podle **SHA-1 hashe obsahu souboru**. Při kolizi vyhrává
výskyt ve splitu s vyšší prioritou (`val` > `test` > `train`), aby se
validační split nekontaminoval ve prospěch trénovacího.

**Odstraněno 5 049 duplicit** z 46 173 kandidátů (10,9 %).

### 5.3 Polygonové anotace

Exporty `rhackathon` a `baseball-v1` míchají bounding boxy s **polygonovými
(segmentačními) anotacemi** ve formátu `class x1 y1 x2 y2 ...` se 4 až 87 body.
Nalezeno **549 takových řádků** (rhackathon 243, baseball-v1 306).

Naivní přečtení prvních čtyř čísel jako `cx cy w h` by dalo **úplně jiný box**.
Polygony se proto převádějí na **opsaný bounding box** (min/max souřadnic),
čímž se anotace zachová.

### 5.4 Výsledek

```
46 173 kandidátů
  → 41 124 unikátních      (−5 049 duplicit)
  → 24 340 obrázků         (−16 784 zahozeno pro nezmapovaný label)
```

| Split | Obrázků | Podíl |
|---|---|---|
| train | 19 924 | 81,9 % |
| val | 3 503 | 14,4 % |
| test | 913 | 3,7 % |

| Cílová třída | Instancí | Podíl |
|---|---|---|
| `gun` | 14 218 | 50,5 % |
| `melee` | 10 730 | 38,1 % |
| `knife` | 3 208 | 11,4 % |
| **celkem** | **28 156** | |

Splity se **dědí z původních datasetů** (train→train, valid→val, test→test),
nerozdělují se náhodně. Důvod: Roboflow aplikuje augmentaci pouze na trénovací
split, takže náhodné rozdělení by rozhodilo augmentované varianty téhož
obrázku mezi train a val — další forma leakage.

---

## 6. Známá omezení datasetu

### 6.1 Domain gap: web a sport vs. CCTV

**Nejzávažnější omezení.** Všechny použité datasety obsahují webové fotografie
a sportovní záběry: zbraně zblízka, dobré osvětlení, velké objekty v záběru,
pohled z úrovně očí.

Cílová doména (Motion Emotion Dataset) je opačná: **CCTV pohled shora, dav,
malé objekty, pohybové rozmazání, horší osvětlení.**

Rozlišení `imgsz=960` a agresivní augmentace (`scale=0.6`, `degrees=10`,
`hsv_*`) tento rozdíl zmírňují, ale neodstraňují.

### 6.2 Sportovní zaujatost třídy `melee`

Dva největší zdroje melee jsou baseballové fotografie:
**7 798 z 10 730 boxů (72,7 %)** je z `baseball-v1` (6 536) a `dataset-pizme`
(1 262). Nesportovních melee dat je jen NTUT (1 240 pálek) a rhackathon
(sekery, meče, kopí, ~1 700).

Riziko: model se může naučit kontext („pálka = baseballová scéna: helmy,
dresy, hřiště, plot") místo tvaru objektu. Vysoká hodnota `mAP50` pro `melee`
na validačním splitu tuto obavu **nevyvrací**, protože val má stejnou
distribuci jako train.

Navíc je export `baseball-v1` celý **konvertován do grayscale**. Pro noční
CCTV to nemusí být nevýhoda, ale je to systematická odchylka od trénovacích
podmínek ostatních tříd.

### 6.3 Nevyváženost třídy `knife`

`knife` má 3 208 instancí proti 14 218 u `gun`, tj. poměr **1 : 4,4**.
Projevilo se to ve výsledcích: recall 0,587 při precision 0,884
(viz `TRAINING_REPORT.md` §5).

### 6.4 Nereprezentativní testovací split

Testovací split (913 obrázků) obsahuje pouze **3 z 5 zdrojových datasetů** —
chybí `yolov7test` a `fypit2`, protože jejich exporty testovací split
neobsahují. Především **neobsahuje ani jeden nůž z `yolov7test`**, což je
nejzašuměnější zdroj.

Testovací split proto vychází **lépe** než validační (mAP50 0,855 vs. 0,775;
u `knife` 0,955 vs. 0,611). **Není to tvrdší benchmark, je užší a snazší.**
Jako hlavní metrika se proto uvádějí výsledky na validačním splitu.

Složení splitů:

| Zdrojový dataset | val | podíl | test | podíl |
|---|---|---|---|---|
| `yolov7test-u13vc` | 1 319 | 37,7 % | 0 | — |
| `baseball-v1` | 774 | 22,1 % | 326 | 35,7 % |
| `ntut-zy5y0` | 600 | 17,1 % | 300 | 32,9 % |
| `rhackathon` | 511 | 14,6 % | 287 | 31,4 % |
| `fypit2` | 299 | 8,5 % | 0 | — |

---

## 7. Reprodukce

```bash
# 1. Roboflow API klíč (registrace zdarma na roboflow.com, Account -> API)
mkdir -p ~/.config/roboflow && chmod 700 ~/.config/roboflow
printf '%s' 'VAS_KLIC' > ~/.config/roboflow/key && chmod 600 ~/.config/roboflow/key

# 2. Stažení datasetů (fixované verze, ~3.3 GB)
python download_datasets.py --out datasets/raw

# 3. Kontrola nejednoznačné třídy (volitelné, ověřovací krok)
python inspect_class.py --ds datasets/raw/weapondetection-mqgm5__weapon-detection-cabwp \
                        --cls weapon --n 48 --out weapon_sample.jpg

# 4. Sloučení, deduplikace, remapování tříd
python build_dataset.py --raw datasets/raw --out datasets/weapons_merged --link

# 5. Trénink
python train_weapon_yolo.py --data datasets/weapons_merged/data.yaml \
       --model yolo11l.pt --imgsz 960 --epochs 100 --batch 32 --workers 6 \
       --optimizer SGD --lr0 0.005
```

Skripty jsou v `docs/scripts/` (kopie) a v `weapon_training/` (aktuální verze).
