# Stav práce (mimo detekci zbraní)

> Sepsáno 2026-10-11 z repozitáře: soubory, výstupy notebooků, JSON s výsledky a git log. Nic se nespouštělo.
> Detekce zbraní (`weapon_training/`) je hotová a zdokumentovaná: trénink, MED test, nasazení v2, ARMED větev
> a koncept kapitoly `weapon_training/docs/KAPITOLA_DETEKCE_ZBRANI.md` (~6 800 slov). Tady se jí dál nevěnuji.

## Aktualizace 2026-10-28 (Windows PC, data z PC i z Macu)

Body tabulky „Co chybí do odevzdání“, které šly udělat bez rozhodnutí autora:

- **Bod 3, RWF BiLSTM, vyhodnoceno** (`eval_rwf_lstm.py`, výsledky `rwf_lstm_eval.json`). Data a model jsou na PC.
  - Checkpoint `best_lstm_pytorch.pt` (epocha 38) po oknech přesně reprodukuje 0,756. Je ale vybraný podle val, který je zároveň jedinou nezávislou sadou, takže je optimistický.
  - **Čistý odhad** (`--holdout`): stejný model a hyperparametry, checkpoint vybraný na 15 % trénovacích videí, oficiální val (400 videí) použit jen jednou jako test, 3 seedy.

    | úroveň | accuracy | macro F1 | F1 Fight | ROC AUC |
    |---|---:|---:|---:|---:|
    | okna (15 snímků) | 0,727 ± 0,003 | 0,721 | 0,679 | 0,789 |
    | **video (průměr oken)** | **0,781 ± 0,006** | 0,778 | 0,752 | 0,808 |
    | video, jen pokrytá (316/400) | 0,805 ± 0,008 | 0,805 | 0,812 | 0,869 |

    (Checkpoint vybraný na val: okna 0,756, video 0,802.)
  - **Pokrytí:** RWF videa mají při 5 FPS jen 25 snímků. V **84 ze 400** val videí (26 Fight) není nikdo sledovaný 15 snímků, takže nevznikne okno a video se počítá jako NonFight. Pro text: buď kratší okno, nebo pokrytí uvádět zvlášť.
  - Srovnání s literaturou (skeletové metody, video accuracy na RWF val). **Čísla ověřit v originálech**, protokoly se liší:
    - Human skeletons + change detection (ConvLSTM): 90,25 %, https://cs.paperswithcode.com/paper/human-skeletons-and-change-detection-for
    - LG-SPIL, 3D skeleton point clouds: 90,0 %; obecné modely pro rozpoznání akcí podle autorů nejvýš 87,3 %, https://arxiv.org/pdf/2308.13866
    - Structured Keypoint Pooling: 93,4 % (agregátor), https://next.hyper.ai/en/sota/tasks/activity-recognition/benchmark/activity-recognition-on-rwf-2000
    - Diplomová práce, póza + rozdíly snímků: 85,25 %, https://ebuah.uah.es/xmlui/handle/10017/60891
    - DIFEM (Random Forest nad interakcemi keypointů): 96,5 % jen ze sekundárního shrnutí, nepoužívat bez originálu.

    Náš výsledek 0,78 je tedy pod skeletovými SOTA (~0,85–0,93). Hlavní vysvětlení jsou okno 15 snímků při 5 FPS (16 % videí bez okna), modelování jednotlivých osob bez interakcí mezi nimi a malý model.
- **Bod 6, labeler, opraveno.** Oba labelery ukládají neolabelované framy jako −1 (`UNLABELED`). `fix_labeled_behaviors.py --write` přepočítal existující `.npz` ze segmentů v JSON (skelety beze změny). Keras notebook `detekce_novy_pristub.ipynb` okna s −1 vynechává (`to_categorical(-1)` by je tiše zařadil do třídy running).

  | adresář | dřív walking / susp. / running | teď walking / susp. / running / neolabelováno |
  |---|---|---|
  | `labeled_behaviors/` | 188 / 0 / 8 | 140 / 0 / 8 / 48 |
  | `labeled_behaviors_05/` | 209 / 4 / 2 | 126 / 4 / 2 / 83 |
- **Bod 4, tracker:** Hungarian a předpověď pohybu, body 4, 5, 7, 8, 9 a 12 z `CODE_REVIEW_PIPELINE.md` jsou opravené. CAVIAR se nezměnil. ARMED nasazení se nezměnilo (5 / 5 / 93 %). Bez COCO pálky drží bitku z 90 % místo 62 %.
- **Bod 5, CAVIAR held-out** (`caviar_loo.py`, `caviar_loo_results.json`). Leave-one-video-out přes 30 XML / 28 scénářů:
  - **precision 100 % (0 FP), recall 75 % (3/4)**, latence 6,0 s.
  - Chybí `LeftBox`: bez něj výběr kvůli nižší latenci zvolí radius 1,0 místo 1,5.
  - Na všech videích má F1 = 1 celkem 21 z 90 konfigurací. Číslo 4/4 je tedy na datech, na kterých se ladilo.
  - `LeftBag_PickedUp` je teď TP (+9,9 s). Definici pozitiva musí rozhodnout autor.
- **Bod 8, ABODA:** `diagnose_aboda.py` znovu pouští plnou pipeline s opraveným trackerem a u každého zavazadla zapisuje detekci, vlastníka a dobu bez vlastníka (`reference_img/aboda_alerts_v2/`). Původní batch běžel ještě s chybou trackeru z bodu 1. Video4: taška je detekovaná jen v ruce majitele (10–19 s). Když leží sama u skříně, YOLO11x ji nenajde a track zanikne. Selhává tedy **detekce** odloženého zavazadla, ne pravidla. Souhrn všech 11 videí je v `reference_img/aboda_alerts_v2/souhrn.json`.
- **Bod 15, NWPU:** `nwpu_eval_30*.json` vznikly také ještě se starým trackerem. NWPU nemá typ anomálie po videích, takže recall opuštění změřit nejde. Smysl má jen počet falešných poplachů na normálních videích (dřív 0 na 10 videích).
- **Úloha 1 ze zadání (pohyb davu) na oficiálních anotacích MED** (`med_features.py`, `med_person_features.py`,
  `med_train.py`, výsledky `med_results.json`). Dosud se MED anotace po snímcích (Panic / Fight / Congestion /
  Obstacle / Neutral, všech 31 videí, `dataset_frames_abnormal_labeling.m`) v projektu nepoužívaly; ruční
  labelování 196 framů tím ztrácí smysl. Model: na každém vzorkovaném snímku (5/s) prostorové příznaky
  ResNet-18 (512), optický tok Farneback (48) a detekce osob YOLO11x (24), nad nimi BiLSTM přes okno 16 snímků
  (3,2 s). Vyhodnocení 5 foldů podle videí (žádné video v tréninku i testu), pevný počet epoch, 3 seedy.

  | příznaky | časový model | přesnost | prům. přesnost po třídách | macro-F1 | normální vs. abnormální AUC |
  |---|---|---:|---:|---:|---:|
  | optický tok | ne | 0,498 | 0,434 | 0,401 | 0,672 |
  | optický tok | BiLSTM | 0,624 | 0,455 | 0,445 | 0,704 |
  | CNN | ne | 0,649 | 0,450 | 0,433 | 0,779 |
  | CNN | BiLSTM | 0,665 | 0,489 | 0,465 | 0,807 |
  | CNN + tok | ne | 0,689 | 0,512 | 0,510 | 0,795 |
  | CNN + tok | BiLSTM | 0,695 | 0,529 | 0,506 | 0,815 |
  | CNN + tok + osoby | ne | 0,687 | 0,512 | 0,509 | 0,798 |
  | **CNN + tok + osoby** | **BiLSTM** | **0,699** | **0,538** | **0,522 ± 0,025** | **0,825** |

  F1 po třídách u nejlepší varianty: Neutral 0,83, Panic 0,69, Fight 0,38, Congestion 0,20, Obstacle 0,51.
  Časová složka pomáhá ve všech variantách (+2 až +4 b. průměrné přesnosti po třídách). Delší okno 32 snímků
  zlepší Congestion (F1 0,20 → 0,38) za cenu ostatních tříd (`med_train_window.log`).
  **Srovnání s literaturou:** Rabiee et al. 2016 (článek k MED, dense trajectories + SVM, leave-one-sequence-out)
  uvádí průměrnou přesnost po třídách 0,387 (low-level), 0,436 (emotion-based), 0,439 (latent emotion);
  https://arxiv.org/abs/1607.07646. Náš protokol je přísnější (5 foldů místo 30/1). Slabé třídy: Fight je lokální
  událost, kterou globální příznaky scény nevidí (na ni cílí RWF skeletová větev), Congestion se plete s Neutral.
- **Úloha 2 ze zadání (abnormální pózy jednotlivců), bez anotací** (`med_skeletons.py` → `data/Motion_Emotion/skeletons/*.pkl`,
  `med_pose_anomaly.py`, výsledky `med_pose_results.json`). Skelety RT-DETR + ViTPose se sledováním pro všech 31 MED videí
  (5 snímků/s, 62 322 oken po 12 snímcích). GRU autoenkodér nad trajektorií jedné osoby (normalizovaný skelet + rychlost),
  učený jen na oknech z Neutral snímků trénovacích videí, skóre = chyba rekonstrukce, skóre snímku = průměr 3 nejvyšších.
  Stejných 5 foldů podle videí, AUC na úrovni snímků proti Neutral:

  | skóre | abnormální | Panic | Fight | Congestion | Obstacle |
  |---|---:|---:|---:|---:|---:|
  | baseline rychlost | 0,680 | **0,921** | 0,682 | 0,532 | 0,652 |
  | GRU autoenkodér | **0,701** | 0,883 | 0,678 | **0,641** | **0,679** |

  Střízlivý výsledek: autoenkodér přidává proti rychlosti jen málo celkem, ale jediný zachytí kongesci (rychlost = náhoda).
  Rvačku nevidí ani jeden (skelety v bitce jsou nekvalitní, tracky se trhají). Pro text: doplněk k modelu davu, ne samostatný detektor.
  Text je v `docs/KAPITOLA_CHOVANI.md` (Y.5) a `docs/typst/kapitola_chovani.typ`.
- **Nová data + přenos modelu davu na cizí scény** (`anomaly_bench.py`, `umn_labels.py`, výsledky
  `anomaly_bench_results.json`; zdroje a licence `data/ZDROJE.md`, citace `docs/typst/reference_data.yaml`).
  Staženo 2026-10-28: UCSD Ped1/Ped2, CUHK Avenue (+GT masky), UMN. Frame-level AUC (v závorce normalizace po videích):

  | dataset | optický tok | in-domain kNN bez anotací | model MED (přenos) |
  |---|---:|---:|---:|
  | Ped1 | 0,74 (0,69) | 0,69 (0,61) | 0,49 |
  | Ped2 | 0,81 (0,86) | **0,89** (0,80) | 0,66 |
  | Avenue | 0,76 (0,68) | 0,76 (**0,79**) | 0,51 |
  | UMN | 0,47 (0,57) | 0,86 (**0,94**) | 0,55 |

  **Model natrénovaný na MED se na jinou scénu nepřenáší** (úroveň náhody); stejné příznaky s modelem normality
  z normálních záběrů cílové kamery dávají 0,79–0,94. Do textu jako negativní, ale důležitý výsledek (kapitola Y.6).
  ShanghaiTech (skeletová anomálie, srovnání úlohy 2 s literaturou) čeká na ruční stažení z OneDrive.
- **Prostředí PC (Python 3.9):** opraveno `Path.hardlink_to` (od 3.10) v `build_*dataset.py` a `oversample_knife.py`. Vzorek původních dat v `build_finetune_dataset.py` byl na Windows jiný (řazení `WindowsPath` nerozlišuje velikost písmen), teď se řadí podle jména a dataset v3 vychází identicky s Macem. Skripty spouštět s `PYTHONUTF8=1` (JSONy z Macu jsou v UTF-8).

## Chronologie (git log, 41 commitů)

| Období | Co vzniklo |
|---|---|
| 08/2025 | První commit: framy, 1stage YOLO-pose skelety (`skeletons_yolo*`), labeler, **olabelovaná data** a MLP/LSTM v Kerasu |
| 09–12/2025 | ViTPose, 2stage detekce, preprocessing, SAHI (zahozeno), NMS, local redetect, RT-DETR vs. YOLO |
| 01–03/2026 | Úklid, README (CZ/EN) |
| 28. 6. 2026 | Velký commit 9bdb871: pre-incident pipeline (opuštěná zavazadla), CAVIAR/NWPU eval, RWF + PyTorch LSTM, demo videa |
| 08–10/2026 | Detektor zbraní (`weapon_training/`), revize kódu `weapon_training/docs/CODE_REVIEW_PIPELINE.md` |

Od srpna 2025 se olabelovaná data ani klasifikátory chování na MED nezměnily.

---

## 1. Extrakce framů

- **Implementace:** `extract_frames.py` (2 FPS → `frames_0.5/`), duplicitní funkce v `extract_skeletons_two_stage.ipynb` (buňka 8).
- **Data:** `data/Motion_Emotion/` má 32 položek (31 videí). `frames_0.5/` obsahuje 3 162 framů ze všech 31 videí. Starší `frames/` (1 FPS) má 1 535 framů.
- **Stav:** hotovo, funguje.
- **Slabiny:** žádné podstatné.

## 2. Super-resolution

- **Implementace:** `upscale_img_folder` v notebooku (buňka 9), `--upscale` v `abandoned_object_detector.py`, `upsacle_img.py`.
- **Skutečnost:** OpenCV na obou strojích nemá `cv2.dnn_superres`, takže běží bikubický fallback. README i README_CZ už to uvádějí (MAE 0,25–0,55), `plan_pre_incident.md` (Limity, bod 2) taky.
- **Pokrytí:** `frames_0.5_upscaled/` obsahuje jen videa 001 (120), 002 (**6 framů**) a 004 (82). Celkem 208 z 3 162 framů.
- **Rozpory:**
  - CLAUDE.md a README_CZ pořád uvádějí v datovém toku „ESPCN 4×“ a gamma γ=0,8. Kód (`suspicious_detector.py:219`) i README.md používají γ=1,3.
  - README.md (sekce 1) mluví o „rozlišení na 4K“.
  - `materiály.md` popisuje build OpenCV s CUDA pro Windows a podporu ESPCN, ta se ale nikdy neprojevila ve výstupech.
- **Pro text práce:** buď poctivě napsat „bikubické 4× zvětšení“, nebo ESPCN zprovoznit (opencv-contrib) a porovnat. Dnes nemáme žádné měření, jestli upscale detekci pomáhá.

## 3. Dvoustupňová detekce osob + ViTPose

- **Implementace:**
  - `suspicious_detector.py`, které je single source of truth: `RTDETRDetector.better_detection` (5 variant obrazu: original, shadow_norm, CLAHE, hist_eq, gamma) a `VitPoseExtractor`.
  - `extract_skeletons_two_stage.ipynb` jen importuje tenhle modul.
  - Custom komponenty MMPose v `models/vitpose/models/`.
  - Starší alternativy: `extract_skeletons.py` (1stage YOLO-pose), `vit_pose.py` (ViTPose bez detektoru), `test_rtdetr.py`.
- **Výsledky:**
  - RT-DETR vs. YOLO11x: README_CZ uvádí tabulku „13 vs. 9 osob, 0,340 s vs. 0,503 s, conf 0,743 vs. 0,670“.
  - `test_rtdetr.py:580` ukazuje, že jde o **jediný obrázek** (`001_0076.jpg`) **bez ground truth**. Precision ani recall se tedy nedají spočítat.
  - README.md 3.2 hodnotí detektory podle „vyššího confidence skóre“. Confidence skóre dvou různých modelů ale nejsou srovnatelná.
  - Kvalita skeletů se nikde neměří, ani vizuálně na vzorku.
- **Výstupy na MED:**
  - `skeletons_yolo_11_upscaled_2/` obsahuje jen **6 `.npy`** (001_0076, 002_0000–0004) s tvarem `(N, 14, 3)` (CrowdPose + skóre).
  - Ostatní adresáře se skelety (`skeletons_yolo/` 1 535, `skeletons_yolo_11/` 3 162, `skeletons_yolo_11_new/` 120) mají tvar `(N, 17, 2)`, jsou ze srpna 2025 a pocházejí z **1stage YOLO-pose**.
  - **Pipeline RT-DETR + ViTPose tedy na MED nikdy neproběhla v celém rozsahu.** Na RWF-2000 proběhla (viz 5b).
- **Rozpory:**
  - CLAUDE.md tvrdí, že stupeň 1 je YOLO11x a výstup `(num_people, 17, 2)`. README_CZ tvrdí, že stupeň 1 je RT-DETRv2-S R18 (48,1 AP). README.md tvrdí R101. Kód používá **R101** (`suspicious_detector.py:46`) a výstup je **14 bodů**.
  - CLAUDE.md odkazuje config `..._without_training.py`, kód používá `_v3.py`.
  - README_CZ zmiňuje „+44 % detekcí“ jako obecný závěr, přitom vychází z jednoho snímku.
  - Vizualizace kreslí spoje COCO-17 přes 14bodové skelety (CODE_REVIEW bod 7), takže obrázky skeletů v práci mohou být chybné.

## 4. Tracking a labeling chování (MED)

- **Implementace:** `label_skeletons.py` (OpenCV GUI, centroidový tracking, ruční korekce) a `label_skeletons_improved.py` (Tkinter/Matplotlib, README ho vede „v přípravě“, popis v `LABELER_IMPROVEMENTS.md`). Oba čtou 17bodové YOLO-pose skelety (`skeleton_base_dir = "skeletons_yolo_11"`).
- **Data (ověřeno načtením `.npz`):**

| Adresář | Zdroj skeletů | Sekvence | Framy | walking / suspicious / running |
|---|---|---|---|---|
| `labeled_behaviors/` | `skeletons_yolo` (1 FPS) | 4 (video 001: 1, 002: 3) | 196 | 188 / 0 / 8 |
| `labeled_behaviors_05/` | `skeletons_yolo_11` (2 FPS) | 2 (001, 002) | 215 | 209 / 4 / 2 |

- **Údaj z CLAUDE.md (4 sekvence, 196 framů, ~96 % walking) pořád platí.** Celkem jsou olabelovaná jen 2 videa z 31.
- **Chyba v exportu** (`label_skeletons.py:657–662`): každý frame mimo označený segment dostane `behavior = 0`. Neolabelované framy se tak počítají jako „walking“. Příklady:
  - `002_person_1` má segment jen 0–10, přesto všech 46 framů nese 0.
  - `labeled_behaviors_05/002_person_0` začíná segmentem od framu 27.

  Podíl walking je tím uměle nafouknutý. Správně by neolabelované framy měly mít −1 a maskovat se.
- **Další slabiny:** labely vznikly nad 1stage skelety, ne nad finální pipeline (14 bodů ViTPose). Mezi formáty se musí převádět (`train_lstm_pytorch.ipynb`, buňka 2: COCO-17 → CrowdPose-14).

## 5. Klasifikace chování

### 5a. MED: MLP a Keras LSTM

- **MLP** (`clean_detekce_osob.ipynb`, model `models/vlastni_tf/frame_classifier_mlp.keras`):
  - Výstup buňky 7 uvádí test accuracy **67,48 %**.
  - Buňka 9 (classification report) ukazuje recall 1,00 pro třídu 0 a **0,00 pro třídy 1 a 2**. Podpora tříd je 496/158/81, takže 496/735 = 0,6748.
  - **Model tedy vždy predikuje „walking“ a 67 % je jen majoritní baseline.**
  - Labely nejsou z labeleru, ale z čísla framu ve videu 001 (`get_label_from_filename1`: 0–37 walking, 38–49 suspicious, 50–57 running). Všechny osoby ve framu tak dostanou stejný label.
- **LSTM Keras** (`detekce_novy_pristub.ipynb`, model `models/vlastni_tf/best_lstm_behavior_model.h5`):
  - Ze 4 sekvencí vzniklo **9 oken** (30×34), rozdělení 5/2/2.
  - Výstupy uvádějí test accuracy **1,0000**, ale classification report (buňka 11) má support 60 framů a **všechny jsou walking**. Výsledek je bezcenný.
- **Rozpory:** README a CLAUDE.md prezentují „MLP 67 % accuracy“ jako výsledek. Soubory `.h5` a `.keras` nejsou v kořeni (jak píše README), ale v `models/vlastni_tf/`.

### 5b. RWF-2000: PyTorch BiLSTM + attention (fight / non-fight)

- **Implementace:**
  - `process_rwf.ipynb` a `run_rwf_batch.py`: RT-DETR + ViTPose → pickly s tracky po 5 FPS.
  - `train_lstm_pytorch.ipynb` a `run_train_lstm.py`: BiLSTM + MHA, normalizace, velocity, focal loss, augmentace, okno 15 framů, 2 třídy.
- **Výsledky:**
  - `training_history.json` obsahuje 49 epoch, nejlepší **val_acc 0,756**, finální train_acc 0,805, val_acc 0,752 a val_loss 0,649 vs. train_loss 0,424.
  - Je to jediný netriviální výsledek klasifikace chování.
  - Chybí confusion matrix, F1, video-level accuracy a srovnání s literaturou (skeletové metody na RWF-2000 typicky hlásí kolem 80–90 %; je potřeba ověřit a ocitovat).
- **Chybějící artefakty na tomto stroji:**
  - `data/RWF-2000/` (pickly) ani `best_lstm_pytorch.pt` tu nejsou, zřejmě zůstaly na Windows PC.
  - Notebooky mají buňky tréninku a vyhodnocení **zakomentované**. `process_rwf.ipynb`, buňka 10, ukazuje „0 / 2000 pickles“, takže výsledek pochází z `run_*.py` a log tréninku v repu chybí.
- **Dokumentace:** jen hlavička notebooku a odstavec v `plan_pre_incident.md`. README o RWF a o PyTorch LSTM **nic neříká**.
- **Integrace:** LSTM se v `abandoned_object_detector.py` ani v `suspicious_detector.py` nepoužívá. „Combined alert“ z plánu není implementovaný.

## 6. Detekce opuštěných zavazadel (pre-incident)

- **Implementace:**
  - `abandoned_object_detector.py`: PersonTracker, ObjectTracker (IoU), vlastnictví podle vzdálenosti, pravidlo opuštění, dvouprůchodový render a `--armed`.
  - `suspicious_detector.py::better_object_detection` (YOLO11x COCO: batoh, kabelka, kufr).
  - Evaluace: `eval_caviar_rules.py`, `tune_caviar_params.py`, `run_aboda_batch.py`, `test_batch.py` a `regen_demos.py`.
- **Dokumentace:** `plan_pre_incident.md` (plán + „Výsledky implementace“) a `datasety.md`. **V README.md tahle část chybí úplně.**

### CAVIAR (pravidla nad GT boxy)

- `plan_pre_incident.md` uvádí 18 scénářů, P = 100 %, R = 75 % (3/4) a latenci 8,7 s při `abandon_sec=5, radius=2.0`.
- `caviar_tuning_results.json` obsahuje grid 90 konfigurací nad **20 scénáři** (4 pozitivní a 16 negativních). F1 = 1,0 má 21 konfigurací, třeba `abandon=2, radius=1.5` s latencí 6,39 s. Konfigurace `abandon=5, radius=2.0` má F1 0,857 (shoduje se s plánem).
- **Problémy:**
  1. Ladí se a testuje na **stejných 20 videích** se 4 pozitivy. F1 = 1,0 je tím optimistické, chybí held-out sada.
  2. Defaulty v kódu (`abandon_sec=2`, `min_own=1`) byly sníženy, aby prošel `LeftBag_PickedUp`. V plánu je ale tentýž případ označen jako *sémanticky správný FN* (taška zvednuta). To je metodický rozpor, který je potřeba v práci vysvětlit, případně scénář z pozitiv vyřadit.
  3. **Chyba trackeru** (CODE_REVIEW bod 1, ověřená testem): track po jednom vynechaném framu už nikdy nespáruje. Eval CAVIAR používá stejné trackery (`eval_caviar_rules.py:134`), takže **parametry jsou vyladěné na chybném chování** a po opravě je nutné přeladit. *Aktualizace: oprava je mergnutá; přeladění ani nebylo potřeba, CAVIAR dává po opravě stejný výsledek (4 TP / 23 TN, 6,4 s), protože GT boxy CAVIAR nemají výpadky.*
  4. Rozpor v počtu scénářů (18 vs. 20).
- **Plná pipeline na CAVIAR:** `demo_outputs/test/LeftBag_alerts.json` a `LeftBox_alerts.json` jsou prázdné (0 alertů). Plán (Limity 1) to vysvětluje tím, že YOLO na CAVIAR v rozlišení 384×288 batohy nevidí.

### ABODA (plná pipeline)

- `reference_img/aboda_alerts/`: video1 má 2 alerty (vizuálně ověřené jako správné), **video2–video11 mají 0 alertů**.
- Plán uvádí „batch v běhu“ a „ABODA nemá public GT“. V repu ale výsledek batche (10/11 videí bez alertu) **nikde komentovaný není**.
- Prakticky jde o recall 1/11 na úrovni videí, za předpokladu, že každé ABODA video obsahuje opuštěný předmět; je potřeba to ověřit.

### NWPU Campus (`eval_nwpu.py`)

- `nwpu_eval_30.json` i `nwpu_eval_30_tuned.json` pokrývají 30 videí (20 anomálních a 10 normálních) a 39 530 framů. **Alerty: 0**, recall 0, video_tp 0/20.
- Commit 9bdb871 to prezentuje jako „0 % FP out-of-domain“. Ve skutečnosti detektor na HD datech nevyhlásil nic. NWPU navíc má jen binární GT, které nerozliší opuštění od jiných anomálií (hlavička `eval_nwpu.py`), takže eval je jako měřítko slabý.
- Data NWPU, CAVIAR ani ABODA nejsou na tomto Macu (`data/` obsahuje jen `Motion_Emotion` a `coco`).

### Loitering a PETS

V plánu jsou, ale **neexistují** (`grep loiter` nenajde nic). PETS 2006/2007 ani AVSS se nestahovaly.

## 7. Ostatní

- **Demo videa** (`demo_outputs/`, 3 + 7 testovacích MP4 s alert JSONy): ukázky pro obhajobu. Alert je jen v `test/020` (1) a `test/video1` (3).
- **Mrtvé nebo zastaralé soubory:**
  - `RTDETR_TRAINING_GUIDE.md` a `download_datasets_manual.md`: plán vlastního RT-DETR na 7 třídách, nahrazený YOLO11l ve `weapon_training/`.
  - `yolo_multiclass_detector.py`, `test_weapon_detection.py` (COCO + Roboflow v8), `test_pytorch_mac_gpu.py`.
  - `yolov8x-pose.pt` a `yolo11x-pose.pt` v kořeni.
- **Závislosti:** `requirements.txt` existuje (35 řádků, 2026-10). CLAUDE.md („No requirements file“) je zastaralý.
- **Text práce:** v repu je jen kapitola o zbraních. Ostatní kapitoly (úvod, rešerše, dataset, detekce a póza, chování, pre-incident, závěr) v repozitáři nejsou. Možná je autor má jinde, je potřeba ověřit.

---

## Souhrn rozporů dokumentace × skutečnost

| Tvrzení | Kde | Skutečnost |
|---|---|---|
| Stupeň 1 = YOLO11x, výstup (N,17,2) | CLAUDE.md | RT-DETR R101, výstup (N,14,3) |
| RT-DETR R18 je produkční | README_CZ | R101 (README.md, kód) |
| RT-DETR +44 % detekcí, +32 % rychlejší | README_CZ | 1 obrázek, bez GT |
| ESPCN 4×, γ=0,8 | CLAUDE.md, README_CZ (tok dat) | bikubika, γ=1,3 |
| Skelety ViTPose v `skeletons_yolo_11_upscaled_2/` | CLAUDE.md, README | jen 6 framů; labely a klasifikátory jsou na 1stage YOLO-pose |
| MLP 67 % accuracy | README, CLAUDE.md | majoritní baseline, recall tříd 1 a 2 = 0 |
| LSTM test accuracy 100 % | notebook | test obsahuje jen třídu 0 |
| CAVIAR F1 = 1,0 | commit 9bdb871, JSON | ladění = test, 4 pozitiva, chybný tracker |
| NWPU „0 % FP“ | commit 9bdb871 | 0 alertů celkem, recall 0 |
| ABODA „batch v běhu“ | plan_pre_incident.md | doběhl, 10/11 videí bez alertu |
| Modely `.h5`/`.keras` v kořeni | README | `models/vlastni_tf/` |
| Žádný requirements | CLAUDE.md | `requirements.txt` existuje |

---

## Co chybí do odevzdání

Seřazeno podle důležitosti. „AI“ znamená, že úkol zvládne asistent sám bez GPU a bez rozhodnutí autora.

| # | Položka | Odhad | Kdo |
|---|---|---|---|
| 1 | **Rozhodnout rámec práce.** Zadání zní „prostorově-temporální NN“, ale klasifikace chování na MED je nefunkční (196 framů, 0 suspicious). Varianty: (a) jádrem bude RWF LSTM + pre-incident pravidla a MED zůstane jako ukázka, (b) dolabelovat MED. | rozhodnutí | **autor** (+ vedoucí) |
| 2 | **Text práce** mimo kapitolu o zbraních: úvod, rešerše, data, detekce/póza, chování, pre-incident, závěr | 2–4 týdny | AI navrhne koncepty z repa, **autor** finalizuje |
| 3 | **RWF LSTM: dohledat artefakty a vyhodnotit.** Najít pickly a `best_lstm_pytorch.pt` na PC, spočítat confusion matrix, F1, video-level accuracy a srovnat s literaturou | 0,5–1 den | **autor** (přenos dat); vyhodnocení pak AI na CPU |
| 4 | **Opravit tracker** (CODE_REVIEW bod 1, příp. Hungarian, body 4–6) a znovu spustit `tune_caviar_params.py` | 0,5 dne | **AI** (CAVIAR eval je rule-only, běží na CPU v sekundách; potřebuje `data/CAVIAR`) |
| 5 | **Poctivá CAVIAR metodika:** held-out (leave-one-out nebo ladit na části a testovat na zbytku), rozhodnout o `LeftBag_PickedUp`, reportovat i zvolenou konfiguraci | 0,5 dne | **AI** (definici pozitiva rozhodne autor) |
| 6 | **Opravit export labeleru** (neolabelované framy = −1 + maska) a přepočítat statistiky tříd | 2 h | **AI** |
| 7 | **Kvantitativní eval detekce osob:** RT-DETR vs. YOLO na ≥ 30–50 MED framech s ručními GT boxy (P/R/AP, čas) místo jednoho snímku | 1–2 dny | **ruční anotace + GPU** (asistent připraví nástroj a skript metrik) |
| 8 | **ABODA:** zjistit, jestli existuje GT, popsat výsledek 1/11 a analyzovat proč (detekce objektů vs. pravidla). Vybrat obrázky selhání | 0,5–1 den | AI rozbor JSONů; nové běhy **GPU** |
| 9 | **Pipeline ViTPose na MED** (aspoň olabelovaná videa), aby labely a klasifikátor seděly s finální pipeline | 1 den výpočtu | **GPU** |
| 10 | **Dolabelovat MED** (pokud zůstane MED klasifikace): více videí, třídy suspicious/running | dny | **ruční anotace (autor)** |
| 11 | **Sjednotit dokumentaci:** README.md doplnit o sekce chování, RWF a pre-incident, opravit CLAUDE.md a README_CZ podle tabulky rozporů, archivovat zastaralé .md | 0,5 dne | **AI** |
| 12 | **Vizualizace skeletů** s 14bodovými spoji CrowdPose (CODE_REVIEW bod 7), přegenerovat obrázky do práce | 2 h + běh | AI kód, regenerace **GPU** nebo MPS |
| 13 | Super-resolution: zprovoznit ESPCN (opencv-contrib) a změřit přínos, nebo v textu jen napsat „bikubika“ | 0,5 dne / 10 min | AI (text); měření GPU |
| 14 | Loitering, PETS, AVSS: buď implementovat, nebo v práci uvést jako budoucí práci | 1–2 dny / 0 | rozhodnutí **autor** |
| 15 | NWPU: buď vyřadit, nebo ho přeformulovat jako test falešných poplachů na HD datech s poctivým výkladem 0 alertů | 1 h | **AI** (text) |
| 16 | Úklid repa (mrtvé skripty, váhy v kořeni, duplicitní adresáře se skelety) | 2 h | AI navrhne seznam, **autor** schválí |

## Doporučené pořadí dalších kroků

1. **Autor rozhodne rámec** (bod 1) a dohledá RWF artefakty na PC (bod 3). Na tom závisí, co se bude psát.
2. **Rychlé opravy bez GPU (AI):** export labeleru (6), tracker (4), přeladění a held-out CAVIAR (5), rozbor ABODA a NWPU JSONů (8, 15).
3. **Vyhodnocení RWF LSTM** na CPU, jakmile budou data (3).
4. **Sjednocení dokumentace** podle nových čísel (11), včetně opravy vizualizace (12).
5. **GPU a anotace:** GT boxy pro eval detekce (7), ViTPose na MED (9), případně dolabelování (10). Patří na DGX Spark nebo Windows PC.
6. **Psaní kapitol** (2) průběžně od bodu 3. Zbytek (13, 14, 16) dělat podle času.
