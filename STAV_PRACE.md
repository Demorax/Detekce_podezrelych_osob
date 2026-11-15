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
- **Bod 1: úloha 2 (anomální pózy) na ShanghaiTech standardním protokolem** (2026-10-29,
  `shanghai_skeletons.py`, `shanghai_pose_anomaly.py`, výsledky `shanghai_pose_results.json`). Stejný GRU autoenkodér
  jako na MED, trénink jen na 330 trénovacích (normálních) videích (129 040 oken po 12 snímcích), test 107 sekvencí,
  40 791 snímků (42 % abnormálních), frame-level micro AUC:

  | metoda | AUC |
  |---|---:|
  | baseline: rychlost 3 nejrychlejších osob | 0,712 |
  | **GRU autoenkodér (tato práce)** | **0,743 ± 0,001** |
  | MPED-RNN (Morais et al. 2019) | 0,734 |
  | GEPC (Markovitz et al. 2020) | 0,761 |
  | STG-NF (Hirschorn & Avidan 2023) | 0,859 |

  Model je srovnatelný s metodou, ze které vychází (o 0,9 bodu lepší), bez ladění na ShanghaiTech (okno 12 a
  architektura převzaté z MED). Literární hodnoty jsou z původních článků, ověřit při citaci.

  **Ablace rozšíření podle MPED-RNN** (2026-10-30, `shanghai_pose_anomaly.py --variants`,
  `shanghai_pose_variants.json`, log `shanghai_variants.log`). Varianty pevně dané předem (ShanghaiTech nemá
  validační sadu s anomáliemi, hlásí se všechny), 3 seedy:

  | varianta | AUC |
  |---|---:|
  | základ (lokální skelet + rychlost) | **0,743 ± 0,001** |
  | + globální větev: poloha a velikost boxu ve snímku | 0,741 ± 0,000 |
  | + předpověď 6 dalších snímků (skóre = rekonstrukce + předpověď) | 0,717 ± 0,001 |
  | + podmínění scénou (13 kamer) | 0,716 ± 0,000 |

  Negativní výsledek: poloha ve snímku nic nepřidá a předpověď budoucnosti skóre zhorší o 2,5 bodu
  (předpověď pohybu osoby je nejistá i u normálního chování, takže její chyba přidává do skóre šum).
  Hlavní konfigurace zůstává základní model; nic se na testu dál neladilo.

  **Normalizační tok podle STG-NF** (2026-11-01, `shanghai_stgnf.py`, `shanghai_stgnf_results.json`, log
  `shanghai_stgnf.log`). Model i hodnocení převzaté z oficiálního kódu STG-NF (výchozí argumenty pro ShanghaiTech:
  K=8 kroků toku se ST-GCN couplingem, prior N(3, 1), Adamax 5e-4, 8 epoch, zrcadlení; normalizace okna jako
  `normalize_pose`, takže okno nese pózu i pohyb osoby). Stejná okna jako u autoenkodéru (12 vzorků po 5 snímcích/s).
  Hodnocení snímků „STG-NF“ = maximum skóre přes osoby, snímky bez osoby nejnormálnější, gaussovské vyhlazení
  sigma 1..6 (`smooth_scores`); „naše“ = průměr 3 nejvyšších, medián 5. Všechny kombinace pevně předem, 3 seedy:

  | model | naše hodnocení | hodnocení STG-NF |
  |---|---:|---:|
  | GRU autoenkodér | 0,743 ± 0,001 | 0,760 ± 0,001 |
  | tok (STG-NF) | 0,737 ± 0,002 | 0,787 ± 0,002 |
  | autoenkodér + tok (součet z-skóre z trénovacích oken) | 0,744 ± 0,002 | **0,789 ± 0,003** |

  Zlepšení 0,743 → 0,787 (tok) / 0,789 (fúze), tedy nad MPED-RNN (0,734) i GEPC (0,761). Zisk dělají obě části:
  hodnocení snímků +1,7 bodu u autoenkodéru, tok dalších +2,7 bodu, a teprve spolu (tok potřebuje maximum
  přes osoby, průměr 3 nejvyšších mu nesedí). Fúze přidá jen 0,2 bodu, takže hlavní model = tok (jednodušší,
  368 parametrů). Zbývající rozdíl proti STG-NF (0,859) je hlavně v datech: oni mají AlphaPose skelety pro
  všech 24 snímků/s a okno 24 snímků, my ViTPose po 5 snímcích/s (stejná pipeline jako zbytek práce).

  **Skelety vs. snímková frekvence** (2026-11-01, `shanghai_stgnf_refposes.py`, `shanghai_stgnf_refposes_results.json`,
  log `shanghai_stgnf_refposes.log`). Stejný tok nad referenčními skelety autorů STG-NF (AlphaPose, COCO 17 + krk,
  `data/STG-NF/`, zdroj v `data/ZDROJE.md`), varianty předem, 3 seedy, hodnocení STG-NF:

  | skelety | vzorkování, okno | AUC |
  |---|---|---:|
  | naše (ViTPose-H, CrowdPose 14) | 5 snímků/s, 12 vzorků | 0,787 ± 0,002 |
  | referenční STG-NF (AlphaPose) | 5 snímků/s, 12 vzorků | 0,788 ± 0,013 |
  | naše (ViTPose-H, CrowdPose 14) | 12 snímků/s, 12 vzorků (1 s) | 0,826 ± 0,004 |
  | referenční STG-NF (AlphaPose) | 12 snímků/s, 12 vzorků (1 s) | 0,844 ± 0,005 |
  | referenční STG-NF (AlphaPose) | 24 snímků/s, 24 snímků (výchozí STG-NF) | **0,844 ± 0,001** |
  | STG-NF, článek | 24 snímků/s | 0,859 |

  Závěry: (1) reimplementace je věrná (0,844 proti 0,859 v článku; zbytek mohou být detaily jako počet epoch
  nebo seed), (2) naše skelety jsou při stejném vzorkování stejně dobré jako referenční (0,787 vs. 0,788, naše
  s menším rozptylem), (3) celý zbývající rozdíl dělá snímková frekvence (+5,6 bodu z 5 na 24 snímků/s),
  přitom 12 snímků/s stačí (0,844, varianta C, `shanghai_stgnf_refposes_c.json`) -> naše skelety znovu po 2 snímcích.

  **Naše skelety po 2 snímcích (12 snímků/s)** (2026-11-15, `shanghai_skeletons.py --step 2` ->
  `data/ShanghaiTech/skeletons_step2/`, `shanghai_stgnf.py --skeletons data/ShanghaiTech/skeletons_step2
  --train-stride 3`, `shanghai_stgnf_step2_results.json`, log `shanghai_stgnf_step2.log`; 131 317 trénovacích
  oken, 40 791 testovacích snímků, 3 seedy):

  | model | naše hodnocení | hodnocení STG-NF |
  |---|---:|---:|
  | GRU autoenkodér | 0,737 ± 0,001 | 0,751 ± 0,002 |
  | tok (STG-NF) | 0,785 ± 0,005 | **0,826 ± 0,004** |
  | autoenkodér + tok | 0,784 ± 0,004 | 0,821 ± 0,004 |

  Tok si polepší 0,787 -> 0,826 (+3,9 bodu), autoenkodér z hustšího vzorkování nic nemá (0,760 -> 0,751) a
  fúze už nepomáhá -> hlavní model úlohy 2 = tok na skeletech po 2 snímcích, **0,826**. Proti referenčním
  skeletům při stejném vzorkování (0,844) zůstává 1,8 bodu: při 5 snímcích/s byly skelety rovnocenné, při
  hustším vzorkování se projeví rozdíl (pravděpodobně plynulejší sledování a menší šum bodů AlphaPose mezi
  sousedními snímky; neověřeno). Extrakce zpracuje 2,5x víc snímků než po 5.
  Referenční skelety slouží jen k tomuto rozboru, hlavní výsledek práce zůstává na vlastní pipeline.
- **Bod 4: spojená pipeline / demo** (2026-10-29, `demo_pipeline.py`, modely `models/behavior/` z
  `train_demo_models.py`; demo videa 5, 10, 15 nebyla v tréninku modelů davu ani póz). Jeden průchod videem:
  osoby + skelety, YOLO objekty + zbraně v2 + statické předměty, tracker, ARMED, opuštění, chování davu
  (Transformer), rvačka ze skeletů (RWF model, okno 2 s), anomální póza (autoenkodér, práh p99).
  Poplachy davu a rvačky s časovým potvrzením (1,6 s, resp. 0,6 s) a 10 s odstupem stejné třídy.
  Výstup `demo_outputs/full/<video>_demo.mp4` (lokálně, ~125 MB) a `<video>_events.json` (v gitu).

  | video | anotace MED | poplachy dema |
  |---|---|---|
  | 005 | neutrální 0–27 s, panika (bitka s tyčemi) 27–32 s | rvačka (skelety) 27,8 s, ARMED 4 osoby 28,4–29,2 s, panika 30,0 s; žádný falešný |
  | 015 | neutrální 0–12 s, rvačka 12–45 s | dav: rvačka 17,0 s (+5 s), 32,8 s, 43,4 s; kongesce 35,6 s (chybně); falešné opuštěné zavazadlo 10,8 s |
  | 010 | neutrální 0–43, kongesce 43–47, překážka 47–70, panika 70–76 s | překážka 8,2 s (chybně), rvačka (skelety) 75,7 s při útěku, panika 77,1 s; kongesce a překážka nezachyceny |

  Demo běželo v rychlém režimu na sdíleném GPU: 0,95–1,3 s na vzorkovaný snímek (čisté měření rychlosti viz bod 6).

  **Demo v2 s obrazovými modely** (2026-10-31, `demo_outputs/full_v2/`, logy `demo_v2_*.log`): dav = MViT-v2-S
  doučený na MED bez demo videí (`med_video.py --final`, `models/behavior/crowd_video.pt`), rvačka navíc obrazový
  MViT-v2-S z RWF-2000 (`fight_video.pt`) nad posledními 5 s, 1× za sekundu, poplach po 2 potvrzeních.
  Ostatní moduly beze změny (skeletová skóre jsou s v1 identická).

  | video | v1 (ruční příznaky davu) | v2 (obrazové modely) |
  |---|---|---|
  | 005 | rvačka (skelety) 27,8, ARMED 28,4–29,2, panika 30,0 | rvačka (skelety) 27,8, ARMED 28,4–29,2, **rvačka (obraz) 29,0**, panika **29,6** |
  | 015 | dav: rvačka **17,0**, 32,8, 43,4; kongesce 35,6 (chybně); zavazadlo 10,8 (chybně) | rvačka (obraz) 27,0, 37,0; dav: rvačka 27,2; **žádná chybná třída davu**; zavazadlo 10,8 (chybně) |
  | 010 | překážka 8,2 (chybně), panika 77,1; překážka nezachycena | **překážka 57,9 (správně)**, **panika 75,5**; rvačka (obraz) 27,0, 49,0 (chybně), 72,1 (při panice) |

  Model davu v2 nemá žádnou chybnou třídu a zachytí překážku, kterou v1 neviděl; rvačku v 015 hlásí o 10 s později
  než v1 (do 25 s ji model vidí jako Neutral). Obrazový model rvaček z RWF se na MED (shora snímaný dav, malé postavy)
  přenáší jen částečně: rvačky v 005 a 015 zachytí, ale v 010 dává 2 falešné poplachy. Pro MED scény je proto
  spolehlivější třída Fight modelu davu; RWF model je určený pro bližší kamery. Prahy ani potvrzení se podle dema
  neladily. Rychlost v2 (rychlý režim, GPU sdílené jen s demem): 351–459 ms na vzorkovaný snímek.

  **Falešné rvačky obrazového modelu RWF v MED – řešení** (2026-11-15, `med_fight_transfer.py`,
  `med_fight_transfer_results.json`, log `med_fight_transfer.log`). Model `fight_video.pt` puštěný přesně jako
  v demu (5 s, 1× za s, potvrzení 2×, odstup 10 s) na 28 MED videích mimo demo (model je nikdy neviděl):
  AUC rvačka vs. ostatní jen 0,820 (na RWF 0,989) a je přesebevědomý – 99. percentil P na normálních oknech je
  0,997. Pravidla pevně předem:

  | pravidlo | falešné poplachy (28 videí) | zachycené rvačky | demo 005 / 010 / 015 |
  |---|---:|---:|---|
  | R0 obraz RWF, práh 0,5 (v2) | 20 (67,5/h) | 7/7 | 1 / 3 falešné / 2 správné |
  | R1 obraz RWF, práh p99 normálních oken (0,997) | 3 (10,1/h) | 3/7 | 0 / 1 falešný / rvačka nezachycena |
  | **R2 obraz RWF + souhlas modelu davu P(Fight) > 0,5** | **1 (3,4/h)** | **6/7** | **0 / 0 / 2 správné (27,0 s)** |
  | R3 jen model davu, P(Fight) > 0,5 | 4 (13,5/h) | 6/7 | 0 / 0 / 1 správný (26,0 s) |

  Hradlo modelu davu se na kalibračních videích bere z křížové validace (`med_video_results.json`, seed 0,
  kauzální posun středu okna o 7 vzorků), v demu skutečné hodnoty (crowd_video.pt demo videa neviděl).
  Samotná kalibrace prahu nestačí (ztratí 4 ze 7 rvaček i rvačku v 015), souhlas obou modelů ubere 95 %
  falešných poplachů proti R0 a 75 % proti samotnému modelu davu za cenu jedné rvačky. Zavedeno do
  `demo_pipeline.py` (`FIGHT_GATE`, vypnutí `--no-fight-gate` pro kamery typu RWF bez davu). Pozn.: v 005 je podle
  GT panika, ne rvačka, takže i poplach v 29,0 s byl falešný. Výhrada: R2 vybráno ze 4 pravidel na stejných
  28 videích (demo je nezávislé, ale jen 3 videa a 1 rvačka).
  **Demo v3** (`demo_outputs/full_v3/`, logy `demo_v3_*.log`) s hradlem potvrzuje přehrání: 005 a 010 bez
  poplachu „rvačka (obraz)“, 015 rvačka (obraz) 27,0 a 37,0 s, dav: rvačka 27,2 s; ostatní poplachy shodné s v2.
  Zbývá: skeletový model rvaček hlásí rvačku při panice (005 27,8 s, 010 75,7 s) – útěk vypadá v pohybu skeletů
  jako rvačka.
- **Bod 3: detektor nově odložených nehybných předmětů** (2026-10-29, `static_object_detector.py`).
  Duální model pozadí (Porikli 2008) nad gradienty; počítá se jen nárůst hran proti dlouhodobému pozadí, osoby
  a přepálené oblasti se maskují. Vývoj podle pozorovaných chyb na CAVIAR (ne ladění na čísla):
  jasová varianta hlásila světlou skvrnu od okna ve všech scénářích, varianta s absolutní změnou gradientu
  hlásila „duchy“ na místech, odkud odešly osoby.
  - **CAVIAR** (`eval_caviar_static.py`, osoby z GT, objekty z videa, 20 scénářů s videem): 0 falešných alertů
    v 16 scénářích bez opuštění (jasová varianta 15); LeftBox a LeftBag_PickedUp včas (7,4 a 9,9 s), LeftBag
    detekován, ale alert 19 s po události (0,6 s za tolerancí), LeftBag_AtChair taška 11 × 9 px nezachycena.
    Horní mez s GT objekty je 4/4.
  - **ABODA** (`run_aboda_static.py`, osoby YOLO11x jeden průchod, snímky alertů v `reference_img/aboda_static/`,
    každý ověřen okem): **opuštěný předmět zachycen v 6 z 11 videí** (1, 3, 5, 6, 9, 10; dříve s COCO
    zavazadly 1/11). Nezachyceno 2, 4, 7, 8; video 11 (dav) neověřitelné. Falešné alerty hlavně ve videích 7 a 8
    (prudká změna osvětlení, noc/IR: 15 a 9), dále 1 ve videu 5 a 2 ve videu 6. Video 1 má teď jeden alert místo
    čtyř (slučování tříd zavazadel + jiný zdroj objektů).
  - Závěr do textu: detektor bez třídy objektu řeší hlavní slabinu YOLO (odložené zavazadlo nerozpozná),
    slabinou modelu pozadí jsou prudké změny osvětlení. Nasazení v pipeline: oba zdroje objektů (YOLO + statický).
- **Bod 5: ABODA starou cestou doběhla** (2026-10-30, `diagnose_aboda.py`, `reference_img/aboda_alerts_v2/`).
  Plná pipeline s COCO zavazadly a opraveným trackerem: alert jen ve videu 1 (**1/11**), videa 8–11 bez alertu
  (zavazadlo bez vlastníka max. 0,2 s, ve videu 11 s davem 3,2 s; nikde nedosáhne 5 s). Potvrzuje závěr bodu 3:
  slabinou je detekce odloženého zavazadla, ne pravidla; statický detektor 6/11.
- **Bod 6: rychlost zpracování** (2026-10-30, `speed_bench.py --frames 60`, volné RTX 3090, `speed_results.json`,
  log `speed_bench.log`). Čas na jeden vzorkovaný snímek (ms), přesný = konfigurace výsledků práce
  (RT-DETR, YOLO11x objekty, každý na 5 variantách snímku; ViTPose-Huge fp32), rychlý = jeden průchod + fp16.

  | modul | MED 005 přesný | rychlý | MED 021 přesný | rychlý | ShanghaiTech přesný | rychlý |
  |---|---:|---:|---:|---:|---:|---:|
  | osoby (RT-DETR) | 187 | 30 | 185 | 30 | 192 | 30 |
  | póza (ViTPose-H) | 286 | 140 | 567 | 252 | 128 | 61 |
  | objekty (YOLO11x) | 203 | 32 | 199 | 33 | 208 | 33 |
  | zbraně (YOLO11l) | 17 | 12 | 13 | 12 | 12 | 12 |
  | příznaky davu (ResNet-18 + tok) | 14 | 14 | 14 | 14 | 14 | 14 |
  | časové modely (dav, rvačka, AE) | 2 | 2 | 1 | 1 | 1 | 1 |
  | **celkem** | **709** | **230** | **979** | **342** | **556** | **152** |
  | **FPS** | **1,4** | **4,4** | **1,0** | **2,9** | **1,8** | **6,6** |
  | osob na snímek | 9,6 | | 19,3 | | 4,0 | |
  | recall osob rychlého vůči přesnému | | 0,99 | | 0,97 | | 0,87 |
  | odchylka kloubů (px) | | 0,29 | | 0,51 | | 0,22 |

  Rychlý režim je 3,1–3,7× rychlejší a na scénách MED zachová 97–99 % osob s odchylkou kloubů pod 1 px;
  na ShanghaiTech (malé vzdálené postavy) ztratí 13 % osob, které přesný režim najde jen díky variantám snímku.
  Úzké hrdlo je ViTPose-Huge (lineárně s počtem osob: ~30 ms na osobu ve fp32, ~13–15 ms ve fp16), v přesném režimu i opakovaná
  detekce na 5 variantách. Časové modely a příznaky davu jsou zanedbatelné. Při vzorkování 5 snímků/s je rychlý
  režim blízko reálného času u řídké scény (6,6 FPS), u davu ne (2,9 FPS); realtime zadání nevyžaduje.
  Obrazové modely MViT-v2-S (2026-10-31, samostatné měření na volné RTX 3090, náhodný vstup 16 × 224 × 224,
  34,2 mil. parametrů): 22 ms na klip ve fp32 a 21 ms v bf16 při dávce 1, 12,6 ms na klip v bf16 při dávce 8.
  Model se volá jednou za klip, ne na každý snímek: v demu dav každý vzorkovaný snímek, rvačka jednou za 5 s.
  Proti ViTPose je to zanedbatelné.
- **Bod 2: RWF s kratším oknem** (2026-10-29, `eval_rwf_lstm.py --holdout --seq-len L --step S`,
  výsledky `rwf_lstm_eval_T*.json`, logy `rwf_T*.log`). Stejný model a protokol jako čistý odhad výše
  (výběr epochy na 15 % trénovacích videí, val jen jako test, 3 seedy).

  | okno | krok | val videí bez okna | přesnost po videích | F1 Fight | AUC | přesnost po oknech |
  |---|---|---:|---:|---:|---:|---:|
  | 15 (3 s) | 5 | 84 | 0,781 ± 0,006 | 0,752 | 0,808 | 0,727 |
  | 15 (3 s) | 3 | 84 | 0,785 ± 0,009 | 0,757 | 0,819 | 0,729 |
  | 10 (2 s) | 5 | 56 | 0,812 ± 0,005 | 0,800 | 0,871 | 0,721 |
  | **10 (2 s)** | **3** | **56** | **0,826 ± 0,005** | **0,817** | **0,876** | 0,729 |
  | 8 (1,6 s) | 3 | 46 | 0,826 ± 0,008 | 0,820 | 0,884 | 0,712 |

  Zlepšení 0,78 → 0,83 dělá hlavně **pokrytí**: kratší okno dá předpověď víc videím (bez okna 84 → 56), samotná
  přesnost po oknech se nemění (~0,72–0,73). Hustší okna (krok 3) přidají ~1,5 bodu. **Výhrada pro text:** délka okna
  byla vybrána porovnáním na val (5 variant), takže 0,826 je mírně optimistické; varianty 10 a 8 jsou ale stejné,
  takže závěr „kratší okno pomáhá kvůli pokrytí“ je robustní. Nová hlavní konfigurace RWF: okno 10, krok 3.
  Literatura (skeletové metody) ~0,85–0,93, rozdíl se zmenšil z ~10 na ~5 bodů.
- **Úloha 3: model nad dvojicí osob** (2026-10-30, `rwf_pair.py`, výsledky `rwf_pair_results.json`, log `rwf_pair.log`).
  Původní model rvaček vidí vždy jen jednu osobu, interakci tedy nemodeluje. Každé okno osoby A dostalo nejbližšího
  souseda B ve stejných snímcích (překryv ≥ půl okna, nejmenší vzdálenost boků); soused existuje u 94 % val oken.
  Okna, osoby i pokrytí videí jsou totožné s modelem jedné osoby (ověřeno, část vstupu za A je shodná), protokol
  stejný (okno 10, krok 3, výběr epochy na 15 % trénovacích videí, val jako test, 3 seedy). Varianty dané předem:

  | vstup | rozměr | přesnost po videích | F1 Fight | AUC | přesnost po oknech |
  |---|---:|---:|---:|---:|---:|
  | jedna osoba (dosavadní) | 56 | 0,826 ± 0,005 | 0,817 | 0,876 | 0,729 |
  | A + celý skelet souseda v souřadnicích A | 113 | 0,820 ± 0,018 | 0,805 | 0,877 | 0,728 |
  | **A + poloha a rychlost souseda vůči A** | 61 | **0,833 ± 0,001** | **0,823** | **0,883** | 0,724 |

  Interakční informace pomůže jen málo: poloha a pohyb souseda přidají ~0,7 bodu a stabilizují výsledek (sd 0,001),
  celá póza souseda nepřidá nic a zvýší rozptyl (2× víc vstupů, stejně dat). Přesnost po oknech se nemění (~0,72–0,73),
  takže strop dává kvalita skeletů na RWF (nízké rozlišení, okluze, kamera z ruky) a pokrytí: 56/400 videí nemá žádné
  okno (z toho 11 Fight) a počítá se jako NonFight; na pokrytých videích je přesnost 0,837. Do textu: úloha 3 je
  řešena modelem nad dvojicí (A + relativní poloha/pohyb souseda) jako hlavní konfigurací; zisk proti jednotlivci
  je malý, ale konzistentní ve všech 3 seedech (0,833 / 0,835 / 0,833 vs. 0,820 / 0,825 / 0,833).
- **RWF-2000 v literatuře, ověřeno 2026-11-15** (accuracy na oficiální val/test části, 400 videí, stejný protokol
  jako u nás; ověřeno v textu článků, pokud není uvedeno jinak):

  | metoda | vstup | acc. | zdroj |
  |---|---|---:|---|
  | I3D (podle Hachiuma 2023, tab. 4) | RGB + tok | 83,4 | arXiv 2303.15270 |
  | Flow Gated Network (Cheng a kol., ICPR 2020) | RGB + tok | 87,25 | arXiv 1911.05913, DOI 10.1109/ICPR48806.2021.9412502 |
  | SepConvLSTM-M (Islam a kol., IJCNN 2021, tab. I) | RGB + rozdíly snímků | 89,75 | arXiv 2102.10590, DOI 10.1109/IJCNN52387.2021.9534280 |
  | SPIL (Su a kol., ECCV 2020) | skelety | 89,3 | podle Hachiuma 2023, tab. 4; DOI 10.1007/978-3-030-58548-8_5 |
  | LG-SPIL (Su, Lin, Wu, arXiv 2023) | skelety | 90,0 | arXiv 2308.13866, tab. VI (*) |
  | skelety + detekce změn (Garcia-Cobo, SanMiguel, CVIU 233, 2023) | skelety + pohyb | 90,25 | jen README autorů (*), DOI 10.1016/j.cviu.2023.103739 |
  | Structured Keypoint Pooling (Hachiuma a kol., CVPR 2023, tab. 4) | skelety | 93,4 | arXiv 2303.15270 |
  | CUE-Net (Senadeera a kol., CVPRW 2024) | RGB (UniformerV2) | 94,00 | arXiv 2404.18952 (abstrakt) |
  | **naše MViT-v2-S** | RGB | **94,9 ± 0,1** | `rwf_video_results.json` |
  | naše skelety, dvojice osob | skelety | 83,3 | `rwf_pair_results.json` |

  (*) LG-SPIL tabulka VI přečtena jen přes výtah stránky, Garcia-Cobo plný text nedostupný (ScienceDirect 403) –
  před citací v textu zkontrolovat v PDF. Obecné modely akcí podle Su 2023: TSN 81,5, 3D-ResNet101 82,6, ECO 83,7,
  TEA 86,9, TIN 87,2; bodové sítě PointNet++ 78,2, DGCNN 80,6 (stejná výhrada). DIFEM 96,5 % stále neověřeno,
  nepoužívat. Závěr: obrazový model je na úrovni nejlepších publikovaných (CUE-Net 94,00); naše skelety (83,3) jsou
  pod skeletovými metodami (89–93), nad obecnými bodovými sítěmi. Pozor: CUE-Net a další nepoužívají seedy, my
  průměr 3 seedů; rozdíl 0,9 bodu = 3–4 videa ze 400, tedy "srovnatelné", ne "lepší".
- **Úloha 3: obrazový prostorově-časový model** (2026-10-31, `rwf_video.py`, výsledky `rwf_video_results.json`,
  log `rwf_video.log`, model `models/behavior/fight_video.pt` (seed 0, mimo git)). Skeletová cesta je omezená kvalitou
  skeletů, proto doučený MViT-v2-S (multiscale vision transformer, předtrénovaný na Kinetics-400, torchvision) nad
  16 snímky rovnoměrně z celého klipu, 224 × 224, celá síť doučená s novou hlavou Fight/NonFight (AdamW 1e-4,
  max. 15 epoch, bf16). Videa znovu stažena (`download_rwf.py`), zdrojová YouTube id se mezi train a val nepřekrývají.
  Protokol stejný jako u skeletů (výběr epochy na 15 % trénovacích videí, val jen jako test, 3 seedy):

  | model | přesnost po videích | F1 Fight | AUC |
  |---|---:|---:|---:|
  | skelety, jedna osoba | 0,826 ± 0,005 | 0,817 | 0,876 |
  | skelety, dvojice (A + poloha a pohyb souseda) | 0,833 ± 0,001 | 0,823 | 0,883 |
  | **obraz: MViT-v2-S** | **0,949 ± 0,001** | **0,950** | **0,989** |
  | fúze obraz + skelety dvojic (průměr pravděpodobností) | 0,947 ± 0,004 | 0,948 | 0,983 |

  Obrazový model je o 12 bodů lepší než skelety a je na úrovni nejlepších publikovaných výsledků na RWF-2000
  (literatura 0,873–0,940, ověřeno 2026-11-15, viz tabulka výše). Pevná fúze se skelety nepomůže, skeletová větev nepřidává
  informaci, kterou obraz nemá. Výběrová přesnost na 15 % trénovacích videí 0,96–0,97, nejlepší epocha 7–13.
  Do textu: hlavní model úlohy 3 = MViT-v2-S (prostorová + časová analýza v jedné síti, zadání), skeletové modely
  jako vysvětlitelná/odlehčená varianta a ablace (osoba vs. dvojice). Rychlost: viz bod 6 (~21 ms na klip).
- **Úloha 1: obrazový prostorově-časový model davu** (2026-10-31, `med_video.py`, výsledky `med_video_results.json`
  (seed 0) a `med_video_results_s12.json` (seedy 1, 2), logy `med_video.log`, `med_video_s12.log`; model pro demo
  `models/behavior/crowd_video.pt` bez demo videí 5, 10, 15, mimo git). Stejný MViT-v2-S (Kinetics-400) jako
  u RWF, 5 tříd MED, okno 16 snímků po 5 snímcích/s (3,2 s, stejné jako model davu T=16) se středem v labelovaném
  snímku. Protokol shodný s `med_train.py`: 5 foldů podle videí, pevně 4 epochy (nic se neladí na testu), váhy tříd,
  trénink na každém 2. okně, metriky přes sloučené predikce všech 31 videí, 3 seedy:

  | model | přesnost | prům. přesnost po třídách | macro-F1 | AUC normální/abnormální |
  |---|---:|---:|---:|---:|
  | CNN + tok + osoby, Transformer T=16 (bod 7) | – | 0,549 | 0,525 | 0,818 |
  | **obraz: MViT-v2-S** | **0,867 ± 0,006** | **0,814 ± 0,007** | **0,822 ± 0,010** | **0,937 ± 0,009** |

  F1 po třídách: Neutral 0,916, Panic 0,887, Fight 0,727, Congestion 0,851 (± 0,033), Obstacle 0,730.
  Nejtěžší je fold 5 (videa 8, 9, 11, 17, 26, 31; průměrná přesnost po třídách 0,70–0,78), ostatní foldy 0,82–0,91.
  Zlepšení o ~27 bodů proti ručním příznakům, takže do textu: hlavní model úlohy 1 = MViT-v2-S, CNN + tok + osoby
  jako výchozí srovnání a ablace časové části (bod 7). Výhrada: Kinetics-400 je předtrénování na cizích datech,
  testovací videa MED v něm nejsou.
- **Bod 7: Transformer vs. BiLSTM v modelu davu** (2026-10-29, `med_train.py --compare-temporal`,
  `med_temporal_results.json`). Transformer: 2 vrstvy, 4 hlavy, učená poziční embedding, ~350 tis. parametrů
  (BiLSTM ~175 tis.), jinak stejný protokol (5 foldů podle videí, 3 seedy, 25 epoch).

  | příznaky | okno | BiLSTM: prům. přesn. po tř. / macro-F1 / AUC | Transformer: prům. přesn. po tř. / macro-F1 / AUC |
  |---|---|---|---|
  | CNN + tok | 3,2 s | 0,529 / 0,506 / 0,815 | **0,548** / **0,528** / 0,809 |
  | CNN + tok | 6,4 s | 0,545 / 0,514 / 0,779 | **0,562** / **0,529** / 0,786 |
  | CNN + tok + osoby | 3,2 s | 0,538 / 0,522 / **0,825** | **0,549** / 0,525 / 0,818 |
  | CNN + tok + osoby | 6,4 s | 0,550 / 0,519 / 0,783 | **0,564** / **0,528** / 0,789 |

  Transformer je ve všech čtyřech konfiguracích o 1–2 body lepší v průměrné přesnosti po třídách (hlavně
  Congestion a Panic), AUC normální/abnormální je srovnatelná (±0,01). Rozdíly jsou na hraně rozptylu mezi seedy
  (sd ~0,015), takže závěr do textu: obě časové architektury dávají srovnatelné výsledky, Transformer mírně lepší
  vyvážení tříd. Rychlost obou je zanedbatelná (BiLSTM 0,4–0,7 ms, Transformer 0,6–0,8 ms na okno, viz bod 6).
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
