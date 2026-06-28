# Datasety a zdroje — analýza pohybu osob ve videu

> **Sestaveno:** 2026-04-29
> **Tematický rozsah:** prostorově-temporální analýza pohybu osob, anomaly detection v surveillance, violence detection, skeleton-based action recognition.
> **Vazba na zadání BP:** _Analýza pohybu osob ve videu s využitím neuronových sítí pro prostorově-temporální zpracování_ — cíle: (a) normální/abnormální pohyb davu, (b) abnormální pózy jednotlivců, (c) interakce mezi jednotlivci.
> **Cíl tohoto dokumentu:** rozšířit seznam zdrojů z původního zadání děkana o relevantní datasety a referenční literaturu, abych se mohl rozhodnout, který je vhodné integrovat do tréninku LSTM klasifikátoru.

---

## Proč jsem některé z těchto datasetů nenašel hned

Akademická komunita pro tento problém **nepoužívá pojem "behavior labeling for surveillance"**. V relevantní literatuře úloha spadá pod různé klíčové pojmy:

| Pojem | Co pokrývá |
|---|---|
| **Anomaly detection (in surveillance video)** | Cokoli neobvyklého v CCTV — nejširší pojem |
| **Violence detection** / **Fight detection** | Specificky agresivní fyzické chování |
| **Action recognition** | Obecná klasifikace akcí, často v controlled prostředí |
| **Skeleton-based action recognition** | Klasifikace nad keypointy (přímý match k mé pipeline) |
| **Crowd behavior analysis** | Davová dynamika, nejen jednotlivci |
| **Group activity recognition** | Interakce a kolektivní akce |

Hledat je třeba pod těmito pojmy + "dataset" + "benchmark", ne pod "behavior labels".

---

## ⭐ TOP DOPORUČENÍ — kam jít první

> **⚠ UPDATE 2026-04-29:** RWF-2000 byl autory stažen kvůli privacy (GDPR / souhlas natáčených osob). Jako náhrada doporučuji **Real-Life Violence Situations Dataset** (Kaggle) — stejný binary task, podobná velikost.

| # | Dataset | Velikost | Labels | Styl | Status | Proč |
|---|---|---|---|---|---|---|
| 1 | **Real-Life Violence Situations** | 2000 klipů × ~5s | binary violence/non | mix CCTV + YouTube | ✓ dostupné (Kaggle) | Funkční náhrada RWF-2000, snadný download |
| 2 | **UCF-Crime** | 1900 videí, 128h | 13 anomaly tříd | real CCTV | ✓ dostupné | Z původního zadání děkana, fine-grained třídy |
| 3 | **XD-Violence** | 4754 videí, 217h | 6 violence tříd | mix CCTV/film/YouTube | ✓ dostupné | Velký, multi-modal (audio i video) |
| 4 | **NTU RGB+D 60/120** | 56-114k klipů | 60-120 akcí + skeleton | studio | ✓ dostupné (request) | Už má skelety — odpadá ViTPose krok |
| ~~5~~ | ~~**RWF-2000**~~ | ~~2000 × 5s~~ | ~~binary~~ | ~~real surveillance~~ | ✗ **STAŽENO** (privacy) | viz cesta C níže pro alternativní mirrory |

**Doporučený postup pro mě:**
1. **Real-Life Violence Situations** jako proof-of-concept (~1 den práce: Kaggle download → extract frames → moje pipeline → trénink)
2. Když pipeline funguje a je smysluplná accuracy → přidat **UCF-Crime subset** (Fighting, Assault, Robbery, Vandalism)
3. Volitelně **NTU RGB+D** pro skeleton-only baseline (jiný formát skeletonu = work navíc, ale velký dataset)

**Cesta C — pokud opravdu chci RWF-2000:**
- GitHub repo `mchengny/RWF2000-Video-Database-for-Violence-Detection` — možný alternative download instructions
- Academic Torrents (academictorrents.com) — research datasety jsou tam často zrcadlené
- Email autorům (Cheng et al.) — sometimes share po podpisu data agreement

---

# 0. ⭐ NOVÉ: Pre-incident / abandoned object datasety (2026-05-09)

> **Pivot směru:** Po V1+V2 LSTM tréninku (binary fight/no-fight) jsme zjistili, že tento přístup je _reactive_ (detekuje incident až za běhu). Cíl je _proactive_: detekce **precursors** (odložený batoh, loitering, podezřelé nesení). Viz [`plan_pre_incident.md`](plan_pre_incident.md) pro detail.

## 0.1 ABODA — Abandoned Object Dataset

- **Autor:** Lin et al. (Sinica IIS, Taiwan)
- **Velikost:** 11 sekvencí, real CCTV
- **Annotace:** ground truth bboxy + labels pro abandoned events
- **Stažení:** [github.com/kevinlin311tw/ABODA](https://github.com/kevinlin311tw/ABODA)
- **Licence:** research, free
- **Vhodnost:** ⭐⭐⭐⭐⭐ — primární eval pro abandonment rule

## 0.2 PETS 2006 — Left Luggage

- **Velikost:** 7 multi-sensor sekvencí (4 kamery každá)
- **Stažení:** [cvg.reading.ac.uk/PETS2006](http://www.cvg.reading.ac.uk/PETS2006/data.html), FTP `ftp.pets.rdg.ac.uk`
- **Licence:** academic research, ICO-approved
- **Vhodnost:** ⭐⭐⭐⭐⭐ — klasický benchmark, BP-friendly citace

## 0.3 PETS 2007 — Left Luggage + Loitering

- **Velikost:** 9 sekvencí, 4 kamery
- **Speciální:** **loitering** definováno jako pobyt > 60s
- **Stažení:** [cvg.reading.ac.uk/PETS2007](http://www.cvg.reading.ac.uk/PETS2007/data.html)
- **Vhodnost:** ⭐⭐⭐⭐⭐ — pokrývá obě úlohy (abandonment + loitering)

## 0.4 AVSS 2007 (i-LIDS subset)

- **Velikost:** 15 sekvencí, londýnské metro, 3 obtížnostní úrovně
- **Charakteristika:** crowded scenes, severe occlusion
- **Stažení:** [eecs.qmul.ac.uk/~andrea/avss2007_d.html](https://www.eecs.qmul.ac.uk/~andrea/avss2007_d.html)
- **Vhodnost:** ⭐⭐⭐⭐ — challenging eval

## 0.5 CCTV-KD (2024)

- **Velikost:** 570 obrázků, 11 104 anotací (7890 person + 3214 baggage)
- **Lokace:** Korzo pedestrian zone + Düsseldorf airport
- **Vhodnost:** ⭐⭐⭐ — moderní, ale jen single-frame anotace (ne sekvence)

---

# 1. Anomaly detection / suspicious behavior v CCTV

## 1.1 Motion Emotion Dataset (MED) — UŽ POUŽÍVÁM

- **Autoři:** Hosseini Mahdi et al.
- **Velikost:** 31 videí
- **Labels:** crowd emotions (6 tříd: angry, happy, excited, scared, sad, neutral)
- **Styl:** mix surveillance / public space
- **Stažení:** [github.com/hosseinm/med](https://github.com/hosseinm/med), [data.mendeley.com/datasets/bbzpxhd22j/2](https://data.mendeley.com/datasets/bbzpxhd22j/2)
- **Popis:** [PMC article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10827673/)
- **Pozn.:** Tvořím z něj svůj ruční dataset. Class imbalance: ~96% walking, 0% suspicious.

## 1.2 UCF-Crime ⭐ — Z PŮVODNÍHO ZADÁNÍ

- **Autoři:** Sultani et al., 2018, "Real-world Anomaly Detection in Surveillance Videos" (CVPR)
- **Velikost:** 1900 untrimmed videí (~128 hodin), průměr 4 minuty/video
- **Labels:** 13 abnormálních tříd: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism + Normal
- **Granularita:** video-level labels (weakly-supervised), test set má frame-level temporal annotations
- **Styl:** real CCTV (rozličné kamery, prostředí, kvality)
- **Velikost ke stažení:** ~120 GB (full HD)
- **Stažení:** [crcv.ucf.edu/projects/real-world](https://www.crcv.ucf.edu/projects/real-world/), Dropbox/MEGA mirrory v paper repu
- **Licence:** research-only, free
- **Vhodnost pro LSTM:** ⭐⭐⭐⭐⭐ — pokrývá všechny 3 cílové kategorie (suspicious, fighting, normal)
- **Gotcha:** Velký, untrimmed → potřeba clip-level segmentace. Začni s 100-200 GB subsetem (např. jen Fighting + Robbery + Assault).

## 1.3 ShanghaiTech Campus ⭐ — Z PŮVODNÍHO ZADÁNÍ

- **Autoři:** Liu et al., 2018, "Future Frame Prediction for Anomaly Detection — A New Baseline" (CVPR)
- **Velikost:** 437 videí, 13 různých scén (univerzitní kampus)
- **Labels:** pixel-level anomaly maps (jemnější než UCF-Crime)
- **Styl:** real CCTV jednoho kampusu
- **Stažení:** [github.com/StevenLiuWen/ano_pred_cvpr2018](https://github.com/StevenLiuWen/ano_pred_cvpr2018) (data.zip ~5GB)
- **Vhodnost:** ⭐⭐⭐⭐ — high-quality labels, ale jen jeden setting, takže může být přeoptimalizace

## 1.4 UCSD Anomaly Detection (Ped1, Ped2) — Z PŮVODNÍHO ZADÁNÍ

- **Autoři:** Mahadevan et al., 2010, UCSD
- **Velikost:** Ped1 (34 train, 36 test videí) + Ped2 (16 train, 12 test)
- **Labels:** frame-level abnormality + bounding boxes pro některé framy
- **Styl:** chodník na kampusu UCSD, low resolution (158×238 / 240×360)
- **Anomálie:** cyklisti, skateboarding, vozidla, invalidní vozíky na chodníku
- **Stažení:** [svcl.ucsd.edu/projects/anomaly](http://svcl.ucsd.edu/projects/anomaly/dataset.html)
- **Vhodnost:** ⭐⭐ — historický benchmark, ale nízké rozlišení a omezená diverzita; spíš pro reprodukci článků

## 1.5 CUHK Avenue

- **Autoři:** Lu et al., 2013, "Abnormal Event Detection at 150 FPS in MATLAB"
- **Velikost:** 16 train + 21 test videí
- **Labels:** frame-level + spatial bounding box
- **Styl:** ulice mezi budovami, jedna kamera
- **Anomálie:** běh, házení předmětů, jízda na kole
- **Stažení:** [cse.cuhk.edu.hk/leojia/projects/detectabnormal](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)
- **Vhodnost:** ⭐⭐ — malý, klasický benchmark

## 1.6 Street Scene

- **Autoři:** Ramachandra & Jones, 2020 (WACV)
- **Velikost:** 46 train + 35 test videí
- **Labels:** spatio-temporal masks pro 17 typů anomálií
- **Styl:** městská ulice, jedna kamera (dlouhé sledování)
- **Anomálie:** vozidla mimo dráhu, chodci na silnici, parkovací anomálie
- **Vhodnost:** ⭐⭐⭐ — moderní, dobré annotace, ale jeden viewpoint

## 1.7 XD-Violence

- **Autoři:** Wu et al., 2020, "Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision" (ECCV)
- **Velikost:** 4754 videí (217 hodin)
- **Labels:** 6 violence tříd (Abuse, CarAccident, Explosion, Fighting, Riot, Shooting) + Normal
- **Styl:** mix CCTV / filmy / YouTube
- **Speciální:** **multi-modal** — má i synchronizovaný audio track
- **Stažení:** [roc-ng.github.io/XD-Violence](https://roc-ng.github.io/XD-Violence/)
- **Vhodnost:** ⭐⭐⭐⭐ — větší než UCF-Crime, novější, ale heterogenní zdroje (ne jen surveillance)

## 1.8 NWPU Campus

- **Autoři:** Cao et al., 2023, "A New Comprehensive Benchmark for Semi-supervised Video Anomaly Detection and Anticipation" (CVPR)
- **Velikost:** 305 train + 242 test videí, 28 train + 14 test scén
- **Labels:** 28 typů anomálií, frame-level
- **Styl:** kampus NWPU (Northwestern Polytechnical University)
- **Speciální:** větší a diverznější než ShanghaiTech, podporuje anomaly **anticipation** (predikce)
- **Stažení:** [github.com/cqzc/NWPU-Campus](https://github.com/ChangsongCao/NWPU-Campus-Dataset) (ověř nejnovější mirror)
- **Vhodnost:** ⭐⭐⭐⭐ — moderní, kvalitní benchmark, dobré pro SOTA srovnání

## 1.9 UBnormal

- **Autoři:** Acsintoae et al., 2022 (CVPR)
- **Velikost:** 543 videí, syntetické (Cinema 4D) + reálné
- **Labels:** action labels pro anomálie (na rozdíl od většiny ostatních datasetů)
- **Styl:** synthetic urban scenes
- **Speciální:** explicit action types (running, sleeping, jaywalking...) místo jen "anomaly"
- **Stažení:** [github.com/lilygeorgescu/UBnormal](https://github.com/lilygeorgescu/UBnormal)
- **Vhodnost:** ⭐⭐⭐ — pokud chci kombinaci syntetických + reálných dat s explicitními labely

## 1.10 MSAD (Multi-Scenario Anomaly Detection)

- **Rok:** 2024
- **Velikost:** 720 videí, 14 různých scenarios
- **Labels:** multi-class
- **Vhodnost:** ⭐⭐⭐ — novější, ale méně etablovaný

## 1.11 Subway Entry/Exit

- **Autoři:** Adam et al., 2008
- **Velikost:** 2 dlouhá videa (1h30m + 43m)
- **Labels:** anomaly events (running, wrong direction, loitering)
- **Vhodnost:** ⭐ — historický, hodně omezený

---

# 2. Violence / Fight Detection

## 2.1 RWF-2000 ⭐⭐⭐⭐⭐ DOPORUČENO START

- **Autoři:** Cheng et al., 2021, "RWF-2000: An Open Large Scale Video Database for Violence Detection"
- **Velikost:** 2000 klipů, každý ~5 sekund
- **Labels:** binary (Fight / NonFight), 1000 + 1000 split
- **Styl:** **real-world surveillance footage** (CCTV-like)
- **Velikost stažení:** ~7 GB
- **Stažení:** [github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)
- **Licence:** research-only (vyžaduje email s žádostí)
- **Vhodnost pro mě:** ⭐⭐⭐⭐⭐
  - Manageable velikost (zpracovatelné za odpoledne)
  - Real surveillance styl (matches Motion Emotion)
  - Per-clip label = jednoduchá integrace
  - 1000 fight klipů = okamžité naplnění chybějící "suspicious" třídy
  - 1000 non-fight = další "walking/normal" data

## 2.2 Real-Life Violence Situations (Kaggle)

- **Velikost:** 2000 videí (1000 violent, 1000 non-violent)
- **Labels:** binary
- **Styl:** mix YouTube / surveillance
- **Stažení:** Kaggle, search "Real Life Violence Situations Dataset"
- **Vhodnost:** ⭐⭐⭐ — alternativa nebo doplněk k RWF-2000, ale méně CCTV-like

## 2.3 Hockey Fight

- **Autoři:** Bermejo Nievas et al., 2011
- **Velikost:** 1000 klipů (500 fight + 500 non-fight)
- **Styl:** ledový hokej (NHL záběry)
- **Vhodnost:** ⭐⭐ — niche, ale klasický baseline pro violence detection

## 2.4 Movies Fight Dataset

- **Autoři:** Bermejo Nievas et al., 2011
- **Velikost:** 200 videí
- **Styl:** akční filmy
- **Vhodnost:** ⭐ — velmi malý

## 2.5 Crowd Violence

- **Autoři:** Hassner et al., 2012
- **Velikost:** 246 videí (123 violent + 123 non-violent crowds)
- **Styl:** zástup lidí v davech
- **Vhodnost:** ⭐⭐⭐ — pokud cílím na **interakce v davech** (jeden z cílů zadání)

## 2.6 CCTV-Fights

- **Autoři:** Perez et al., 2019
- **Velikost:** 1000 fight klipů z reálných CCTV
- **Styl:** real surveillance
- **Vhodnost:** ⭐⭐⭐ — pokud potřebuji víc fight dat než RWF nabízí

---

# 3. Skeleton-based action recognition

> Tyto datasety už **mají extrahované keypointy** — odpadá tak úplně 1. a 2. fáze pipeline (RT-DETR + ViTPose). Stačí trénovat LSTM/GCN/Transformer přímo na skeletonech.

## 3.1 NTU RGB+D 60 ⭐

- **Autoři:** Shahroudy et al., 2016 (CVPR)
- **Velikost:** 56,880 klipů
- **Labels:** 60 akčních tříd (eat, drink, fall, punch, kick, push, take photo, ...)
- **Modality:** RGB video, depth, IR, **3D skeleton (Kinect, 25 joints)**
- **Styl:** studio, jeden subject nebo dva v interakci, multiple camera angles
- **Stažení:** [rose1.ntu.edu.sg/dataset/actionRecognition](https://rose1.ntu.edu.sg/dataset/actionRecognition/) (vyžaduje request)
- **Vhodnost:** ⭐⭐⭐⭐
  - **Pre-extracted skeletons** → rychlý training
  - Akce jako falling/punching/kicking jsou relevantní pro suspicious behavior
- **Gotcha:** **Kinect 25-joint skeleton ≠ tvůj COCO 17-joint** → musím udělat conversion (mapping subset 17 z 25 nebo retrain LSTM na 25)

## 3.2 NTU RGB+D 120

- **Velikost:** 114,480 klipů, 120 tříd
- **Vhodnost:** ⭐⭐⭐⭐ — extension, větší a více tříd

## 3.3 Kinetics-Skeleton

- **Autoři:** Yan et al., 2018 (paper o ST-GCN)
- **Popis:** skeletony extrahované z Kinetics-400 přes OpenPose (18 keypoints)
- **Velikost:** ~250k klipů (subset Kinetics-400 s úspěšnou pose estimation)
- **Stažení:** [github.com/yysijie/st-gcn](https://github.com/yysijie/st-gcn) (s návodem)
- **Vhodnost:** ⭐⭐⭐ — velký, ale Kinetics akce jsou často sportovní/everyday, ne surveillance-specific

## 3.4 PKU-MMD

- **Autoři:** Liu et al., 2017
- **Velikost:** 1076 dlouhých videí, 51 akčních tříd
- **Modality:** RGB + Depth + IR + skeleton
- **Speciální:** untrimmed multi-action sequences (vhodné pro detection ne jen recognition)
- **Vhodnost:** ⭐⭐⭐

## 3.5 SBU Kinect Interaction

- **Autoři:** Yun et al., 2012
- **Velikost:** 282 videí, 8 tříd two-person interakcí
- **Akce:** kicking, punching, pushing, hugging, ... → **přímo "interakce mezi jednotlivci"** z mého zadání
- **Vhodnost:** ⭐⭐⭐ — malý, ale přesný match k cílu c) z mého zadání

## 3.6 Volleyball Activity Dataset

- **Autoři:** Ibrahim et al., 2016
- **Velikost:** 4830 klipů
- **Speciální:** **group activity** (kolektivní akce týmu)
- **Vhodnost:** ⭐⭐ — niche, ale přesně řeší group activity

---

# 4. General action recognition (pro pretraining / transfer learning)

## 4.1 Kinetics-400 / 600 / 700

- **Autoři:** Carreira et al., DeepMind
- **Velikost:** 240k / 480k / 650k YouTube klipů (10s každý)
- **Labels:** 400 / 600 / 700 akčních tříd
- **Stažení:** [deepmind.com/research/open-source/kinetics](https://deepmind.com/research/open-source/kinetics) (skripty, ne přímý dataset)
- **Vhodnost:** ⭐⭐⭐⭐ — defactostandard pro pretraining video modelů; **použij jako pretrained backbone** ne jako primární dataset

## 4.2 AVA / AVA-Kinetics

- **Autoři:** Gu et al., 2018, "AVA: A Video Dataset of Spatio-Temporally Localized Atomic Visual Actions" (CVPR)
- **Velikost:** 430 videí (15 min každé), AVA-Kinetics navíc 230k klipů
- **Labels:** 80 akčních tříd, **bounding box + action per person per frame** (1Hz)
- **Speciální:** unique frame-level person-action annotations (perfektní pro mou pipeline)
- **Stažení:** [research.google.com/ava](https://research.google.com/ava/)
- **Vhodnost:** ⭐⭐⭐⭐⭐ — pokud chci **per-person action labels**

## 4.3 UCF-101

- **Autoři:** Soomro et al., 2012
- **Velikost:** 13,320 videí, 101 tříd
- **Vhodnost:** ⭐⭐⭐ — klasický benchmark, ale akce jsou hlavně sportovní

## 4.4 HMDB-51

- **Autoři:** Kuehne et al., 2011
- **Velikost:** 6766 klipů, 51 tříd
- **Vhodnost:** ⭐⭐ — historický

## 4.5 ActivityNet

- **Autoři:** Caba Heilbron et al., 2015
- **Velikost:** 20,000 videí, 200 tříd
- **Speciální:** untrimmed s temporal action localization
- **Vhodnost:** ⭐⭐⭐

## 4.6 Something-Something v2

- **Autoři:** Goyal et al., 2017
- **Velikost:** 220k videí, 174 tříd
- **Speciální:** crowd-sourced, focused on object-interaction actions
- **Vhodnost:** ⭐ — niche, není surveillance

---

# 5. Crowd behavior / density analysis

## 5.1 ShanghaiTech Crowd (POZOR — NE Campus!)

- **Autoři:** Zhang et al., 2016 (CVPR)
- **Velikost:** 1198 obrázků, ~330k anotovaných osob
- **Úloha:** crowd counting, ne behavior
- **Vhodnost:** ⭐⭐ — relevantní pokud chci crowd density jako feature

## 5.2 UCF_CC_50

- **Autoři:** Idrees et al., 2013
- **Velikost:** 50 obrázků, ~64k osob (extrémně husté davy)
- **Vhodnost:** ⭐ — niche, jen counting

## 5.3 WorldExpo'10

- **Autoři:** Zhang et al., 2015
- **Velikost:** 1132 anotovaných obrázků z 108 video sekvencí
- **Speciální:** multi-camera Shanghai Expo footage
- **Vhodnost:** ⭐⭐ — counting + nějaká motion data

## 5.4 Mall

- **Autoři:** Chen et al., 2012
- **Velikost:** 2000 framů z mall surveillance
- **Vhodnost:** ⭐⭐ — counting

## 5.5 CUHK Crowd

- **Autoři:** Shao et al., 2014
- **Velikost:** 474 videí
- **Speciální:** crowd behavior labels (running, walking, gathering, dispersion)
- **Vhodnost:** ⭐⭐⭐⭐ — **přímo crowd behavior labels** (z cíle a) zadání!)

## 5.6 PETS 2009 / 2017

- **Velikost:** několik scénářů, ~miliony framů
- **Speciální:** multi-camera, multi-view, person tracking + density
- **Stažení:** [cs.binghamton.edu/~mrldata/pets2009](http://cs.binghamton.edu/~mrldata/pets2009)
- **Vhodnost:** ⭐⭐⭐ — classic surveillance benchmark, výborně dokumentované

## 5.7 Crowd-11

- **Autoři:** Dupont et al., 2017
- **Velikost:** 6262 videí, 11 crowd behavior tříd (gas/liquid/solid + various)
- **Vhodnost:** ⭐⭐⭐ — explicitní crowd behavior klasifikace

---

# 6. Person detection a tracking (referenční)

## 6.1 MOTChallenge (MOT15-MOT20)

- **Velikost:** desítky videí pro každý ročník
- **Úloha:** multi-object tracking (assoc. mezi framy)
- **Vhodnost:** ⭐⭐⭐ — pro **person tracking** ve vlastní pipeline místo simple centroid

## 6.2 CrowdHuman

- **Autoři:** Shao et al., 2018
- **Velikost:** 24,370 obrázků, ~470k anotovaných lidí
- **Speciální:** dense crowd pedestrian detection
- **Vhodnost:** ⭐⭐⭐ — pretraining detektoru pro davy

## 6.3 VIRAT

- **Autoři:** Oh et al., 2011 (DARPA)
- **Velikost:** stovky hodin outdoor surveillance
- **Labels:** activity annotations
- **Vhodnost:** ⭐⭐⭐ — komplexní real-world surveillance

---

# 7. Pose estimation (referenční datasety)

## 7.1 CrowdPose — UŽ POUŽÍVÁM (přes ViTPose)

- **Autoři:** Li et al., 2019
- **Velikost:** ~20k obrázků, dense crowd
- **Stažení:** [github.com/Jeff-sjtu/CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)

## 7.2 COCO Keypoints

- 17 keypoints, 250k osob
- Defacto standard pro 2D pose

## 7.3 MPII Human Pose

- 25k images, 16 keypoints
- Single-person focus, classic

---

# 8. Klíčové papery k tématu

## 8.1 Spatio-temporální video modely (relevant pro architektury)

- **C3D**: Tran et al., 2015, "Learning Spatiotemporal Features with 3D Convolutional Networks" (ICCV)
- **Two-Stream Networks**: Simonyan & Zisserman, 2014, "Two-Stream Convolutional Networks for Action Recognition in Videos" (NeurIPS)
- **I3D**: Carreira & Zisserman, 2017, "Quo Vadis, Action Recognition?" (CVPR)
- **SlowFast**: Feichtenhofer et al., 2019, "SlowFast Networks for Video Recognition" (ICCV)
- **TimeSformer**: Bertasius et al., 2021, "Is Space-Time Attention All You Need for Video Understanding?" (ICML)
- **VideoMAE**: Tong et al., 2022 (NeurIPS)
- **ConvLSTM**: Shi et al., 2015, "Convolutional LSTM Network" (NeurIPS) — relevant k LSTM části zadání

## 8.2 Anomaly detection v surveillance

- **Sultani et al., 2018**, "Real-world Anomaly Detection in Surveillance Videos" (CVPR) — UCF-Crime paper, MIL-based weakly-supervised
- **Liu et al., 2018**, "Future Frame Prediction for Anomaly Detection" (CVPR) — ShanghaiTech paper
- **Gong et al., 2019**, "Memorizing Normality to Detect Anomaly" (ICCV) — memory autoencoder
- **Wang et al., 2020**, "Cluster Attention Contrast for Video Anomaly Detection" (ACM MM)
- **Tian et al., 2021**, "Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning" (ICCV) — RTFM
- **Chen et al., 2023**, "MGFN: Magnitude-Contrastive Glance-and-Focus Network" — SOTA na UCF-Crime

## 8.3 Skeleton-based action recognition

- **ST-GCN**: Yan et al., 2018, "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition" (AAAI) — defining paper
- **2s-AGCN**: Shi et al., 2019, "Two-Stream Adaptive Graph Convolutional Networks" (CVPR)
- **MS-G3D**: Liu et al., 2020 (CVPR)
- **PoseConv3D**: Duan et al., 2022, "Revisiting Skeleton-based Action Recognition" (CVPR) — používá heatmap volumes místo grafů

## 8.4 Violence detection specifically

- **Bermejo Nievas et al., 2011**, "Violence Detection in Video Using Computer Vision Techniques"
- **Cheng et al., 2021**, "RWF-2000" paper
- **Wu et al., 2020**, "Not only Look, but also Listen" — XD-Violence multi-modal
- **Sudhakaran & Lanz, 2017**, "Learning to Detect Violent Videos using Convolutional Long Short-Term Memory" — ConvLSTM přímo na violence

## 8.5 Pose estimation (pro pipeline)

- **ViTPose**: Xu et al., 2022 (NeurIPS) — můj 2-stage stage 2
- **RT-DETR**: Lv et al., 2023 (CVPR) — můj 2-stage stage 1
- **OpenPose**: Cao et al., 2019 (TPAMI) — bottom-up alternativa
- **HRNet**: Sun et al., 2019 (CVPR) — top-down alternativa

---

# 9. Praktické integrační poznámky

## 9.1 Licenční reality check

Většina výzkumných datasetů je **research-only**. Pro BP/diplomku tohle stačí, ale:
- **UCF-Crime, NTU RGB+D, RWF-2000** vyžadují vyplnit form / poslat email
- **AVA Actions** je free pro research (Google)
- **Kinetics** je v podstatě seznam YouTube URL — některá videa zmizí časem

## 9.2 Format unification — můj problém

Můj pipeline produkuje **17 keypoints (COCO/CrowdPose)**. Externí datasety mají různé formáty:

| Dataset | Skeleton format | Konverze |
|---|---|---|
| Motion Emotion (já) | 17 COCO/CrowdPose | — |
| NTU RGB+D | 25 Kinect joints | mapping subset 17 z 25 (lose info) NEBO retrain LSTM na 25 |
| Kinetics-Skeleton | 18 OpenPose | downsample na 17 (drop neck) |
| AVA | bbox + action (no skeleton) | rozjet ViTPose nad jejich bboxy |
| RWF-2000 | raw video | rozjet celý můj 2-stage pipeline |
| UCF-Crime | raw video | rozjet celý můj 2-stage pipeline |

**Doporučení:** unifikovat na 17 COCO keypoints přes můj existující ViTPose pipeline. Tím se ztratí info z NTU 25, ale bude konzistence s mým primary datasetem.

## 9.3 Class imbalance strategie (kritický problém)

Aktuálně mám:
- ~96% walking
- ~4% running
- 0% suspicious

Po integraci RWF-2000 (1000 fight klipů → ~10k+ skeleton sekvencí):
- ~50% walking
- ~5% running
- ~45% suspicious (z fight klipů)

→ Classes budou víc balanced. Můžu pak ještě:
- **Class weights** v cross-entropy
- **Focal loss**
- **Oversampling** menšinových tříd

## 9.4 Workflow doporučení (sekvenčně)

1. ✅ **Mít hotovou pipeline** (RT-DETR + ViTPose) — DONE (2026-04)
2. **Stáhnout RWF-2000** (~7GB)
3. **Procesovat přes pipeline:**
   ```
   RWF2000_videos/Fight/*.mp4 → frames → skeletons.npy
   RWF2000_videos/NonFight/*.mp4 → frames → skeletons.npy
   ```
4. **Auto-label** všechny skelety v Fight klipu jako class 1 ("suspicious"), v NonFight jako 0 ("walking")
5. **Sloučit s ručním Motion Emotion datasetem**
6. **Trénink LSTM** s class weights nebo focal loss
7. **Eval na hold-out + cross-dataset eval** (trénovat na MED, testovat na RWF a obráceně)
8. (Volitelně) **Přidat UCF-Crime fine-grained subset** pro multi-class (Fighting/Robbery/Vandalism distinkce)

---

# 10. Shrnutí: Co dělat dál

**Pro plnění cíle a) "normální/abnormální pohyb davu":**
- RWF-2000 (start), UCF-Crime, ShanghaiTech, CUHK Crowd, Crowd-11

**Pro plnění cíle b) "abnormální pózy jednotlivců":**
- NTU RGB+D (skeleton-level), AVA (per-person actions), UBnormal (action labels)

**Pro plnění cíle c) "interakce mezi jednotlivci":**
- SBU Kinect Interaction, Volleyball Activity, Crowd Violence, NTU RGB+D mutual interactions subset

**Pro pretraining backbonu:**
- Kinetics-400/600 (defacto standard), AVA, UCF-101

**Konkrétní next step:**
Stáhnout **RWF-2000** a integrovat přes existující pipeline. Pokud funguje, expand na **UCF-Crime** subset (Fighting + Assault + Vandalism + Normal).
