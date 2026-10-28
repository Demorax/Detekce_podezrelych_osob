# Zdroje dat použitých v práci

Pro každý dataset: odkud byl stažen, kdy, v jaké podobě, licence/podmínky a citace (klíč do
`docs/typst/reference_data.yaml`). Velké soubory nejsou v gitu. Datasety zbraní (Roboflow) jsou
popsané zvlášť v `weapon_training/docs/DATASETS.md`.

| dataset | složka | použití | staženo | velikost | citace |
|---|---|---|---|---|---|
| Motion Emotion Dataset (MED) | `data/Motion_Emotion/` | hlavní dataset: chování davu, anotace zbraní, anomálie póz | 2025 (od vedoucího) | 31 MP4 | `med2016`, `medEmotion2016` |
| RWF-2000 | `data/RWF-2000/` | rvačky ze skeletů | 2026-04-29 | 12 GB, 2 007 AVI | `rwf2000` |
| AVA (jen anotace) | `data/AVA/annotations/` | nepoužito (videa nestažena) | 2026-04-29 | 58 MB | `ava2018` |
| CAVIAR | `data/CAVIAR/` | opuštěná zavazadla, GT boxy (XML) | 2026-05-09 | 216 MB, 28 scénářů | `caviar2004` |
| ABODA | `data/ABODA/` | opuštěná zavazadla, plná pipeline | 2026-05-09 | 513 MB, 11 AVI | `aboda2015` |
| NWPU Campus | `data/NWPU_Campus/` | falešné poplachy na HD datech (30 testovacích videí) | 2026-05 | ~76 GB (část) | `nwpu2023` |
| UCSD Anomaly (Ped1, Ped2) | `data/UCSD/` | přenos modelu davu na cizí scénu | 2026-10-28 | 740 MB | `ucsd2010` |
| CUHK Avenue | `data/Avenue/` | přenos modelu davu na cizí scénu | 2026-10-28 | 813 MB + 0,9 MB GT | `avenue2013` |
| UMN crowd | `data/UMN/` | přenos modelu davu (panika) | 2026-10-28 | 25 MB | `umn2009` |
| ShanghaiTech Campus | `data/ShanghaiTech/shanghaitech/` | benchmark skeletové anomálie (úloha 2) | 2026-10-28 | 7,0 GB (330 train videí, 107 test sekvencí) | `shanghaitech2017` |
| COCO (předtrénované váhy YOLO/RT-DETR, ImageNet pro ResNet) | `models/` | detekce osob a objektů, příznaky | — | — | `coco2014`, `imagenet2009` |

## Podrobnosti

### Motion Emotion Dataset (MED)
- Autoři: Rabiee, Haddadnia, Mousavi, Kalantarzadeh, Nabi, Murino (AVSS 2016).
- Odkaz: https://github.com/hosseinm/med, https://data.mendeley.com/datasets/bbzpxhd22j/2
- Obsah: 31 videí 854×480, 30 FPS, 44 130 snímků; anotace chování po snímcích
  (`dataset_frames_abnormal_labeling.m`: 0 nothing, 1 Panic, 2 Fight, 3 Congestion, 4 Obstacle, 5 Neutral)
  a emocí (`dataset_frames_emotion_labeling.m`).
- Licence: výzkumné použití, podle Mendeley Data CC BY 4.0 (ověřit na stránce datasetu).
- Poskytl vedoucí práce.

### RWF-2000
- Autoři: Cheng, Cai, Li (ICPR 2020).
- Odkaz: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
- Obsah: 2 000 klipů po 5 s (1 600 train / 400 val), rvačka / bez rvačky.
- Poznámka: autoři dataset 2026 stáhli z veřejné distribuce (GDPR); kopie byla pořízena 2026-04-29,
  v práci uvést jako „dataset již není veřejně dostupný“. Použití jen pro výzkum.

### CAVIAR
- Autoři: EC Funded CAVIAR project/IST 2001 37540, Fisher et al. (University of Edinburgh / INRIA).
- Odkaz: https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/
- Obsah: 28 scénářů (INRIA hala), 384×288, 25 FPS, XML anotace boxů a rolí (`role="leaving object"`).
- Licence: CC BY-SA (uvést autory).

### ABODA
- Autoři: Lin, Chen, Hung (IEEE TCSVT 2015).
- Odkaz: https://github.com/kevinlin311tw/ABODA
- Obsah: 11 sekvencí 640/720×480 AVI s opuštěnými předměty; bez veřejného GT po snímcích.
- Licence: výzkumné použití (README repozitáře).

### NWPU Campus
- Autoři: Cao, Lu, Wang, Zhang (CVPR 2023).
- Odkaz: https://campusvad.github.io/
- Obsah: 547 videí 1080p, binární GT po snímcích (`NWPU_Campus_gt.npz`); použito 30 testovacích videí.
- Licence: výzkumné použití po registraci.

### UCSD Anomaly Detection Dataset (Ped1, Ped2)
- Autoři: Mahadevan, Li, Bhalodia, Vasconcelos (CVPR 2010).
- Odkaz: http://www.svcl.ucsd.edu/projects/anomaly/dataset.html,
  soubor `UCSD_Anomaly_Dataset.tar.gz` (740 306 953 B, md5 začíná `5006421b8988`), staženo 2026-10-28.
- Obsah: Ped1 34 train / 36 test klipů 238×158, Ped2 16 train / 12 test klipů 360×240, ~10 FPS, TIF;
  GT po snímcích v `UCSDpedX.m`, pixelové masky pro část klipů v `TestXXX_gt`.
- Licence: výzkumné použití (stránka datasetu).

### CUHK Avenue
- Autoři: Lu, Shi, Jia (ICCV 2013).
- Odkaz: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html,
  soubory `Avenue_Dataset.zip` (813 227 845 B, md5 `b7a34b212ecd…`) a `ground_truth_demo.zip`
  (931 269 B, md5 `e8e3bff99195…`), staženo 2026-10-28.
- Obsah: 16 trénovacích a 21 testovacích videí 640×360, 25 FPS; GT pixelové masky po snímcích
  (`testing_label_mask/N_label.mat`, label snímku = maska neprázdná).
- Licence: výzkumné použití (stránka datasetu).

### UMN Unusual Crowd Activity
- Autoři: University of Minnesota, Mehran, Oyama, Shah (CVPR 2009) dataset poprvé použili.
- Odkaz: https://mha.cs.umn.edu/proj_events.shtml#crowd, soubor `Crowd-Activity-All.avi`
  (24 821 310 B, md5 `14bfa21c6e02…`), staženo 2026-10-28.
- Obsah: jedno video 7 739 snímků 320×240, 30 FPS; 3 scény (1 453 / 4 143 / 2 143 snímků),
  11 klipů, každý končí panikou. Oficiální GT po snímcích není; anotace odvozena z nápisu
  „Abnormal Crowd Activity“ vypáleného do obrazu (`umn_labels.py` → `umn_labels.npz`, 1 136 abnormálních snímků).
- Licence: výzkumné použití.

### ShanghaiTech Campus
- Autoři: Luo, Liu, Gao (ICCV 2017); distribuce z repozitáře Liu et al. (CVPR 2018).
- Odkaz: https://svip-lab.github.io/dataset/campus_dataset.html,
  https://github.com/StevenLiuWen/ano_pred_cvpr2018 (OneDrive `shanghaitech.tar.gz`).
- Obsah: 330 trénovacích a 107 testovacích videí 856×480, 13 scén, GT po snímcích i pixelové masky.
- Soubor: `shanghaitech.tar.gz` (6 992 962 418 B, md5 začíná `08494decd30f`), stažen ručně z OneDrive
  2026-10-28 jako samostatný soubor (stažení celé složky jako zip se při ~7,5 GB usekne).
- Obsah ověřen: 330 trénovacích videí (všechny scény 01–13), 107 testovacích sekvencí, 40 791 testovacích
  snímků (17 326 abnormálních, `test_frame_mask/*.npy`), pixelové masky `test_pixel_mask/`.
- Skelety: `shanghai_skeletons.py` -> `data/ShanghaiTech/skeletons/{training,testing}/`.

### Roboflow Universe (datasety zbraní)
- Viz `weapon_training/docs/DATASETS.md` (6 datasetů, licence CC BY 4.0 a další, odkazy a verze).
