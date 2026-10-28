# Pokračování práce na jiném PC

> Předávací dokument (2026-10-28). Shrnuje, co je hotové, co čeká na tebe a znalosti, které
> jsou jinak jen v paměti Clauda na Macu. Na novém PC stačí Claudovi napsat
> „přečti POKRACOVANI.md a pokračuj“.

## 1. Kde to stojí

- **Detektor zbraní (bod 4) je hotový a nasazený:** YOLO11l v2, třídy gun/knife/melee, větev
  ARMED v `abandoned_object_detector.py --armed`. Všechno je v `weapon_training/`, začni
  `weapon_training/docs/README.md`. Výsledky jsou v `docs/TRAINING_REPORT.md` (§6e–§6g),
  text kapitoly v `docs/KAPITOLA_DETEKCE_ZBRANI.md`.
- **Revize kódu pipeline** je v `weapon_training/docs/CODE_REVIEW_PIPELINE.md`. Body 1, 2, 3, 6, 10
  jsou opravené (včetně trackeru), otevřené jsou 4, 5, 7, 8, 9, 11.
- **Audit zbytku práce:** `STAV_PRACE.md`. Hlavní nálezy:
  - MLP 67,48 % = majoritní baseline (vždy „chůze“).
  - LSTM na MED má testovací sadu jen s jednou třídou.
  - Olabelováno je jen 196 framů.
  - Jediný skutečný výsledek klasifikace chování je RWF-2000 BiLSTM, val_acc 0,756 (data jsou na PC).

## 2. Co je na tobě (v tomhle pořadí)

1. **Rozhodnout, co bude vedle zbraní jádrem práce** (`STAV_PRACE.md`, závěr).
   - **Varianta A:** RWF BiLSTM + pravidla pro opuštěná zavazadla. Najdi na PC pickly a model RWF;
     Claude pak doplní evaluaci a text.
   - **Varianta B:** dolabelovat MED v `label_skeletons.py`. Pozor, labeler ukládá neolabelované
     framy jako „chůze“ (`label_skeletons.py:657–662`), nejdřív to nech opravit.
2. **64 sporných případů z anotace** (tyč vs. paže/stín):
   ```bash
   cd weapon_training
   python review_ambiguous.py sheets --device 0   # vytvoří review_ambiguous/list_*.png
   # vyplň sloupec rozhodnuti v review_ambiguous/rozhodnuti.csv: T = tyč, N = není, ? = nevím
   python review_ambiguous.py apply
   python build_finetune_dataset.py --out datasets/med_finetune_v3 \
       --neg-dirs annotate_neg_train annotate_neg_hard \
       --exclude-file annotation_results/exclusions_ambiguous.txt
   ```
3. **Natrénovat v3** (v2 + 41 těžkých negativ + rozhodnuté sporné případy). Hyperparametry jsou
   v `weapon_training/spark_setup_and_train_v3.sh`; neměň je, ať je srovnání s v2 čisté.
   - **Na PC s NVIDIA GPU** to jde i bez Sparku. Ve `weapon_training/` zkopíruj
     `../models/weapon_detection_yolo11l/best.pt` jako `base_best.pt`, v
     `datasets/med_finetune_v3/data.yaml` oprav `path:` na absolutní cestu na PC a spusť stejný
     příkaz `train_weapon_yolo.py` jako ve skriptu. `nice`, `--workers 2` a `--cache ram` jsou
     jen ohledy na sdílený Spark, na PC je můžeš vynechat (na Windows klidně `--workers 0`).
   - **Na Sparku:** nejdřív sám `cloudflared access login https://<spark host>`, pak Claude.
     Viz §4.
4. **Po v3 vyhodnotit stejnými testy jako v2:**
   - `eval_test005.py`: přidej v3 do slovníku `MODELS`.
   - `calibrate_melee_threshold.py --weights <v3>`.
   - `eval_temporal_filter.py` a `eval_armed_pipeline.py`: konstanta `WEIGHTS`. Cache detekcí je
     pro v2; pro v3 ji přesměruj jinam.

   Nasadit v3 jen když je lepší na testu 005 i na negativech. Pak změnit `_WEAPON_MODEL`
   v `suspicious_detector.py`.

## 3. Přenos dat (není v gitu)

Na Macu je připravený **`~/bp_prenos.tar` (2,5 GB)**. Přenes ho na PC (disk / cloud), naklonuj repo
a rozbal **do kořene klonu**:

```bash
git clone https://github.com/Demorax/Detekce_podezrelych_osob.git
cd Detekce_podezrelych_osob
tar -xf /cesta/k/bp_prenos.tar
```

Archiv obsahuje:

| Obsah | Proč |
|---|---|
| `models/weapon_detection_yolo11l*/best.pt` | základní model, v1 a nasazená v2 |
| `weapon_training/annotate_*/images` | obrázky k anotacím (labely jsou v gitu) |
| `weapon_training/datasets/weapons_merged`, `med_finetune_v2`, `med_finetune_v3` | sestavení v3 a kalibrace |
| `weapon_training/armed_eval/video_*.json` | cache detekcí pro `eval_armed_pipeline.py` / `make_demo_video.py` |
| `data/Motion_Emotion`, `data/CAVIAR` | videa MED, anotace CAVIAR |
| `clean_detekce_osob.ipynb`, `extract_skeletons.py`, `vit_pose.py`, `extract_skeletons_two_stage.ipynb`, `test_pytorch_mac_gpu.py` | tvoje necommitnuté soubory (pracovní verze z Macu) |

**Nejsou v archivu** (na PC je nejspíš máš z dřívějška): `models/rtdetr/*.pth`,
`models/vitpose/vitpose-h-multi-crowdpose.pth` (2,4 GB), `models/yolo/yolo11x.pt`, `models/vlastni_tf/`.
Necommitnuté výstupy (`skeletons_yolo_11_new/`, `frames_0.5_upscaled/004/`, nové `.npy`) zůstaly jen na Macu.

**Prostředí:** `pip install -r requirements.txt`, PyTorch nainstaluj zvlášť pro CUDA na PC (viz hlavička souboru).
Pro detektor zbraní a evaluace stačí torch + ultralytics + opencv.

## 4. Znalosti, které jinde nejsou

**Trénink:**
- **Nikdy `optimizer='auto'`** (ultralytics 8.4): vybere MuSGD a ten na tomhle datasetu diverguje.
  Vždy explicitně `SGD`.
- **Fine-tune na MED se zmrazeným backbone (`--freeze 11`)** je lepší než bez zmrazení.
- **Trénink na Apple MPS je rozbitý** (box_loss 0). Inference na MPS je v pořádku.
- **Rychlost epoch:** neodhaduj z okamžitého `s/it` v progress baru, ber čas epochy z `results.csv`.

**Data a metodika (NEMÍCHAT videa MED):**

| Role | Videa |
|---|---|
| Trénink | 001 (část), 004, 012, 014, 016, 025 |
| Těžká negativa | 003, 006, 008, 010, 011, 013, 015, 022–024, 028, 029, 031 |
| Test | 005 = jediné další video se zbraněmi; 017 + 026 = negativa; 007, 009, 018–021, 027, 030 = časové potvrzení a ARMED |

- **Práh neladit na testu 005.** Kalibrace melee proběhla jen na časové polovině (framy ≤ 902).
- **Headline číslo základního modelu je val** (mAP50 0,775), ne test split. Test split nemá
  všechny datasety.
- **`frames_0.5_upscaled/` je bikubika 4×, ne ESPCN** (ověřeno MAE). Nové framy proto upscaluj
  bikubicky.
- **Roboflow datasety:** metadata z API neodpovídají exportu, kontroluj `data.yaml`. Detaily
  v `weapon_training/docs/DATASETS.md`.

**Nasazená konfigurace (`suspicious_detector.py`, `abandoned_object_detector.py`):**
- Prahy: gun 0,35, knife 0,30, melee 0,20; COCO pálka a nůž 0,50.
- ARMED: 2 snímky po sobě, hold 8 snímků, 5 FPS.

Dopad: 4 falešné poplachy na 5,5 min rvaček bez zbraní (po opravě trackeru 5), bitka 005
pokrytá z 93 %, zpoždění 0,4 s.

**Spark (DGX, aarch64, GB10):**
- Přístup je `ssh spark2` přes Cloudflare Access. Záznam z `~/.ssh/config` a `~/.ssh` klíč
  přenes z Macu ručně, **nedávej je do gitu** (repo je veřejné).
- Login dělá vždy uživatel: `cloudflared access login …`, platí ~24 h. Nedokončený login nechá
  `~/.cloudflared/*-token.lock`, ty smaž.
- Na Sparku je potřeba torch build **cu130** (GPU sm_121).
- Je to sdílený produkční stroj, takže:
  - trénovat s `nice -n 19` a málo workery,
  - přihlašovat se zřídka,
  - **nemazat logy ani historii**.
- Roboflow klíč je jen na Sparku v `~/.config/roboflow/key`, nikdy ho necommituj.

**Pravidla pro Clauda v tomhle repu:**
- Commitovat jen na požádání: větev → ff-merge do master → push.
- Necommitovat uživatelovy soubory ze seznamu výše.
