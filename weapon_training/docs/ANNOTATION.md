# Anotace framů pro domain adaptation — návod

Cíl: doplnit ~200–300 anotovaných framů z **cílové domény** (CCTV shora)
a dotrénovat na nich existující model. To je podle výsledků nejúčinnější
jediný krok — viz [`TRAINING_REPORT.md`](TRAINING_REPORT.md) §6.3 a §8.

Nástroj **nenahrazuje** `label_skeletons.py`. Ten řeší jinou úlohu: tracking
osoby napříč framy a labelování chování v čase (walking / suspicious / running).
Tady se anotují **objekty v jednotlivých framech** pro detektor.

---

## Připraveno k anotaci

```
weapon_training/annotate_med/     (115 MB, už vygenerováno)
├── images/     208 MED framů (videa 001, 002, 004)
├── labels/     předanotace od modelu — 78 kandidátních boxů v 53 framech
└── classes.txt gun, knife, melee
```

Předanotace vznikly puštěním natrénovaného modelu s **conf = 0,10**, tedy
mnohem níž než v produkci (0,30–0,40). Záměr je vysoký recall: každý zachycený
kandidát ušetří kreslení, falešný se smaže jedním stiskem. Při produkčním prahu
by model našel jen 17 boxů, při 0,10 jich je **78**.

---

## Spuštění

```bash
cd weapon_training
python annotate_boxes.py --dir annotate_med
```

Závislosti: `tkinter`, `PIL`, `numpy` — **žádné nové**, a záměrně **bez `cv2`**,
protože na tomto Macu není nainstalováno.

---

## Ovládání

| Klávesa / akce | Význam |
|---|---|
| **tažení myší** | nový box |
| **klik na box** | vybrat |
| **tažení rohu** | změnit velikost |
| **tažení uvnitř** | posunout |
| `1` `2` `3` | přepnout třídu (a přeřadit vybraný box) |
| `DEL` / `BACKSPACE` | smazat vybraný box |
| `C` | smazat všechny boxy ve framu |
| `D` / `→` | další frame |
| `A` / `←` | předchozí frame |
| `X` | **frame je prázdný** → uložit jako pozadí a jít dál |
| `TAB` | skočit na další nezkontrolovaný frame |
| **kolečko** | zoom pod kurzorem |
| `+` / `-` | zoom |
| `F` | přizpůsobit oknu |
| prostřední tlačítko / `SHIFT`+tažení | posun obrazu |
| `CTRL+Z` | zpět (40 kroků) |
| `S` | uložit |
| `Q` / `ESC` | konec (uloží) |

**Šedě přerušovaně** = předanotace, kterou jsi ještě nepotvrdil. Jakmile s boxem
cokoli uděláš (posuneš, změníš třídu), přebarví se na barvu třídy.

---

## Na co si dát pozor

### Zoom je nutnost

Medián předanotovaného boxu je **106 × 124 px** na framu **3416 × 1920**, tedy
**3,1 % šířky**. Nejmenší je 43 × 50 px. Bez přiblížení se anotovat nedá —
proto je zoom kolečkem hlavní ovládací prvek. Nad 2× se přepíná na `NEAREST`
interpolaci, aby byly vidět jednotlivé pixely.

### Prázdné framy anotuj klávesou `X`, ne přeskakuj

Frame bez zbraně je **platné trénovací pozadí** a je to hodnotná informace —
učí model, co zbraň není. Klávesa `X` uloží prázdný label a označí frame jako
zkontrolovaný.

Nástroj rozlišuje „zkontrolováno a nic tam není" od „ještě nezkontrolováno"
pomocí `progress.json`. Prázdný `.txt` sám by tyto dva stavy nerozlišil.

### Tyče a hole anotuj jako `melee`, nezakládej novou třídu

Chybějící „tyč/hůl" je hlavní funkční mezera modelu (§6.3). Je svůdné na ni
udělat čtvrtou třídu, ale **nedělej to**:

- Nová třída vyžaduje změnu detekční hlavy (`nc=3` → `4`) a model by se ji
  učil od nuly, místo aby jen rozšířil to, co už zná.
- Downstream ARMED logika pracuje s množinou labelů — čtvrtá třída znamená
  úpravy i tam.
- Semanticky tyč do `melee` patří: je to tupá zbraň držená v ruce.

Cílem je naučit model, že `melee` zahrnuje i **dlouhé tenké tyče**, ne jen
baseballové pálky. To je přesně ta chybějící část tvarového spektra.

### Kolik anotovat

Odhad z reportu je **200–300 framů**. Máš připraveno 208, což na spodní hranici
stačí. Prioritně:

1. **Video 004** (82 framů) — hromadná bitka, několik osob s tyčemi.
   **Nejcennější materiál**, tady je většina skutečných zbraní.
2. Videa 001 (120) a 002 (6) — převážně pozadí, ale právě proto užitečná
   pro potlačení false positives (model tam hlásí zbraně na předloktích).

Kdyby to nestačilo, dá se z videa 004 vyextrahovat víc framů — při 2 FPS jich
je 82, při 5 FPS by jich bylo ~205. Sousední framy jsou si ale podobné, takže
hodnota klesá.

---

## Po anotaci: dotrénování

```bash
# 1. Sestavit dataset z anotací (train/val split)
#    Framy zkopírovat na Spark do ~/weapon_training/

# 2. Dotrénovat existující model s NÍZKÝM learning rate.
#    lr0=0.0005, protože jde o domain adaptation, ne nový trénink —
#    model má zbraně už naučené a jen se adaptuje na novou doménu.
python train_weapon_yolo.py \
    --data <dataset s MED framy>/data.yaml \
    --model runs/detect/runs/weapons/weapons_sgd_lr005/weights/best.pt \
    --imgsz 960 --epochs 30 --batch 16 --workers 6 \
    --optimizer SGD --lr0 0.0005 --warmup-epochs 1 --patience 10
```

Odhad výpočtu: **1–2 h** na Sparku.

**Důležité pro vyhodnocení:** nový model se musí porovnat na **původním
validačním splitu** (aby se zjistilo, jestli se nezhoršil na zbraních obecně)
**i na MED framech** (jestli se zlepšil na cílové doméně). Nejlepší je
vyčlenit ~20 % anotovaných MED framů jako testovací a netrénovat na nich —
jinak nebude co měřit.

Pozor na **catastrophic forgetting**: při dotrénování na malém datasetu
z jedné domény může model zapomenout, co umí. Proto nízký lr a málo epoch.
Pokud by mAP na původní validaci výrazně klesla, řešením je smíchat MED framy
s částí původního datasetu (např. 1 : 3) místo trénování jen na MED.

---

## Použité skripty

| Skript | Účel |
|---|---|
| `prelabel_frames.py` | Vygeneruje předanotace modelem (běží na stroji s GPU) |
| `annotate_boxes.py` | GUI pro korekci (běží lokálně, bez GPU a bez cv2) |
