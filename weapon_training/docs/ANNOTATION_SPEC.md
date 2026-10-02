# Anotační specifikace — detekce zbraní v MED framech

Závazná pravidla pro anotaci. Cílem je **konzistence**: různí anotátoři musí
u stejného objektu nakreslit stejný box, jinak se model učí ze šumu.

---

## Třídy

| ID | Název | Co do ní patří |
|---|---|---|
| 0 | `gun` | Pistole, revolvery, pušky, brokovnice, samopaly |
| 1 | `knife` | Nože, dýky, řezáky, sekáčky, mačety |
| 2 | `melee` | **Tyče, hole, kůly, latě**, baseballové pálky, sekery, meče, kopí |

**Tyče a hole patří do `melee`.** Nezakládej čtvrtou třídu — cílem je rozšířit
`melee` o dlouhé tenké tyče, které v trénovacích datech chybí.

---

## Co anotovat a co ne

**ANO:**
- Zbraň nebo tyč držená v ruce
- Zbraň ležící na zemi nebo odložená
- Zbraň částečně zakrytá, pokud je jasně rozpoznatelná

**NE:**
- **Holá paže nebo předloktí.** Tohle je nejčastější false positive: obnažené
  předloktí je při nízkém rozlišení podlouhlé, světlé a zužuje se — vypadá
  jako tyč.
- Stín tyče na zemi
- Noha, obuv, pásek, popruh brašny
- Cokoli, u čeho si nejsi jistý → označ jako `ambiguous`, neanotuj

### Jak rozlišit tyč od paže

| Znak | Tyč | Paže |
|---|---|---|
| Šířka | **konstantní** po celé délce | zužuje se od ramene k ruce |
| Napojení | odděluje se od ruky, pokračuje za ni | vychází z ramene |
| Barva | často jiná než kůže (bílá, tmavá, dřevo) | odstín kůže, plynulý přechod |
| Konec | ostrý, rovný konec | ruka / prsty |

Když jedna jasná diagonála obsahuje **obojí** (paže → ruka → tyč), anotuj
**jen tu část za rukou**. Bod zlomu bývá vidět jako změna šířky nebo barvy.

**Pokud nelze rozhodnout, je to `ambiguous`.** Nehádej. Špatný label je horší
než žádný, protože učí model detekovat paže.

### Jak rozlišit tyč od jejího stínu

Druhá nejčastější past (po pažích). Za slunečného dne vrhá tyč na zem tmavý
podlouhlý stín, který má stejný tvar jako tyč sama.

| Znak | Skutečná tyč | Stín tyče |
|---|---|---|
| Hrana | často **světlý odlesk** po délce | nikdy žádný odlesk |
| Jas | světlejší i tmavší než okolí | vždy **tmavší** než okolí |
| Ohraničení | ostré | měkké, rozostřené |
| Napojení | na ruku osoby | na **stínovou masu** těla |
| Směr | libovolný | shodný se **směrem stínů** ostatních objektů |

Postup: najdi si ve framu směr stínů (podle lidí) a zkontroluj, jestli
podezřelý pruh nevychází ze stínu člověka ve stejném směru. Ve videu 004
padají stíny **dolů a vpravo**.

### Statické artefakty pozadí

Namalované linie na asfaltu, praskliny a odřený povrch se opakují na
**identických souřadnicích napříč framy**. Zbraň držená v ruce se v surveillance
záběru nemůže nehýbat, takže opakovaná detekce na stejném místě je artefakt.

Ověřené případy:

| Video | Souřadnice | Co to je |
|---|---|---|
| 004 | `(3100–3210, 730–920)` | světlá namalovaná čára na asfaltu, model ji hlásí jako `melee` |
| 004 | `(805–845, 897–948)` | prasklina / spára v betonu |
| 004 | `(890–955, 765–825)` | odložená taška, přítomná od framu 0009 dál |
| 001 | `(2750–2950, 535–600)` | hrana stínu / prasklina, ve framech 0000, 0048, 0112 |
| 001 | `(2455–2565, 545–575)` | tmavá skvrna na hranici světla a stínu |
| 001 | `(2795–2885, 550–580)` | tmavá skvrna na hranici světla a stínu |
| 001 | `(1240–1370, 465–512)` | tmavá skvrna na hranici světla a stínu |
| 001 | `(1440–1620, 730–880)` | odložený batoh, framy ~0067–0115 |

Ověřeno na framu 001 `0115`, kde je scéna téměř prázdná a artefakty tam přesto jsou.

**Směr stínů je v obou videích převážně dolů a mírně doprava.** Ověřeno
vizuálně: video 001 frame `0062` (stín odloženého batohu), video 004 frame
`0065` (stín chodce — nohy u `(1620, 950)`, konec stínu u `(1790, 1090)`).

**Ale neber směr jako konstantu přes celý obraz.** Kamera je nakloněná a snímá
shora, takže stíny, které jsou v realitě rovnoběžné, se v obraze perspektivou
sbíhají — zdánlivý směr se proto liší podle místa ve framu. Dva anotátoři
nezávisle hlásili odchylky (jednou „dolů-vlevo" ve videu 001, jednou „přímo
dolů" v části videa 004).

Praktický postup: **směr si ověř lokálně** na nejbližší osobě nebo kompaktním
objektu se zřetelným stínem, ne podle pravidla pro celý frame. Spolehlivější
než směr je hledat **dvojici téměř rovnoběžných útvarů** (objekt + jeho stín)
a křížové ověření na sousedním framu.

Kontrolu lze zautomatizovat:
```bash
python find_static_fp.py --dir annotate_med --video 004 --min-repeats 2
```
Pozor: statická detekce **není** automaticky FP — zbraň odložená na zemi je
také statická. Vždy zkontroluj vizuálně, jestli tam objekt je.

---

## Jak kreslit box

- **Těsně** kolem objektu — box se má dotýkat krajních pixelů objektu.
- U šikmé tyče kresli **osově zarovnaný box opsaný celé tyči** (bude obsahovat
  i pozadí v rozích, to je správně, YOLO jiný formát nemá).
- Anotuj **jen viditelnou část.** Zakrytou část nedomýšlej.
- Minimální rozumná velikost je ~15 × 15 px. Menší objekt neanotuj.

Pro kalibraci: model má tendenci kreslit boxy **příliš velké**. U jednoho
ověřeného případu měla tyč 40 × 68 px, ale model ji ohraničil 100 × 145 px.
Tvůj box má být těsný, ne jako od modelu.

---

## Postup (coarse-to-fine)

Bez přiblížení je odhad boxu nepřesný (naměřeno IoU 0,70). S mřížkou a zoomem
se souřadnice **odečítají**, ne hádají — přesnost ±5 px.

```bash
cd weapon_training

# 1) Celý frame s mřížkou a předanotacemi → najít objekty
python assist_annotate.py overview --dir annotate_med --stem <STEM> \
       --show-prelabels --out /tmp/ov_<STEM>.png

# 2) Přiblížený výřez → odečíst přesné souřadnice
python assist_annotate.py crop --dir annotate_med --stem <STEM> \
       --at <X>,<Y> --size 300 --zoom 4 --out /tmp/cr_<STEM>.png

# při nejasnosti zvětši výřez a uvidíš kontext (celou osobu)
python assist_annotate.py crop --dir annotate_med --stem <STEM> \
       --at <X>,<Y> --size 480 --zoom 2.5 --out /tmp/cr2_<STEM>.png
```

V obrázku jsou **žlutá čísla = X**, **fialová = Y**, obojí v souřadnicích
originálu (3416 × 1920). Souřadnice čti z mřížky, neodhaduj z poměrů.

---

## Výstup

Nepiš labely přímo. Vrať JSON, aby šel výsledek zkontrolovat:

```json
[
 {
  "stem": "004_004_0065",
  "prelabel_verdicts": [
    {"box": [2198,619,2299,764], "name": "melee",
     "verdict": "retighten", "new_box": [2226,628,2266,696],
     "note": "spravna detekce tyce, ale box 2.5x prilis velky"},
    {"box": [862,1115,933,1171], "name": "gun",
     "verdict": "reject", "note": "false positive na zemi, zadny objekt"}
  ],
  "missed": [
    {"approx_box": [1118,620,1163,708], "name": "melee", "confidence": "medium",
     "note": "bila tyc v ruce osoby v mentolove kosili; prechod paze/tyc kolem y=700"}
  ],
  "ambiguous": [
    {"approx_box": [1750,800,1800,830],
     "note": "neco v ruce osoby v modrem, nelze rozhodnout tyc vs paze"}
  ],
  "empty": false
 }
]
```

Hodnoty `verdict`: `accept` (box je dobrý), `retighten` (objekt tam je, ale box
je špatně — vyplň `new_box`), `reject` (false positive).

Hodnoty `confidence`: `high` / `medium` / `low`.

`empty: true` znamená, že ve framu není žádná zbraň — to je **platná a cenná
informace** (trénovací pozadí), ne chybějící práce.
