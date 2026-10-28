# Revize kódu pipeline (2026-10-11)

Revize `abandoned_object_detector.py` (AOD) a `suspicious_detector.py` (SD). Kód se při ní jen četl,
nic se neopravovalo. Nález 1 je navíc ověřený jednotkovým testem.

**Stav oprav:**
- **Opraveno:** 2 (`--conf-weapon` je teď volitelný jednotný práh), 3 (COCO `knife` od 0,50),
  6 (souřadnice v JSON s alerty) a nápověda `--abandon-sec` z bodu 12.
- **Opraveno (větev `fix-tracker-rematch`, mergnuto):** 1 a 10. CAVIAR zůstává beze změny
  (27 scénářů, 4 TP / 23 TN, zpoždění 6,4 s), protože anotace CAVIAR nemají výpadky a chybu
  neodhalí. Na MED s osobami z YOLO11x-pose klesl počet ID osob z 2 649 na 1 474. ARMED při
  nasazeném nastavení: 5 poplachů na rvačkách místo 4, bitka 93 % beze změny.
- **Opraveno (2026-10-28, Windows PC):** 4, 5, 7, 8, 9 a zbytek 12.
  - 4: párování osob i objektů je Hungarian (`match_tracks`) s gatingem. Hungarian sám míjení
    neřeší (dvě blízké osoby je „levnější“ prohodit), proto `PersonTracker` páruje podle
    předpovězené polohy (konstantní rychlost, `use_motion=True`). Jednotkový test: dvě osoby jdoucí
    proti sobě si ID dřív prohodily, teď ne.
  - 5: `--max-person-dist` je v px vstupního videa, s `--upscale` se přepočítá do prostoru detekce.
  - 7: vizualizace (SD i `visualization_helper.py`) volí spoje podle počtu bodů (14 = CrowdPose).
  - 8: `_wrist_points` bere u 17bodového skeletu COCO indexy (9, 10).
  - 9: pass 2 bere rozměr z pass 1 a video otevírá znovu, bez `seek` na 0.
  - 12: odstraněna nepoužitá `DEFAULT_MIN_OBJ_TRACK_LEN`.
  - Dopad: CAVIAR beze změny (4/4, 6,4 s). ARMED nasazené nastavení beze změny (5 poplachů na
    rvačkách, 5 v klidu, bitka 93 %); varianta bez COCO pálky drží bitku 005 z 90 % místo 62 %.
    `armed_eval/results.json` přepočítán (předchozí verze byla ještě před opravou bodu 1).
- **Otevřené:** 11 (designová otázka).

## Vysoká závažnost

**1. Tracky se po jediném vynechaném framu už znovu nespárují (ověřeno testem).**
AOD `PersonTracker.update` a `ObjectTracker.update`.

Nejdřív se všem trackům zvýší `missed` o 1 a pak se přeskočí každý track s `missed > 1`. Track,
který v předchozím framu chyběl, má teď `missed == 2` a už se nikdy nespáruje. `max_missed` tak
jen odkládá smazání mrtvého tracku. Test: osoba viděná ve framech 0, 1, 3 a 4 dostane ID 0, 0, –, 1, 1.

Důsledky:
- **Opuštěné zavazadlo:** když detektor tašku na jeden frame vynechá, vznikne nový track bez
  `owner_history`. `check_abandonment` ho přeskočí a alert nepřijde.
- **ARMED:** když osoba vypadne z detekce na jeden frame, nový track ztratí `armed`, `armed_hold`
  i `armed_count`. Hystereze 8 framů tak v praxi platí, jen dokud je osoba detekovaná souvisle.
- Simulace ARMED v `weapon_training/eval_armed_pipeline.py` používá stejný kód, takže tohle
  chování obsahuje.

Oprava: `if t['missed'] > self.max_missed: continue`, případně podmínku úplně vypustit, protože
mrtvé tracky se stejně mažou. **Změní ale chování, na kterém byly vyladěné parametry na CAVIAR**,
takže je potřeba je přeladit. Po opravě rozhodnout i bod 10.

## Střední závažnost

**2. `--conf-weapon` se pro gun, knife i melee tiše ignoruje.** SD `detect()` filtruje
`WEAPON_CONF_PER_CLASS.get(label, conf_weapon)` a všechny tři třídy v tabulce jsou. Nápověda CLI
(výchozí 0,55) je proto zavádějící. Oprava: parametr odstranit, nebo z něj udělat globální
spodní mez.

**3. COCO `knife` jde do ARMED už od conf 0,15.** `COCO_CONF_PER_CLASS` obsahuje jen
`baseball bat`. Ve 14 minutách testovacích videí se COCO nůž neobjevil ani jednou, z hlediska
logiky je to ale stejná díra jako dřív u pálky (telefon v ruce → ARMED). Oprava: přidat
`'knife': 0.5`, nebo COCO nůž z ARMED vyřadit, protože nože pokrývá vlastní model.

**4. Párování tracků je hladové podle pořadí tracků**, ne podle globálně nejbližší dvojice. Když
se dvě osoby míjejí, mohou si prohodit ID a s ním i stav ARMED nebo vlastnictví zavazadla.
Oprava: Hungarian (`scipy.optimize.linear_sum_assignment`) s gatingem.

**5. `max_person_dist = 100` je v pixelech detekčního prostoru.** S `--upscale` je frame 4×
větší, takže stejný parametr se chová 4× jinak. Oprava: vztáhnout k výšce boxu.

**6. JSON s alerty má s `--upscale` boxy v upscalovaných souřadnicích.** Snapshoty se škálují
správně, alerty ne. Oprava: `a['box'] = (np.array(a['box']) * scale).tolist()`.

## Nízká závažnost

7. Vizualizace v SD kreslí spoje skeletu pro COCO-17, ale ViTPose vrací 14 bodů CrowdPose, takže
   spoje jsou nesmyslné. Týká se jen obrázků.
8. `_wrist_points` předpokládá indexy CrowdPose (4, 5). S 17bodovým configem by vzal ucho a
   rameno. Teď to nevadí, protože se používá 14bodový config.
9. Pass 2 se vrací na začátek přes `cap.set(POS_FRAMES, 0)`, což je u `.mpg` nepřesné. Navíc chybí
   kontrola `ret`.
10. Po opravě bodu 1: když osoba chybí, její čítače ARMED se zmrazí. Sekvence zbraň, výpadek,
    zbraň by tak dala ARMED, i když framy nejdou po sobě.
11. Jedna zbraň může označit jako ARMED víc osob v davu. Je to spíš designová otázka.
12. Nepoužitá konstanta `DEFAULT_MIN_OBJ_TRACK_LEN`. Nápověda `--abandon-sec` uvádí výchozí
    hodnotu 5, skutečná je 2.

## Ověřeno jako v pořádku

- Sémantika ARMED „N framů po sobě, pak hold 8“.
- Škálování boxů a skeletů mezi upscalovaným a originálním framem (kromě bodu 6).
- Převod na sekundy přes `out_fps`.
- Pořadí skeletů vůči boxům.
- Ošetření prázdných výstupů.
- Interpolace v pass 2.
