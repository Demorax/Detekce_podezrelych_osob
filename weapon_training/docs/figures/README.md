# Obrázky do práce

Generuje `python make_thesis_figures.py --device mps` (z adresáře `weapon_training/`).
Vše se počítá znovu z vah a anotací, čísla jsou v `figures_data.json`. Výjimkou je AP50 na
původní webové validaci (0,830 / 0,796 / 0,807), které je převzato z TRAINING_REPORT §5.1 a §6d,
protože 24k dataset je jen na Sparku. Každý obrázek je ve verzi PDF (sazba) a PNG (náhled).

Barvy modelů jsou ve všech obrázcích stejné: výchozí modrá, v1 oranžová (čárkovaně) a v2 zelená.
Paleta prošla kontrolou rozlišitelnosti pro barvoslepé. Křivky se navíc liší typem čáry
a mají popisky, takže obrázky fungují i v černobílém tisku.

| Soubor | Navržený popisek |
|---|---|
| `fig1_trenink` | **Průběh tréninku na webových datech.** Výchozí `optimizer='auto'` zvolil MuSGD s lr 0,01, který během warmupu divergoval (mAP50 0,30 → 0,10, validační klasifikační loss 2,5×). Explicitní SGD s lr0 0,005 konverguje k mAP50 0,78. Propad SGD ve 3. epoše je vrchol warmupu learning rate, ze kterého se model zotavil. (§4.2) |
| `fig2_porovnani` | **Zobecnění na CCTV vs. ztráta na webových datech.** Doménová adaptace zvýšila AP50 třídy melee na nezávislé bitce z videa 005 téměř trojnásobně (0,219 → 0,612). Na původní webové validaci přitom ztratila jen 0,02–0,03. (§6c, §6d) |
| `fig3_pr_krivky` | **PR křivky třídy melee na testu 005** (48 framů, 67 ručně ověřených tyčí; na videu 005 se netrénovalo). Výchozí model drží precision nad 0,5 jen do recallu ~0,2. Dotrénované modely ji drží do ~0,6. Žádný model nepřekročí recall 0,75: zbylé tyče jsou rozmazané nebo zakryté a model je nevidí ani s nízkou jistotou. (§6c) |
| `fig4_falesne_poplachy` | **Falešné poplachy na 24 testovacích framech rvaček bez zbraní** v závislosti na prahu. Přidání 60 negativních framů (v2) snížilo počet falešných poplachů zhruba na polovinu při každém prahu. Při nasazeném prahu 0,20 jsou to 2 poplachy proti 5 u v1. (§6d) |
| `fig5_kalibrace` | **Kalibrace prahu melee pro model v2.** Na kalibrační polovině testu 005 je F1 mezi prahem 0,10 a 0,20 plochá. Zvolen byl nejnižší práh plošiny s precision 1,00 (0,20). Na ověřovací polovině tím recall stoupl z 0,17 (práh 0,40) na 0,31 a precision zůstala 1,00. (§6e) |
| `fig6_ukazky` | **Ukázky výstupu modelu v2 při prahu 0,20.** a) obě tyče zachyceny; b) tyč zvednutá nad hlavu splývá s paží a model ji přehlédne; c) falešný poplach na rvačce bez zbraní: natažená paže s předmětem v ruce. Výběr je automatický: frame s nejvíc správnými detekcemi, frame s nejvíc přehlédnutími a falešný poplach s nejvyšší jistotou. |
