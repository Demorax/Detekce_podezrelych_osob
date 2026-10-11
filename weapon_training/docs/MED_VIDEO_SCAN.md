# Průchod všemi MED videi — kde jsou zbraně

**Datum:** 2026-10-11. **Účel:** najít další scény se zbraněmi kromě bitky ve videu 004,
aby šlo ověřit, že dotrénovaný model zobecní mimo jednu událost.

## Metoda

Pro každé z 28 dosud neprohlédnutých videí (001, 002, 004 už byla anotována) vznikl
kontaktní list 16 framů z 2 FPS extrakce — 8 rovnoměrně a 8 v místech největší
meziframové změny. Kandidátní okna se pak prohlédla v plném rozlišení.

**Metrika pohybu bitku neodliší.** Ve videu 004 má klidná fáze průměrnou meziframovou
změnu 11,5 a bitka 11,0 — při odstupu 0,5 s se lidé hýbou pořád. Výběr „pohybových"
framů proto sloužil jen jako doplněk; rozhodovalo vizuální posouzení všech listů.

## Výsledek

| Video | Obsah | Zbraně |
|---|---|---|
| 004 | bitka s tyčemi (už anotováno) | ✅ |
| **005** | **bitka s bílými tyčemi, latěmi a červenými pruty, 2 FPS framy ~0060–0072** | **✅** |
| 003, 006, 008 | panika, lidé utíkají | ne |
| 007 | rvačka, pohyblivá kamera, silné rozmazání | nevidím |
| 009 | inscenované napadení zezadu (ověřeno zblízka) | nevidím |
| 012, 014, 016, 017 | rvačky pěstmi | ne |
| 018, 019, 020 | shluk kolem osoby na zemi | ne |
| 010, 011 | dýmovnice / petarda | ne |
| 025, 026 | pochod s bílými papíry zdviženými nad hlavou | ne |
| 013, 015, 021–024, 027–031 | procházení, dav, motorka | ne |

**Jediné další video se zbraněmi je 005.** Stejný dvůr a kamera jako 004, ale jiná
událost — proto slouží jako **nezávislý testovací set** (`annotate_005/`), na kterém
se netrénuje.

**Užitečná negativa pro další trénink** (zatím nevyužitá):
- rvačky bez zbraní (012, 014, 016, 017) — přesně situace, kdy model hlásí předloktí,
- zdvižené bílé papíry (025, 026) — vizuálně blízko bílé tyči nad hlavou.
