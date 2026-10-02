# Vizuální kontrola obsahu tříd

Náhledy z `inspect_class.py` — vzorky výřezů boxů dané třídy. Sloužily k rozhodnutí,
jak zdrojovou třídu namapovat (viz `../../DATASETS.md` §2.1 a §4.1).

| Soubor | Co ukazuje | Rozhodnutí |
|---|---|---|
| `weapon_sample.jpg` | třída `weapon` z datasetu cabwp (48 výřezů) | směs pistolí, pušek a nožů → dataset vyřazen celý |
| `sample_hw.jpg` | třída `heavy-weapon` z m7qso | jen dlouhé palné zbraně → `gun` |
| `sample_pizme.jpg` | nepojmenovaná třída `0` z dataset-pizme | baseballové pálky → `melee` |
| `sample_bat.jpg` | třída `Bat` z baseball-v1 | baseballové pálky (grayscale export) → `melee` |
| `merged_melee.jpg` | sloučená třída `melee` po prvním mapování | odhalil dýky ze `short sword` → přesunuty do `knife` |
| `merged_knife.jpg`, `merged_gun.jpg` | sloučené třídy po finálním mapování | kontrola správnosti remapu |
