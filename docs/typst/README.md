# Kapitoly v Typstu pro projekt na typst.app

Soubory jsou psané pro šablonu z projektu „Bakalářka (copy)“ (`main.typ` + `template.typ`,
bibliografie Hayagriva v `reference.yaml`, `#set heading(numbering: "1.")`, citace `@klic`).

| soubor | obsah | stav |
|---|---|---|
| `kapitola_chovani.typ` | Klasifikace chování osob a davu (úlohy 1–3 ze zadání: MED dav, RWF rvačky, anomálie póz) | koncept; sekce o pózách čeká na výsledky (`// TODO`) |
| `kapitola_zbrane.typ` | Detekce zbraní a nebezpečných předmětů | převod z `weapon_training/docs/KAPITOLA_DETEKCE_ZBRANI.md` skriptem `md2typ.py` |
| `reference_chovani.yaml` | bibliografie ke kapitole o chování | sloučit do `reference.yaml` |
| `figures/` | všechny PNG obrázky obou kapitol | nahrát do projektu do složky `figures/` |
| `md2typ.py` | převod Markdown → Typst (nadpisy, tabulky, obrázky, zvýraznění) | pro další kapitoly psané v MD |

## Vložení do projektu

1. Nahrát `kapitola_*.typ`, `reference_chovani.yaml` a složku `figures/` do projektu na typst.app.
2. Do `main.typ` za `#set heading(numbering: "1.")` přidat:
   ```typst
   #include "kapitola_chovani.typ"
   #include "kapitola_zbrane.typ"
   ```
3. Obsah `reference_chovani.yaml` připojit na konec `reference.yaml` a na konci dokumentu mít
   `#bibliography("reference.yaml", title: "Použitá literatura")`.
4. Projít `@TODO-citace` / `@TODO-odkaz` (hledat `TODO`) a nahradit klíči z bibliografie
   nebo `@label` kapitol (`<kap-detekce>`, `<kap-zbrane>` jsou očekávané labely kapitol o pipeline
   a o zbraních: přidat za nadpis, např. `= Detekce zbraní ... <kap-zbrane>`).
5. V `kapitola_zbrane.typ` zůstaly textové odkazy „tabulka X.6“, „obrázek X.4“: nahradit `@label`
   (labely tabulkám a obrázkům doplnit). Komentáře `// Poznámky pro autora` na konci souboru jsou
   pracovní a před odevzdáním se smažou.

## Co změnit v šabloně pro diplomovou práci

Šablona je z bakalářské práce (2024). Pro diplomovou práci upravit v `main.typ`:

- titulní strana: „DIPLOMOVÁ PRÁCE“ místo „BAKALÁŘSKÁ PRÁCE“, rok, název práce
  („Analýza pohybu osob ve videu s využitím neuronových sítí pro prostorově-temporální zpracování“);
- `#image("KvapilN-1-1.png")` / `-2.png` = naskenované zadání, nahradit zadáním DP;
- prohlášení: odkaz na směrnici univerzity. Text z roku 2024 cituje směrnici č. 7/2019; podle
  webu univerzity ji nahradila **směrnice č. 7/2025** „Pravidla pro odevzdávání, zpřístupňování
  závěrečných prací veřejnosti a jejich formální úpravu“ (studenti.upce.cz). Aktuální znění
  prohlášení a formální požadavky převzít z ní; PDF se mi automaticky stáhnout nepodařilo,
  je potřeba otevřít ručně;
- poděkování, anotace, klíčová slova (CZ/EN), seznam zkratek.

Formát (A4, okraje 3,5 / 1,5 / 2,5 cm, 12 pt, číslování stránek od úvodu) zůstává.
