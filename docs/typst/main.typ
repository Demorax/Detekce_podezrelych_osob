// Kopie main.typ z projektu na typst.app (stav 2026-10-28). Zdroj pravdy je projekt na typst.app;
// tady jen pro přehled a zálohu. Kapitoly: typst/kapitola_*.typ (v projektu ve složce typst/).
#import "template.typ": *

// TODO: položky označené TODO doplnit (zadání, rok, poděkování, anotace, zkratky, kapitoly).
#let todo(x) = text(fill: red, weight: "bold")[[TODO: #x]]

#let parLeading = 1em

#set text(
  lang: "cs",
  region: "cz",
  size: 12pt,
  hyphenate: false,
  font: "Nimbus Roman No9 L"
)
#set page(
  paper: "a4",
  margin: (y: 2.5cm, left: 3.5cm, right: 1.5cm),
  binding: left
)
#set par(
  justify: true,
  leading: parLeading,
  linebreaks: "simple"
)

#show par: set block(below: parLeading + 8pt)
#show heading: set block(below: parLeading, above: 12pt)
#show heading.where(level: 1): set text(size: 16pt)
#show heading.where(level: 2): set text(size: 14pt)
#show heading.where(level: 3): set text(size: 13pt)

#[
  #set text(size: 22pt)

  #align(center)[
    UNIVERZITA PARDUBICE \
    FAKULTA ELEKTROTECHNIKY \
    A INFORMATIKY
  ]

  #place(center + horizon, "DIPLOMOVÁ PRÁCE")

  #place(bottom + left, "2026")
  #place(bottom + right, "Tomáš Kvapil")
]

#pagebreak()

#[
  #set text(size: 14pt)

  #align(center)[
    Univerzita Pardubice \
    Fakulta elektrotechniky a informatiky
  ]

  #place(center + horizon)[
    Analýza pohybu osob ve videu s využitím neuronových sítí pro prostorově-temporální zpracování \
    Diplomová práce
  ]
  #place(bottom + center, [Tomáš Kvapil \ 2026])
]

#pagebreak()
// TODO: naskenované zadání diplomové práce (2 strany), např. #image("zadani-1.png")
#todo[zadání práce]
#pagebreak()

Prohlašuji:

Práci s názvem Analýza pohybu osob ve videu s využitím neuronových sítí pro prostorově-temporální zpracování jsem vypracoval samostatně. Veškeré literární prameny a informace, které jsem v práci využil, jsou uvedeny v seznamu použité literatury.

Byl jsem seznámen s tím, že se na moji práci vztahují práva a povinnosti vyplývající ze zákona č. 121/2000 Sb., o právu autorském, o právech souvisejících s právem autorským a o změně některých zákonů (autorský zákon), ve znění pozdějších předpisů, zejména se skutečností, že Univerzita Pardubice má právo na uzavření licenční smlouvy o užití této práce jako školního díla podle § 60 odst. 1 autorského zákona, a s tím, že pokud dojde k užití této práce mnou nebo bude poskytnuta licence o užití jinému subjektu, je Univerzita Pardubice oprávněna ode mne požadovat přiměřený příspěvek na úhradu nákladů, které na vytvoření díla vynaložila, a to podle okolností až do jejich skutečné výše.

Beru na vědomí, že v souladu s § 47b zákona č. 111/1998 Sb., o vysokých školách a o změně a doplnění dalších zákonů (zákon o vysokých školách), ve znění pozdějších předpisů, a směrnicí Univerzity Pardubice č. 7/2025 Pravidla pro odevzdávání, zpřístupňování závěrečných prací veřejnosti a jejich formální úpravu, ve znění pozdějších dodatků, bude práce zveřejněna prostřednictvím Digitální knihovny Univerzity Pardubice. #todo[ověřit aktuální znění prohlášení podle směrnice 7/2025]

#align(bottom)[
  V Pardubicích dne #todo[datum]
  #align(right)[
    Tomáš Kvapil
  ]
]

#pagebreak()

#place(bottom + left, [#text(size: 16pt, weight: "bold")[PODĚKOVÁNÍ]\ #todo[poděkování]])

#pagebreak()

= ANOTACE
#todo[anotace]

= KLÍČOVÁ SLOVA
#todo[klíčová slova]

= TITLE
Analysis of Human Motion in Video Using Neural Networks for Spatio-Temporal Processing

= ANNOTATION
#todo[annotation]

= KEYWORDS
#todo[keywords]

#pagebreak()

#outline(
  title: [OBSAH],
  indent: auto,
  target: heading.where().after(label("obsah"))
)<obsah>

#pagebreak()

= Seznam ilustrací a tabulek
#outline(
  title: none,
  target: figure.where(kind: table).or(figure.where(kind: image).or(figure.where(kind: raw)))
)

#pagebreak()

= Seznam zkratek a značek

AUC – Area Under the ROC Curve\
CNN – Convolutional Neural Network\
GRU – Gated Recurrent Unit\
LSTM – Long Short-Term Memory\
MED – Motion Emotion Dataset\
mAP – mean Average Precision\
YOLO – You Only Look Once\
#todo[doplnit další zkratky]

#pagebreak()

#set page(numbering: "1")
= Úvod
#todo[úvod]

#pagebreak()

#set heading(numbering: "1.")

// TODO: kapitoly rešerše, data, detekce osob a odhad pózy (label <kap-detekce>), pre-incident detekce
#include "typst/kapitola_chovani.typ"
#include "typst/kapitola_zbrane.typ"

#pagebreak()
#show heading.where(level: 1): it =>[
  #block(it.body)
]

= Závěr
#todo[závěr]

#pagebreak()
#bibliography(("reference.yaml", "typst/reference_chovani.yaml"), title: "Použitá literatura")
