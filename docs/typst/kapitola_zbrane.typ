= Detekce zbraní a nebezpečných předmětů

== Motivace a cíl

Analýza chování osob popsaná v předchozích kapitolách pracuje s pohybem odvozeným ze skeletu. Ten však o míře ohrožení často nerozhoduje: strkanice dvou osob může být hrou i napadením. Rozhodující informací bývá, zda některá z osob drží předmět, kterým lze způsobit zranění.

Cílem této části práce bylo doplnit do zpracovatelského řetězce detektor, který v jednotlivých snímcích z kamery vyhledá střelné zbraně, nože a tupé či sečné zbraně držené v ruce. Jeho výstup se kombinuje s detekcí osob a s odhadem jejich skeletu: osoba je označena jako ozbrojená (příznak ARMED) teprve tehdy, když se box detekované zbraně nachází u zápěstí některé z osob. Tato podmínka odpovídá zadání, protože zajímá zbraň v ruce, nikoli předmět ležící opodál, a zároveň odfiltruje část falešných detekcí v místech, kde žádná ruka není.

Cílovou doménou jsou záběry z Motion Emotion Datasetu (MED) @TODO-citace: statická bezpečnostní kamera snímající prostranství šikmo shora, davy lidí, malé objekty, pohybové rozmazání a proměnlivé osvětlení. Jak ukáže zbytek kapitoly, právě rozdíl mezi touto doménou a fotografiemi, z nichž vycházejí veřejně dostupné datasety zbraní, se stal ústředním problémem celé úlohy.

== Volba přístupu

=== Detektor a jeho velikost

Hotový detektor tupých zbraní v použitelné kvalitě není k dispozici, a cestou bylo proto dotrénování (fine-tuning) obecného detektoru objektů. Zvažovány byly tři varianty: model YOLO11 @TODO-citace ve velikosti _l_ při vstupním rozlišení 960 px, větší varianta _x_ při 640 px a trénink transformerového detektoru RT-DETR @TODO-citace od nuly.

Trénink transformeru od nuly byl zamítnut jako první: pro přibližně 24 tisíc obrázků, které se podařilo sestavit, by trval desítky hodin a s velkou pravděpodobností by prohrál s dotrénováním vah předtrénovaných na datasetu COCO @TODO-citace. Mezi zbylými variantami rozhodla povaha cílových dat. U zbraní v záběru bezpečnostní kamery není limitujícím faktorem kapacita sítě, ale velikost objektu, neboť tyč nebo pálka zabírá v MED snímku jen desítky pixelů. Vyšší vstupní rozlišení proto přináší při srovnatelném výpočetním čase víc než hlubší síť. Zvolen byl model YOLO11l se vstupem 960 px, inicializovaný vahami z COCO.

=== Tři třídy místo šesti

Původní plán počítal se šesti třídami: `gun`, `knife`, `melee`, `backpack`, `handbag` a `suitcase`. Zavazadel je ve veřejných datasetech málo (kabelek přibližně 200, kufrů přibližně 2 700 a batohů 819), a hlavně jsou všechny tři třídy součástí datasetu COCO a předtrénovaný detektor YOLO11x, který pipeline už používá, je rozpoznává na základě zhruba 118 000 obrázků. Dotrénování z několika set nebo tisíců instancí by dalo horší výsledek. Vlastní detektor proto řeší jen to, co COCO model neumí, tedy zbraně. Třída `gun` zahrnuje pistole, revolvery, pušky, brokovnice a samopaly, třída `knife` nože, dýky, řezáky a sekáčky a třída `melee` tupé a sečné zbraně držené v ruce, tedy pálky, sekery, meče, kopí a po doménové adaptaci popsané níže i tyče, hole a latě.

Pro tyče a hole nebyla založena samostatná třída, ačkoli se později ukázaly jako hlavní mezera modelu. Nová třída by vyžadovala změnu detekční hlavy a model by se ji učil od začátku, místo aby rozšířil pojem, který už zná. Sémanticky tyč do třídy `melee` patří, a navazující logika označování ozbrojených osob navíc pracuje s pevnou množinou tříd.

== Trénovací data

=== Zdroje a sestavení datasetu

Trénovací data pocházejí z veřejných datasetů platformy Roboflow Universe @TODO-citace. Ze sedmi stažených datasetů (3,3 GB) bylo použito šest. Nejkvalitnějším zdrojem byl dataset s vyváženými třídami baseballová pálka, nůž a pistole (1 240, 1 012 a 1 144 instancí) pod licencí CC BY 4.0; dalšími zdroji byly datasety s různými typy střelných a chladných zbraní, dva datasety baseballových pálek a menší dataset pistolí [CITACE pro každý dataset].

Zdrojové třídy (jen do třídy `gun` jich bylo namapováno přes dvacet) se převáděly na cílové podle názvu, u nejednoznačných tříd však na základě vizuální kontroly náhodného vzorku výřezů. Tak bylo ověřeno, že třída `heavy-weapon` (11 598 boxů) obsahuje výhradně dlouhé palné zbraně a že nepojmenovaná třída `0` v jednom z datasetů jsou baseballové pálky. Třída `short sword`, původně mapovaná na `melee`, naopak obsahovala dýky vizuálně nerozeznatelné od velkých nožů; jejích 501 instancí bylo přesunuto do třídy `knife`, což odstranilo soupeření dvou tříd o stejný vzhled a posílilo nejslabší třídu.

Sestavení vyžadovalo vyřešit tři metodické problémy. Pokud se z obrázku odstraní anotace nezmapovatelné zbraňové třídy a obrázek zůstane, stane se z něj pozadí a model se učí, že zbraň je pozadím; u nejednoznačných tříd (například obecné `weapon`) se proto zahazoval celý obrázek, u neškodných tříd (například `person`) jen label. Protože datasety často vznikají kopírováním jiných, může se tentýž obrázek objevit v trénovacím splitu jednoho a validačním splitu druhého zdroje; deduplikace podle hashe obsahu odstranila 5 049 duplicit (10,9 %), při kolizi s předností validačního splitu. Konečně dva exporty míchaly obdélníkové boxy s polygony (549 řádků), které byly převedeny na opsaný obdélník.

Z 46 173 kandidátních obrázků zbylo po deduplikaci 41 124 a po vyřazení obrázků s nezmapovanými labely 24 340. Rozdělení do splitů bylo převzato z původních datasetů, protože augmentované varianty téhož obrázku se v exportech vyskytují jen v trénovacím splitu a náhodné rozdělení by je rozptýlilo i do validace. Výsledný dataset má 19 924 trénovacích, 3 503 validačních a 913 testovacích obrázků s 28 156 instancemi, z toho 14 218 (50,5 %) `gun`, 10 730 (38,1 %) `melee` a 3 208 (11,4 %) `knife`.

=== Zjištěné vady veřejných datasetů

Tři datasety doporučené v rešerši jako zdroje tupých zbraní byly nepoužitelné z důvodů, které z popisu ani metadat nebyly patrné: názvy tříd prvního byly zbytky textu README, druhý neobsahoval žádné tupé zbraně ani exportovanou verzi a třetí obsahoval tři varianty téže třídy lišící se jen tečkou na konci názvu. Holá tyč, hůl nebo obušek jako samostatně anotovaná třída ve veřejně dostupných datech neexistuje.

Největší nalezený dataset (15 981 obrázků, 16 966 boxů) s jedinou třídou `weapon` obsahoval podle kontroly vzorku 48 výřezů směs pistolí, pušek a nožů. Namapování na `gun` by do této třídy vneslo nože a rozdělení by vyžadovalo kompletní přeanotování, proto byl vyřazen celý. U datasetů z otevřených platforem tedy nelze spoléhat na metadata (rozhraní platformy u některých vracelo jiný seznam tříd, než obsahoval skutečný export); třídy je nutné číst z exportované konfigurace a ověřit pohledem.

Některé vady sestaveného datasetu odstranit nešlo. Celkem 7 798 z 10 730 boxů třídy `melee` (72,7 %) pochází z baseballových fotografií, takže hrozí, že se model naučí kontext sportovní scény místo tvaru předmětu. Třída `knife` je vůči třídě `gun` zastoupena v poměru 1 : 4,4. A především jde výhradně o webové a sportovní fotografie se zbraněmi zblízka, v dobrém světle a z úrovně očí, tedy o opak cílové domény.

== Trénink

Trénink probíhal na stroji NVIDIA DGX Spark s GPU GB10 a 121 GB paměti sdílené mezi procesorem a grafickým akcelerátorem, s knihovnou Ultralytics @TODO-citace ve verzi 8.4.115; kvůli nižší propustnosti sdílené paměti model zpracoval při dávce 32 a rozlišení 960 px zhruba 14 obrázků za sekundu.

Finální konfigurace používala optimalizátor SGD s počáteční rychlostí učení 0,005 a kosinovým útlumem, dávku 32 a smíšenou přesnost. Augmentace (geometrické transformace, horizontální převrácení, mozaika, mixup a posuny barevného tónu, sytosti a jasu) byly nastaveny s ohledem na rozdíl domén. Trénink byl naplánován na 100 epoch s předčasným ukončením po 25 epochách bez zlepšení a zastavil se po 90 epochách a 43,4 hodinách. Nejlepší epocha 65 byla vybrána podle kombinované metriky knihovny (0,1 · mAP50 + 0,9 · mAP50-95), která upřednostňuje přesnou lokalizaci; nejvyšší mAP50 (0,7804) dosáhla epocha 76, rozdíl je však zanedbatelný.

=== Divergence optimalizátoru MuSGD

K finální konfiguraci se dospělo až po dvou neúspěšných bězích. Při výchozím nastavení `optimizer='auto'` zvolila použitá verze knihovny optimalizátor MuSGD založený na metodě Muon @TODO-citace s rychlostí učení 0,01. Během warmupu model divergoval: mAP50 klesla z 0,304 v první epoše na 0,0999 ve třetí a validační klasifikační ztráta vzrostla z 2,466 na 6,119. Současně rostla i trénovací ztráta, takže nešlo o šum ani o přeučení. Pravděpodobnou příčinou je, že metoda Muon počítá s výrazně menší efektivní rychlostí učení pro maticové parametry než klasický SGD. Explicitní volba SGD s rychlostí učení 0,005 problém okamžitě odstranila (mAP50 po první epoše 0,558 proti 0,304). Průběh obou běhů srovnává obrázek X.1. U této verze knihovny proto nelze doporučit spoléhat na automatickou volbu optimalizátoru.

#figure(
  image("figures/fig1_trenink.png", width: 90%),
  caption: [Průběh tréninku na webových datech. Výchozí `optimizer='auto'` zvolil MuSGD s rychlostí učení 0,01, který během warmupu divergoval (mAP50 0,30 → 0,10, validační klasifikační ztráta 2,5×). Explicitní SGD s lr0 0,005 konverguje k mAP50 0,78. Propad SGD ve 3. epoše je vrchol warmupu rychlosti učení, ze kterého se model zotavil.],
)

== Vyhodnocení na webových datech

Na validačním splitu (3 503 obrázků, 4 058 instancí) dosáhl model mAP50 0,775 a mAP50-95 0,523 při precision 0,878 a recallu 0,724 (tabulka X.1).


#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    table.header([Třída], [Precision], [Recall], [mAP50], [mAP50-95]),
    [`gun`], [0,878], [0,822], [0,884], [0,634],
    [`melee`], [0,871], [0,763], [0,830], [0,533],
    [`knife`], [0,884], [0,587], [0,611], [0,403],
    [celkem], [0,878], [0,724], [0,775], [0,523],
  ),
  caption: [Výsledky výchozího modelu na validačním splitu webových dat.],
)


Testovací split dává lepší čísla (mAP50 0,855, u třídy `knife` 0,955), jako hlavní výsledek se však neuvádí: obsahuje jen tři z pěti zdrojových datasetů a chybějí v něm zejména nože z nejobtížnějšího zdroje. Je tedy užším a snazším měřítkem.

Zřetelnou slabinou je třída `knife`. Precision 0,884 při recallu 0,587 znamená, že model přibližně 41 % nožů přehlédne, což je pro detektor hrozeb horší typ chyby. Obě zjevná řešení selhala. Oversampling, při němž byl každý trénovací obrázek s nožem zařazen třikrát (poměr `gun` : `knife` se změnil z 4,4 : 1 na 1,6 : 1), recall nezvýšil (0,587 → 0,585) a celkovou mAP50 mírně snížil (0,775 → 0,770). Snížení prahu jistoty z 0,55 na 0,12 zvedlo recall nožů jen z 0,574 na 0,621 za cenu poklesu precision o 0,125; chybějící nože tedy nejsou slabé detekce těsně pod prahem, model je nevidí vůbec. Příčinou není nevyváženost tříd, ale nedostatečná rozmanitost trénovacích dat, kterou mohou odstranit jen nová anotovaná data.

Prahy jistoty pro jednotlivé třídy byly odečteny z křivky F1 na validačním splitu (tabulka X.2). Zvolené hodnoty leží záměrně mírně pod maximem F1, protože požadavek zápěstí u boxu zbraně část falešných detekcí odfiltruje sám a posun k vyššímu recallu se proto vyplatí.


#figure(
  table(
    columns: 6,
    align: (left, right, right, right, left, left),
    table.header([Třída], [Práh při max. F1], [Max. F1], [Zvolený práh], [Precision], [Recall]),
    [`gun`], [0,411], [0,851], [0,35], [0,831], [0,852],
    [`knife`], [0,388], [0,709], [0,30], [~0,85], [~0,60],
    [`melee`], [0,494], [0,817], [0,40], [0,806], [0,792],
  ),
  caption: [Volba prahů jistoty z křivky F1 na validačním splitu (výchozí model).],
)


== Rozdíl domén: test na CCTV záběrech

Validační data mají stejné rozložení jako trénovací, a o cílové doméně proto nevypovídají. Model byl tedy spuštěn na 208 snímcích z videí 001, 002 a 004 datasetu MED, zpracovaných stejně jako v ostatních částech pipeline, s prahem jistoty 0,25.

Model vrátil jen 17 detekcí v 17 snímcích (8,2 %): 15 v třídě `melee` s průměrnou jistotou 0,408 a maximem 0,684, dvě v třídě `gun` a žádnou v třídě `knife`. Teprve při prohlížení snímků se ukázalo, že video 004 zachycuje hromadnou bitku, v níž lidé drží improvizované zbraně, především tyče; původní předpoklad, že MED zbraně neobsahuje, byl mylný.

Kvalitativní rozbor ukázal oba typy chyb. Model správně označil podlouhlý předmět v ruce osoby v davu, viděný shora, takže přenos do nové domény alespoň částečně funguje. Ve stejném videu však přehlédl dlouhou bílou tyč drženou přes celou šíři těla, nejnápadnější předmět ve snímku. Falešné detekce mířily na ruce a obnažená předloktí, která mají při tomto rozlišení podobný tvar i jas jako tyč.

Příčinu lze pojmenovat konkrétně. Třída `melee` byla naučena převážně na baseballových pálkách ze sportovních fotografií, tedy na krátkých kuželovitých předmětech, zatímco dlouhá tenká tyč v trénovacích datech chybí. K tomu přistupuje obecný rozdíl mezi fotografií zbraně zblízka a pohledem kamery shora, kde má předmět desítky pixelů a je rozmazaný a částečně zakrytý. Model trénovaný na webových datech tedy výstup nezaplavuje falešnými poplachy, ale většinu hrozeb nenajde.

== Doménová adaptace

Z testu vyplynulo, že nejúčinnějším krokem nebude další ladění na webových datech, ale dotrénování modelu na anotovaných snímcích přímo z cílové domény.

=== Postup anotace

Snímky mají po zvětšení rozměr 3416 × 1920 px a box zbraně zabírá jen několik procent jejich šířky, takže bez přiblížení nelze objekt spolehlivě rozpoznat ani ohraničit. Anotace proto probíhala s asistencí velkého jazykového modelu s multimodálním vstupem a jeho podřízených agentů @TODO-citace podle písemné anotační specifikace, která definuje třídy, pravidla kreslení boxů a postup v nejasných případech. Výstupy modelu nebyly přebírány automaticky; každý krok byl navržen tak, aby šel ověřit.

Postup měl tři fáze. Nejprve natrénovaný detektor vygeneroval předanotace s nízkým prahem jistoty 0,10, aby vzniklo co nejvíce kandidátů ke kontrole. Poté agenti každý snímek prošli od hrubého k jemnému: nejprve celý snímek se souřadnicovou mřížkou a předanotacemi, poté přiblížené výřezy kolem jednotlivých kandidátů. Mřížka se ukázala jako zásadní. Bez ní dosahoval box odhadnutý jazykovým modelem shody IoU s referencí jen 0,70, což je pro trénink příliš volné; s přiblíženým výřezem a mřížkou se souřadnice odečítají s přesností přibližně ±5 px. U tmavého asfaltu a stínů pomohlo percentilové roztažení kontrastu. Agenti nevraceli přímo labely, ale strukturované verdikty ke každé předanotaci (přijmout, zpřesnit box, zamítnout) spolu se seznamem přehlédnutých objektů a nejasných případů; nejasný případ se do dat nezapsal, ale byl odložen ke kontrole člověkem. Ve třetí fázi byly všechny boxy před zápisem prohlédnuty na kontaktních listech, tedy přehledových tabulích výřezů, na nichž lze vedle sebe porovnat desítky boxů, a sporné boxy byly vyloučeny.

Do specifikace byly průběžně doplněny tři opakující se pasti. První je záměna tyče s obnaženým předloktím: tyč má konstantní šířku, často jinou barvu než kůže a odděluje se od ruky, zatímco paže se zužuje od ramene k ruce. Druhou je záměna tyče s jejím stínem: skutečná tyč mívá po hraně světlý odlesk a stín se napojuje na stínovou masu těla, přičemž směr stínů se perspektivou nakloněné kamery napříč snímkem mění a je nutné jej ověřit lokálně. Třetí pastí jsou statické artefakty pozadí, například namalované čáry, praskliny v betonu nebo odložené tašky. Zbraň držená v ruce se nemůže dlouhodobě nehýbat, takže opakovaná detekce na stejných souřadnicích ukazuje na artefakt; specifikace jich dokumentuje jedenáct, osm v trénovacích videích a tři v testovacím videu 005.

Předanotace výchozího modelu byly málo přesné: ze 126 kandidátních boxů byl beze změny přijat jediný, 21 obsahovalo skutečnou tyč, ale s boxem 1,5–2,5krát větším, a 104 byly falešné detekce (předloktí, pěsti, stíny, brašny, boty, značky na asfaltu). Naopak 34 tyčí, které anotátoři našli, model nenavrhl vůbec.

Při přípravě dat se navíc ukázalo, že dříve zvětšené snímky MED nevznikly superrezolucí ESPCN, ale bikubickým zvětšením 4× (odchylka na úrovni šumu komprese JPEG); nové snímky byly proto zvětšeny stejně.

=== Anotovaná data

Řídká sada obsahuje 208 snímků vzorkovaných s frekvencí 2 snímky za sekundu z videí 001, 002 a 004; zbraň obsahovalo jen 10 z nich (16 boxů). Protože se zbraně objevují jen v krátkém okně bitky ve videu 004, byla doplněna hustá sada 44 snímků z tohoto okna vzorkovaných s frekvencí přibližně 6 snímků za sekundu, z toho 22 se zbraní (40 boxů). Celkem bylo anotováno 252 snímků s 56 boxy třídy `melee` a 220 negativy.

=== Nastavení dotrénování

Snímky husté sady jsou od sebe vzdáleny jen 0,17 s a sousední snímky jsou téměř totožné. Náhodné rozdělení by tak poslalo prakticky stejný snímek do tréninku i do validace a validace by měřila zapamatování. Data proto byla rozdělena časově: validaci tvoří souvislý blok snímků bitky ve videu 004 s ochrannou mezerou pěti snímků na každé straně, poslední čtvrtina videa 001 a celé video 002. Trénovací část má 209 snímků se 43 boxy, validační 41 snímků s 10 boxy; zbývající dva snímky se třemi boxy padly do ochranné mezery.

Snímky MED obsahují jen tyče, a model dotrénovaný pouze na nich by mohl ztratit schopnost rozpoznat ostatní třídy (katastrofické zapomínání @TODO-citace). Do trénovacích dat bylo proto přimícháno 800 obrázků z původního webového datasetu. Dotrénování vycházelo z nejlepších vah hlavního tréninku s nízkou rychlostí učení (SGD, lr0 0,0005, 30 epoch, dávka 16, 960 px). Dva běhy se liší pouze tím, zda je zmrazen backbone sítě (vrstvy 0–10), a tvoří tak čistou ablaci. Vyhodnocovaly se váhy z poslední epochy, protože při pouhých 10 validačních boxech by výběr epochy podle téže validace výsledek nadhodnotil.

=== Výsledky

Dotrénování výrazně zlepšilo výkon v cílové doméně za cenu mírného poklesu na webových datech (tabulka X.3).


#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header([], [Výchozí model], [Dotrénování (celá síť)], [Dotrénování (zmrazený backbone)]),
    [Validace MED: mAP50], [0,138], [0,498], [0,623],
    [Validace MED: recall `melee`], [0,10], [0,40], [0,50],
    [Validace MED: precision `melee`], [0,84], [0,97], [0,99],
    [Původní validace: mAP50], [0,777], [0,745], [0,755],
    [Původní validace: mAP50-95], [0,525], [0,477], [0,490],
    [`gun` AP50], [0,890], [0,867], [0,881],
    [`knife` AP50], [0,610], [0,600], [0,590],
    [`melee` AP50], [0,830], [0,768], [0,796],
  ),
  caption: [Ablace doménové adaptace. Validace MED pochází ze stejné bitky jako trénovací data.],
)


Varianta se zmrazeným backbonem dosáhla na validaci MED vyšší mAP50 a na původních datech zapomíná méně (pokles mAP50 o 0,022 místo 0,032); dotrénování bylo navíc zhruba třikrát rychlejší (přibližně 21 minut proti 61). Zapomínání se tím jen zmírnilo a nejvíce postihlo třídu `melee` (0,830 → 0,796), protože dotrénování posouvá význam třídy k dlouhým tyčím a model přichází o část schopnosti rozpoznat baseballové pálky. Lepší výsledek zmrazené varianty na MED lze vysvětlit tím, že s menším počtem učených parametrů se méně přeučí na 43 trénovacích boxech. Rozdíl recallu 0,50 a 0,40 však odpovídá jediné tyči z deseti a validace pochází ze stejné bitky, kamery a osob jako trénovací data, takže měří interpolaci uvnitř jedné události, nikoli zobecnění. Do pipeline byla nasazena varianta se zmrazeným backbonem (dále v1).

== Nezávislé ověření, negativní příklady a kalibrace prahu

=== Test na jiné bitce

Všech 28 dosud neprohlédnutých videí MED bylo prohlédnuto na kontaktních listech po šestnácti snímcích a kandidátní úseky v plném rozlišení; automatická metrika pohybu bitku od klidné scény neodlišila (průměrná změna mezi snímky 11,5 v klidné fázi proti 11,0 při bitce). Jediným dalším videem se zbraněmi je video 005: stejný dvůr a kamera jako u videa 004, ale jiná událost, jiní lidé a jiné typy předmětů, například červené pruty.

Z videa 005 bylo hustě vzorkováno 48 snímků a anotováno stejnou metodikou. Z 79 boxů navržených agenty bylo na kontaktních listech 12 vyloučeno (mimo jiné stín označený jako tyč s vysokou jistotou a pět téměř neviditelných rozmazaných čárek) a zůstalo 67 ověřených boxů třídy `melee`. Na videu 005 se nikdy netrénovalo. Výsledky shrnuje tabulka X.4; model v2 je popsán v následující části.


#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    table.header([Model], [AP50], [AP50-95], [Precision], [Recall]),
    [Výchozí], [0,219], [0,042], [0,44], [0,25],
    [Dotrénování (celá síť)], [0,640], [0,302], [0,87], [0,51],
    [v1 (zmrazený backbone)], [0,602], [0,281], [0,89], [0,49],
    [v2 (v1 + negativa)], [0,612], [0,288], [0,85], [0,45],
  ),
  caption: [Nezávislý test na videu 005 (48 snímků, 67 boxů `melee`). Precision a recall jsou uvedeny v bodě maxima F1.],
)


Dotrénování zobecňuje mimo bitku, na které se model učil: na události, kterou nikdy neviděl, vzrostla AP50 zhruba trojnásobně (0,22 → 0,60–0,64). Jde o nejsilnější doklad přínosu doménové adaptace v této práci, protože na rozdíl od validace MED (10 boxů ze stejné bitky) jde o 67 boxů z nezávislé scény. Pořadí obou variant dotrénování se zde obrátilo (0,640 pro celou síť proti 0,602), rozdíl mezi nimi tedy leží v pásmu šumu a žádná není v cílové doméně prokazatelně lepší. Rozhodujícím kritériem zůstalo menší zapomínání na původních datech, a nasazena proto zůstala varianta se zmrazeným backbonem (obrázek X.2).

#figure(
  image("figures/fig2_porovnani.png", width: 90%),
  caption: [Zobecnění na CCTV vs. ztráta na webových datech. Doménová adaptace zvýšila AP50 třídy melee na nezávislé bitce z videa 005 téměř trojnásobně (0,219 → 0,612). Na původní webové validaci přitom ztratila jen 0,02–0,03.],
)

Křivky precision–recall (obrázek X.3) ukazují, že žádný model nepřekročí recall 0,75. Test zároveň ukázal, že dotrénovaný model je v nové doméně podsebevědomý: maximum F1 leží u prahu jistoty přibližně 0,05 a při tehdejším produkčním prahu 0,40 měl model v1 precision 1,00, ale recall jen 0,31. Prahu se věnuje poslední část této podkapitoly.

#figure(
  image("figures/fig3_pr_krivky.png", width: 90%),
  caption: [PR křivky třídy melee na testu z videa 005 (48 snímků, 67 ručně ověřených tyčí; na videu 005 se netrénovalo). Výchozí model drží precision nad 0,5 jen do recallu ~0,2, dotrénované modely do ~0,6. Žádný model nepřekročí recall 0,75: zbylé tyče jsou rozmazané nebo zakryté a model je nevidí ani s nízkou jistotou.],
)

=== Negativní příklady z rvaček bez zbraní

Nejčastější falešnou detekcí dotrénovaného modelu byla natažená holá paže, typická i pro rvačky pěstmi a pro pochod s bílými papíry zdviženými nad hlavou, které se ve videích MED rovněž vyskytují. Šedesát snímků z rvaček ve videích 012, 014 a 016 a z pochodu ve videu 025 bylo přidáno do trénovacích dat jako negativa; 24 snímků z rvačky ve videu 017 a z pochodu ve videu 026 bylo ponecháno pro vyhodnocení.

Aby negativa neobsahovala skrytou zbraň, byly na nich všechny tři modely spuštěny s prahem 0,10 a všech 19 jejich detekcí bylo prohlédnuto; všechny byly falešné (natažené paže, stíny paží, člověk fotografující mobilem se zdviženými lokty, paže s papírem, nohy, čára na asfaltu).

Dotrénování modelu v2 bylo s v1 ve všech parametrech totožné včetně rozdělení a přimíchaných webových obrázků; jediným rozdílem bylo 60 negativních snímků navíc, takže jejich efekt je změřen izolovaně (tabulka X.5, obrázek X.4).


#figure(
  table(
    columns: 3,
    align: (left, left, left),
    table.header([], [v1], [v2 (+ negativa)]),
    [Falešné poplachy na 24 testovacích negativech, práh 0,10], [7 (ve 4 snímcích)], [3 (ve 2 snímcích)],
    [Falešné poplachy na 24 testovacích negativech, práh 0,40], [2], [0],
    [Test 005: AP50 `melee`], [0,602], [0,612],
    [Test 005: precision / recall], [0,89 / 0,49], [0,85 / 0,45],
    [Původní validace: mAP50], [0,755], [0,761],
    [Původní validace: `melee` AP50], [0,796], [0,807],
    [Validace MED: mAP50 (10 boxů)], [0,623], [0,609],
  ),
  caption: [Vliv negativních snímků z rvaček bez zbraní.],
)


#figure(
  image("figures/fig4_falesne_poplachy.png", width: 90%),
  caption: [Falešné poplachy na 24 testovacích snímcích rvaček bez zbraní v závislosti na prahu. Přidání 60 negativních snímků (v2) snížilo počet falešných poplachů zhruba na polovinu při každém prahu. Při nasazeném prahu 0,20 jsou to 2 poplachy proti 5 u v1.],
)

Na scénách bez zbraní hlásí model v2 zhruba poloviční počet falešných poplachů a při tehdejším produkčním prahu 0,40 žádný. Detekce zbraní se přitom nezhoršila: AP50 na nezávislém testu je prakticky shodná, posuny precision a recallu na testu i pokles validace MED (0,623 → 0,609) leží při daném počtu boxů v pásmu šumu a na původních datech model zapomíná o něco méně. Při 24 testovacích negativech a jednotkách falešných poplachů je však zřejmý jen směr efektu, nikoli jeho velikost. Model v2 byl nasazen do pipeline.

=== Kalibrace prahu

Práh 0,40 pro třídu `melee` byl odečten z křivky F1 výchozího modelu na webových datech. Dotrénovaný model však tyče v CCTV záběrech vidí často s jistotou pod 0,2, a práh 0,40 tak propouštěl jen asi šestinu tyčí. Při kontrole zapojení do pipeline to bylo patrné přímo: v deseti snímcích vrcholu bitky ve videu 005, kde jsou přibližně tři tyče na snímek, prošlo prahem pět detekcí.

Validace MED je s 10 boxy pro kalibraci nepoužitelná. Kalibrovalo se proto na časové polovině testu z videa 005: kalibrační část tvoří 19 snímků s 31 boxy, následuje jeden snímek se dvěma boxy jako ochranná mezera a ověřovací část tvoří 28 snímků s 34 boxy. Předem zvoleným kritériem bylo maximum F1 na kalibrační polovině. Falešné poplachy byly navíc měřeny na 24 testovacích snímcích rvaček bez zbraní a na 36 snímcích klidné scény z videí 001 a 002 (tabulka X.6, obrázek X.5).


#figure(
  table(
    columns: 5,
    align: (left, right, right, left, right),
    table.header([Práh], [Kalibrační P / R / F1], [Ověřovací P / R / F1], [FP rvačky (snímků z 24)], [FP klid]),
    [0,027], [0,91 / 0,65 / 0,75], [0,63 / 0,46 / 0,53], [11 (9)], [2],
    [0,10], [0,94 / 0,55 / 0,70], [0,86 / 0,35 / 0,50], [3 (2)], [1],
    [0,15], [0,97 / 0,55 / 0,70], [0,94 / 0,32 / 0,48], [2 (2)], [0],
    [_0,20_], [_1,00 / 0,54 / 0,70_], [_1,00 / 0,31 / 0,47_], [_2 (2)_], [_0_],
    [0,30], [1,00 / 0,48 / 0,65], [1,00 / 0,20 / 0,33], [1 (1)], [0],
    [0,40 (původní)], [1,00 / 0,46 / 0,63], [1,00 / 0,17 / 0,29], [0], [0],
  ),
  caption: [Kalibrace prahu třídy `melee` pro model v2. FP = falešné poplachy.],
)


#figure(
  image("figures/fig5_kalibrace.png", width: 90%),
  caption: [Kalibrace prahu třídy melee pro model v2. Na kalibrační polovině testu z videa 005 je F1 mezi prahem 0,10 a 0,20 plochá. Zvolen byl nejnižší práh plošiny s precision 1,00 (0,20). Na ověřovací polovině tím recall stoupl z 0,17 (práh 0,40) na 0,31 a precision zůstala 1,00.],
)

Maximum F1 vyšlo na samém okraji rozsahu, u prahu 0,027. Ten byl zamítnut, protože na snímcích bez zbraní dává 11 falešných poplachů v 9 z 24 snímků, tedy poplach zhruba v každém třetím snímku rvačky, a na ověřovací polovině precision klesá na 0,63. Na kalibrační polovině je F1 mezi prahy 0,10 a 0,20 plochá (0,70) a práh 0,20 je nejnižší bod této plošiny s precision 1,00. Proti původnímu prahu 0,40 se tím na ověřovací polovině zvýšil recall z 0,17 na 0,31 při zachované precision 1,00; na negativních snímcích rvaček přibyly dva falešné poplachy, na klidné scéně žádný. AP50 třídy `melee` činí na kalibrační polovině 0,699 a na ověřovací 0,510, což naznačuje, že se obtížnost scény výrazně mění i v rámci jednoho videa. Prahy tříd `gun` a `knife` se neměnily, protože pro ně v CCTV záběrech nejsou data.

Je nutné poctivě uvést, že změna kritéria (od maxima F1 k nejnižšímu prahu plošiny s plnou precision) padla až poté, co byla viděna ověřovací polovina pro prahy 0,027 a 0,40 a výsledky na negativních snímcích. Čísla ověřovací poloviny pro práh 0,20 proto nejsou zcela čistým testem. Samotná volba však vychází z vlastností kalibrační poloviny a z počtu falešných poplachů, nikoli z maxima na ověřovací polovině, které by naopak vedlo k prahu 0,027. Čistý odhad výkonu při zvoleném prahu by vyžadoval další anotované video.

Obrázek X.6 ukazuje typické chování modelu v2 při prahu 0,20.

#figure(
  image("figures/fig6_ukazky.png", width: 90%),
  caption: [Ukázky výstupu modelu v2 při prahu 0,20. a) obě tyče zachyceny; b) tyč zvednutá nad hlavu splývá s paží a model ji přehlédne; c) falešný poplach na rvačce bez zbraní: natažená paže s předmětem v ruce. Výběr je automatický: snímek s nejvíce správnými detekcemi, snímek s nejvíce přehlédnutími a falešný poplach s nejvyšší jistotou.],
)

=== Časové potvrzení poplachů

Dosavadní vyhodnocení pracovalo s jednotlivými snímky, systém však poplach vyhlašuje průběžně nad videem a rozhodující otázkou je, kolik falešných poplachů vznikne za minutu provozu. Celá videa proto byla zpracována stejně jako v pipeline, tedy při 5 snímcích za sekundu v nativním rozlišení, včetně hystereze pipeline: jednou vyhlášený poplach trvá ještě 8 snímků (1,6 s). Jako negativa posloužilo sedm videí rvaček bez zbraní (5,5 minuty) a tři klidné scény (3,4 minuty). Pět z rvaček nebylo dosud použito vůbec, ani v tréninku, ani při ladění prahu.

Ukázalo se, že test na 24 snímcích falešné poplachy výrazně podhodnotil. Při prahu 0,20 a vyhlášení poplachu po jediné detekci vzniklo na rvačkách 27 poplachů, tedy zhruba pět za minutu, a na klidných scénách dalších 12. Prvním zdrojem jsou jako dříve natažené paže, druhým, dosud neznámým, namalované čáry skákacího panáka na asfaltu, které model hlásí opakovaně na stejném místě.

Falešné detekce jsou však většinou ojedinělé záblesky v jednom snímku, kdežto skutečná tyč je vidět souvisle. Testováno bylo proto pravidlo, že poplach vznikne až tehdy, když detekce na stejném místě přetrvá k po sobě jdoucích snímků (tabulka X.7). Simulace vynechává kontrolu zápěstí, kterou pipeline provádí navíc, takže počty falešných poplachů jsou horním odhadem.


#figure(
  table(
    columns: 6,
    align: (right, left, right, left, right, right),
    table.header([práh], [k], [bitka v poplachu], [zpoždění], [poplachy na rvačkách (5,5 min)], [poplachy v klidu (3,4 min)]),
    [0,40], [1 (původní)], [90 %], [0,2 s], [10], [8],
    [0,20], [1], [93 %], [0,2 s], [27], [12],
    [_0,20_], [_2_], [_72 %_], [_0,4 s_], [_3_], [_5_],
    [0,20], [3], [69 %], [0,6 s], [2], [1],
    [0,10], [2], [90 %], [0,4 s], [10], [11],
  ),
  caption: [Časové potvrzení poplachů (model v2, 5 snímků/s). „Bitka v poplachu“ je podíl snímků bitky ve videu 005, během nichž je poplach aktivní; zpoždění se měří od prvního snímku s tyčí.],
)


Požadavek dvou po sobě jdoucích snímků snížil počet falešných poplachů na rvačkách z 27 na 3, na pěti dosud nepoužitých videích z 18 na 1. Skutečnou bitku přitom systém zachytí o 0,2 s později a poplach je aktivní 72 % jejího trvání místo 93 %. Proti původnímu nastavení (práh 0,40, jeden snímek) má nová kombinace méně falešných poplachů na rvačkách i v klidu (3 proti 10 a 5 proti 8). Zbývající poplachy v klidu působí hlavně namalované čáry, které časový filtr z principu nezachytí, protože jsou vidět trvale; zkoušené potlačení detekcí opakujících se na stejném místě pomohlo jen málo (5 → 3) a nasazeno nebylo. Do pipeline bylo nasazeno potvrzení dvěma snímky, které se v rámci logiky ARMED vztahuje ke konkrétní osobě, protože zbraň musí být u jejího zápěstí v obou snímcích.

Čáry na zemi a natažené paže ukazují, jaká negativa má smysl doplnit. Pro další verzi modelu bylo proto vybráno 41 snímků z dalších videí bez zbraní (panika, dýmovnice, klidné scény), na kterých model v2 falešně hlásí tyč. Videa se nepřekrývají se žádnou testovací sadou.

Předchozí simulace pracovala jen s detekcemi modelu v2. Pipeline ale dělá dvě věci navíc. Zbraň musí být u zápěstí konkrétní osoby. Do označení ARMED navíc vstupují i detekce tříd `baseball bat` a `knife` z obecného COCO detektoru, který pipeline používá pro zavazadla. Proto byla celá větev spuštěna znovu se skutečným kódem pipeline: detektorem objektů, trackery osob a objektů i logikou ARMED. Jedinou náhradou byl model pro osoby a skelety. Odhad pózy použitý v pipeline na testovacím počítači nešel spustit, a tak osoby a zápěstí dodal model YOLO11x-pose (tabulka X.8).


#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    table.header([varianta], [bitka v poplachu], [poplachy na rvačkách (5,5 min)], [z toho dosud nepoužitá videa], [poplachy v klidu (3,4 min)]),
    [původní: práh 0,40, jeden snímek], [100 %], [25], [21], [7],
    [práh 0,20, dva snímky], [100 %], [10], [7], [5],
    [_nasazeno: dva snímky, COCO pálka až od 0,50_], [_93 %_], [_5_], [_2_], [_5_],
    [dva snímky, bez COCO pálky], [90 %], [4], [1], [5],
  ),
  caption: [Poplachy celé větve ARMED s kontrolou zápěstí (5 snímků/s). Poplachem se rozumí úsek, kdy je aspoň jedna osoba označena jako ozbrojená.],
)


Časové potvrzení se osvědčilo i v plné pipeline: počet poplachů na rvačkách klesl z 25 na 10, na dosud nepoužitých videích z 21 na 7. Kontrola zápěstí naproti tomu falešné poplachy prakticky nesnížila. Detekce na natažené paži zápěstí z podstaty obsahuje, a čáry a stíny na zemi leží v dosahu rukou kolemjdoucích. Nečekaným zjištěním je role COCO detektoru. Na rvačkách bez zbraní hlásí třídu `baseball bat` hlavně na diagonálních stínech na asfaltu, a je tak hlavním zdrojem zbývajících poplachů. Ve skutečné bitce ale stejná třída často správně najde bílé a červené tyče a doplňuje model v2: bez ní je bitka v poplachu 90 % času místo 93 %. Pro třídu `baseball bat` byl proto zaveden vlastní práh 0,50, zatímco obecný nízký práh zůstal pro zavazadla. Počet poplachů na rvačkách tím klesl z 10 na 5 při pokrytí bitky 93 % (obrázky X.7 a X.8). Volbu podporuje i to, že skutečné tyče měly u COCO pálky jistotu zhruba 0,42–0,64 a stíny na asfaltu 0,16–0,34. Hodnota však byla zvolena na testovacích videích a čistě ji bude možné ověřit až na dalším záznamu se zbraněmi. Absolutní čísla se v provozu mohou lišit, protože YOLO11x-pose detekuje osoby jinak než model použitý v pipeline. Srovnání variant mezi sebou je ale férové, protože všechny pracují se stejnými osobami a skelety.

#figure(
  image("figures/fig7_armed.png", width: 90%),
  caption: [Celá větev ARMED s kontrolou zápěstí při 5 snímcích/s. Každý bod je jedno nastavení: vodorovně počet falešných poplachů na 5,5 minuty rvaček bez zbraní, svisle podíl bitky ve videu 005, po který je poplach aktivní. Nasazené nastavení (zeleně) kombinuje práh 0,20, potvrzení dvěma snímky a práh 0,50 pro COCO pálku.],
)

#figure(
  image("figures/fig8_armed_ukazka.png", width: 90%),
  caption: [Výstup větve ARMED při nasazeném nastavení. a) první poplach v bitce 005: dvě osoby s tyčí u zápěstí; b) vrchol bitky: čtyři ozbrojené osoby, jednu z nich zachytil COCO detektor jako pálku; c) rvačka bez zbraní ve videu 019 bez poplachu. Šedě osoby, červeně osoby označené jako ozbrojené, žlutě a oranžově detekce zbraní.],
)

=== Zapojení do pipeline

Detektor je zapojen do modulu detekce podezřelých objektů vedle COCO detektoru zavazadel. Test zapojení, který nahrazuje jen detektor osob a odhad skeletu, potvrdil, že se dodržují prahy jednotlivých tříd a že detekce třídy `melee` vstupuje do logiky označování ozbrojených osob, která je ve výchozím nastavení vypnutá a zapíná se přepínačem. Zvětšení vstupního snímku detekci neovlivní: snímek 854 × 480 px a jeho bikubické zvětšení 4× dávají prakticky stejné detekce (14 proti 16 nad prahem 0,05, stejných 5 nad 0,40), protože model vstup stejně převádí na 960 px.

== Omezení a diskuse

Prvním a nejzávažnějším omezením je, že v CCTV záběrech je ověřena pouze třída `melee`. Videa MED neobsahují pistole ani nože, takže pro třídy `gun` a `knife` existují jen výsledky na webových datech, které o cílové doméně, jak ukázala podkapitola X.6, vypovídají málo. Prahy těchto tříd nebyly pro dotrénovaný model přeladěny a třída `knife` má již na webových datech recall jen 0,587.

Druhým omezením je úzkost cílových dat. Všechny pozitivní příklady z CCTV pocházejí ze dvou bitek na stejném dvoře, natočených stejnou kamerou za podobného osvětlení. Nezávislý test na videu 005 prokázal zobecnění na jinou událost, nikoli na jiné prostředí, úhel kamery, rozlišení nebo noční podmínky. Doménová adaptace tedy přizpůsobila model jedné konkrétní scéně, nikoli záběrům bezpečnostních kamer obecně.

Třetím omezením jsou malé testovací sady. Validace MED má 10 boxů (během tréninku kolísala mAP50 mezi epochami přibližně o 0,1), nezávislý test 67 boxů, po rozdělení pro kalibraci 31 a 34, a snímkových testovacích negativ je 24. Spolehlivé jsou proto jen velké efekty: zhruba trojnásobné zlepšení AP50 díky doménové adaptaci a zhruba poloviční počet falešných poplachů díky negativům. Ve snímcích, z nichž byly sporné objekty vyloučeny, zůstaly tyto objekty neoznačené a jejich detekce se počítá jako falešná, takže precision je spíše podhodnocená.

Čtvrtým omezením je kalibrace prahu, jejíž kritérium se změnilo poté, co byla viděna část ověřovacích dat. Ani po kalibraci model nezachytí většinu tyčí: na ověřovací polovině je recall 0,31. Detektor je tedy konzervativní: má málo falešných poplachů, ale mnoho přehlédnutí.

Za zmínku stojí i role jazykového modelu při anotaci. Ten sám byl zdrojem chyb, například stínu označeného jako tyč s vysokou jistotou, a kvalitu dat zajistila až struktura postupu: mřížka místo odhadu, strukturované verdikty místo přímého zápisu, odkládání nejistých případů a vizuální kontrola na kontaktních listech. Šedesát čtyři sporných případů (tyč proti paži nebo stínu) dosud čeká na posouzení člověkem.

Konečně trvá mezera v dostupných datech: holá tyč nebo hůl jako anotovaná třída ve veřejných datasetech chybí a třída `melee` je v původních datech ze 72,7 % tvořena baseballovými pálkami. Obecné řešení by vyžadovalo anotaci záznamů z více prostředí, například z datasetů násilí zachyceného bezpečnostními kamerami, jako jsou UCF-Crime, RWF-2000 nebo CCTV-Fights @TODO-citace.

== Shrnutí

Do pipeline byl doplněn detektor zbraní založený na modelu YOLO11l se třemi třídami (`gun`, `knife`, `melee`); zavazadla zůstala na COCO detektoru. Model byl natrénován na 24 340 obrázcích ze šesti veřejných datasetů, jejichž sestavení vyžadovalo deduplikaci a vizuální ověření tříd. Na validaci webových dat dosáhl model mAP50 0,775, se slabinou u nožů, kterou nevyřešil oversampling ani ladění prahu.

Hlavním zjištěním je, že výkon na webových datech o použitelnosti v kamerových záznamech téměř nevypovídá. Na CCTV záběrech z MED model většinu zbraní nenašel, protože tyče a hole, které se v bitkách skutečně používají, ve veřejných datech chybějí. Doménová adaptace na 252 asistovaně anotovaných snímcích s časovým rozdělením dat, zmrazeným backbonem a přimícháním původních dat zvýšila AP50 třídy `melee` na nezávislé bitce z 0,219 na 0,602 při ztrátě jen 0,02–0,03 na původních webových datech. Negativní snímky z rvaček bez zbraní snížily počet falešných poplachů zhruba na polovinu při AP50 0,612. Po kalibraci prahu na 0,20 zachytí nasazený model na ověřovací části testu zhruba třetinu tyčí s precision 1,00 a časové potvrzení dvěma snímky spolu s vyšším prahem třídy „baseball bat“ obecného COCO detektoru snížilo počet poplachů celé větve ARMED na rvačkách bez zbraní z 25 na 5 za 5,5 minuty videa při pokrytí skutečné bitky 93 %. Výsledky platí pro jednu kameru a pro jedinou třídu zbraní ověřenou na CCTV (tyče, latě a pruty) a jejich zobecnění na jiná prostředí zůstává otevřenou otázkou pro další práci.


// ---- Poznámky pro autora (z Markdownu) ----
// 
// **Co ověřit:**
// 
// - Číslování: kapitola, obrázky a tabulky používají zástupné „X“; nahradit skutečným číslem kapitoly.
// - Výchozí model na původní validaci: v podkapitole X.5 je mAP50 0,775 / mAP50-95 0,523 (hlavní vyhodnocení), v tabulce X.3 je 0,777 / 0,525 (opakované vyhodnocení při ablaci). Obojí je převzato ze zdroje; zvážit, zda rozdíl vysvětlit poznámkou pod čarou.
// - Kalibrace na stejném objektu: technická zpráva uvádí box natrénovaného modelu 101 × 145 px, odhad hlavního modelu 40 × 68 px (příliš malý) a box subagenta s mřížkou 69 × 90 px (sedí); anotační specifikace naopak uvádí 40 × 68 px jako správnou velikost a 100 × 145 px od modelu. Proto do textu nebyla rozměry zařazena; před případným doplněním rozpor vyjasnit.
// - Recall 0,31 při prahu 0,40 u v1 na celém testu 005 (X.8) a recall 0,31 při prahu 0,20 u v2 na ověřovací polovině jsou dvě různá čísla, která se náhodou shodují; zkontrolovat, že čtenář je nezamění.
// - Popisek obrázku X.4 uvádí 5 falešných poplachů u v1 při prahu 0,20; tato hodnota je jen v datech obrázku, ne v textu zprávy (souhlasí s figures_data.json).
// - Název „MuSGD“ a jeho vztah k metodě Muon ověřit v dokumentaci Ultralytics.
// - Při zkracování (říjen 2026) byly z X.4 vypuštěny přesné hodnoty hyperparametrů a augmentací: moment 0,937, 3 epochy warmupu, rotace do 10°, změna měřítka 0,6, posun 0,15, mozaika vypnutá v posledních 10 epochách, mixup 0,1. Pokud je oponent bude chtít, patří spíš do přílohy (tabulka konfigurace). Z X.4 vypadl i údaj, že rychlost učení během warmupu MuSGD vystoupala „zhruba na trojnásobek“ a že klasifikační ztráta vzrostla „přibližně 2,5krát“ (druhý údaj zůstal v popisku obr. X.1). Z X.7 vypadla věta, že roztažení kontrastu „v několika případech ukázalo, že domnělá tyč jsou ve skutečnosti dvě předloktí“, a srovnání „hustá extrakce dala 2,5krát více boxů na pětkrát menším počtu snímků“; z X.6 věta „žádná detekce nepřekročila jistotu 0,8“. Ze shrnutí (X.10) vypadla zmínka o divergenci MuSGD (zůstává v X.4) a z X.4 vysvětlení propadu SGD ve 3. epoše (zůstává v popisku obr. X.1); z X.7 počet 198 negativních snímků řídké sady (dopočitatelný z 208 − 10). Vrátit, pokud jsou pro argumentaci potřeba.
// - Shrnutí (X.10) uvádí výsledek plné simulace větve ARMED (25 → 5 poplachů za 5,5 min) místo dřívějších „dvou falešných poplachů ve 24 snímcích“; snímkový test falešné poplachy podhodnotil.
// - **Čísla tabulky X.8 a navazujícího textu přepočítána 2026-10-28** po opravě trackeru (osoba po výpadku detekce a při míjení drží ID, viz `CODE_REVIEW_PIPELINE.md` body 1 a 4). Nasazené nastavení: 5 poplachů na rvačkách místo 4, pokrytí 93 % beze změny. **Varianta bez COCO pálky má teď pokrytí 90 % místo 69 %**, takže argument „COCO pálka a model v2 se doplňují“ je slabší (+3 p. b. pokrytí za +1 poplach). Zvážit přeformulování odstavce za tabulkou X.8, případně i volbu nasazení. Pozor: při přepočtu na Windows PC (CUDA, ultralytics 8.3) najde COCO detektor pálku mnohem méněkrát (17 místo 104 detekcí) a tam bez COCO pálky klesne pokrytí na 62 % proti 90 % s ní (`TRAINING_REPORT.md` §6h). Role COCO pálky tedy závisí i na prostředí, v němž detekce vznikly; v textu to uvést jako nestabilní efekt.
// - Tabulka X.7: pokrytí bitky (90 % při prahu 0,40) vypadá v rozporu s recallem boxů (0,17). Rozdíl dělá hystereze (poplach trvá 8 snímků) a to, že k poplachu stačí jedna potvrzená tyč. Zvážit vysvětlující větu. Bitka 005 navíc sloužila i ke kalibraci prahu, takže pokrytí není nezávislý údaj.
// - Nejasné, z kterých snímků pochází 126 kandidátních boxů předanotace (řídká sada, hustá, nebo obě); doplnit z anotačních záznamů.
// - Argument pro zmrazený backbone stojí jen na menším zapomínání (rozdíl ~0,01 mAP50); na nezávislém testu bylo pořadí variant opačné. Formulovat opatrně.
// - Tvrzení „precision je spíše podhodnocená“ platí jen pro snímky s vyloučenými spornými objekty; míra efektu není změřena.
// 
// **Kde chybí citace [CITACE]:**
// 
// - Motion Emotion Dataset (MED) — v README projektu je odkaz na repozitář, ověřit primární publikaci.
// - YOLO11 / Ultralytics (v README projektu je softwarová citace).
// - RT-DETR (citace je v README projektu).
// - Dataset COCO (Lin et al.).
// - Roboflow Universe a jednotlivé použité datasety (u CC BY 4.0 je uvedení autora povinné).
// - Optimalizátor Muon (a případně MuSGD v Ultralytics).
// - Katastrofické zapomínání (catastrophic forgetting).
// - Použitý jazykový model a agentní anotace (zvážit i obecnou literaturu o asistované/LLM anotaci).
// - UCF-Crime, RWF-2000, CCTV-Fights.
// - ESPCN je citován jinde v práci (README projektu), zde jen zmíněn.
