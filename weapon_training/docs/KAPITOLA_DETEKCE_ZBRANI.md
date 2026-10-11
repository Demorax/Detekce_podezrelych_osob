# X Detekce zbraní a nebezpečných předmětů

## X.1 Motivace a cíl

Analýza chování osob, které se věnují předchozí kapitoly, pracuje s pohybem: s polohou, trajektorií a postojem odvozeným ze skeletu. Pohyb sám ale o míře ohrožení často nerozhoduje; strkanice dvou osob může být hrou i napadením. Rozhodující informací bývá, zda některá z osob drží předmět, kterým lze způsobit zranění, a upozornění na ozbrojenou osobu patří k nejcennějším signálům, jaké může automatický dohledový systém poskytnout.

Cílem této části práce bylo doplnit do zpracovatelského řetězce detektor zbraní a nebezpečných předmětů, který v jednotlivých snímcích z kamery vyhledá střelné zbraně, nože a tupé či sečné zbraně držené v ruce. Detektor nepracuje izolovaně. Jeho výstup se kombinuje s výsledky předchozích stupňů: s detekcí osob a s odhadem jejich skeletu. Osoba je označena jako ozbrojená (příznak ARMED) teprve tehdy, když se box detekované zbraně nachází u zápěstí některé z osob. Tato podmínka odpovídá zadání (zajímá nás zbraň v ruce, nikoli předmět ležící opodál) a zároveň funguje jako dodatečný filtr, který odstraní část falešných detekcí v místech, kde žádná ruka není.

Cílovou doménou jsou záběry z Motion Emotion Datasetu (MED) [CITACE], tedy statická bezpečnostní kamera snímající prostranství šikmo shora, s davy lidí, malými objekty, pohybovým rozmazáním a proměnlivým osvětlením. Jak ukáže zbytek kapitoly, právě rozdíl mezi touto doménou a fotografiemi, na kterých jsou postaveny veřejně dostupné datasety zbraní, se stal ústředním problémem celé úlohy.

## X.2 Volba přístupu

### Detektor a jeho velikost

Předběžná rešerše ukázala, že hotový předtrénovaný detektor tupých zbraní v použitelné kvalitě není k dispozici. Realistickou cestou proto bylo dotrénování (fine-tuning) obecného detektoru objektů na existujících anotovaných datech. Zvažovány byly tři varianty: dotrénování modelu YOLO11 [CITACE] ve velikosti *l* při vstupním rozlišení 960 px, dotrénování větší varianty *x* při 640 px a trénink transformerového detektoru RT-DETR [CITACE] od nuly.

Trénink transformeru od nuly byl zamítnut nejdříve. Přibližně 24 tisíc obrázků, které se podařilo sestavit, je pro něj málo, trénink by trval desítky hodin a s velkou pravděpodobností by prohrál s dotrénováním vah předtrénovaných na datasetu COCO [CITACE]. Mezi zbylými dvěma variantami rozhodla povaha cílových dat. U zbraní v záběru bezpečnostní kamery není limitujícím faktorem kapacita sítě, ale velikost objektu: tyč nebo pálka zabírá v MED snímku jen desítky pixelů. Vyšší vstupní rozlišení proto přináší víc než hlubší síť při srovnatelném výpočetním čase. Zvolen byl model YOLO11l se vstupem 960 px, inicializovaný vahami předtrénovanými na COCO.

### Tři třídy místo šesti

Původní plán počítal se šesti třídami: `gun`, `knife`, `melee`, `backpack`, `handbag` a `suitcase`. Průzkum dostupných dat však ukázal, že pro zavazadla je ve veřejných datasetech jen málo instancí: kabelek (`handbag`) bylo ve všech nalezených zdrojích dohromady přibližně 200, kufrů (`suitcase`) přibližně 2 700 a batohů (`backpack`) 819.

Podstatnější než samotný nedostatek dat ale bylo, že všechny tři třídy jsou součástí datasetu COCO. Předtrénovaný detektor YOLO11x, který pipeline už používá, je rozpoznává na základě učení ze zhruba 118 000 obrázků. Dotrénování z několika set nebo tisíců instancí by dalo horší, nikoli lepší výsledek. Vlastní detektor proto řeší výhradně to, co COCO model neumí, tedy zbraně, a zavazadla zůstávají na původním modelu. Výsledné třídy jsou:

- `gun`: pistole, revolvery, pušky, brokovnice a samopaly,
- `knife`: nože, dýky, řezáky a sekáčky,
- `melee`: tupé a sečné zbraně držené v ruce, tedy pálky, sekery, meče, kopí a (po doménové adaptaci popsané níže) i tyče, hole a latě.

Pro tyče a hole nebyla založena samostatná čtvrtá třída, ačkoli se to později ukázalo jako hlavní mezera modelu. Nová třída by vyžadovala změnu detekční hlavy a model by se ji učil od začátku, místo aby rozšířil pojem, který už zná. Sémanticky tyč do třídy `melee` patří, protože jde o tupou zbraň drženou v ruce, a downstream logika označování ozbrojených osob pracuje s pevnou množinou tříd, kterou by bylo nutné upravit.

## X.3 Trénovací data

### Zdroje a sestavení datasetu

Trénovací data pocházejí z veřejných datasetů platformy Roboflow Universe [CITACE]. Staženo bylo sedm datasetů o celkovém objemu 3,3 GB, z nichž šest bylo použito. Nejkvalitnějším zdrojem byl dataset s vyváženými třídami baseballová pálka, nůž a pistole (1 240, 1 012 a 1 144 instancí) pod licencí CC BY 4.0; dalšími zdroji byly datasety s rozmanitými typy střelných a chladných zbraní, dva datasety baseballových pálek a menší dataset pistolí [CITACE pro každý dataset].

Zdrojové třídy (jen do třídy `gun` jich bylo namapováno přes dvacet) se na cílové převáděly podle názvu, u nejednoznačných tříd však vizuální kontrolou náhodného vzorku výřezů. Tak bylo ověřeno, že třída `heavy-weapon` (11 598 boxů) obsahuje výhradně dlouhé palné zbraně a že nepojmenovaná třída `0` v jednom z datasetů jsou baseballové pálky. Naopak třída `short sword`, původně mapovaná na `melee`, se ukázala obsahovat dýky vizuálně nerozeznatelné od velkých nožů. Jejích 501 instancí bylo přesunuto do třídy `knife`, což zároveň odstranilo soupeření dvou tříd o stejný vzhled a posílilo nejslabší třídu.

Při sestavování bylo nutné vyřešit tři metodické problémy. Pokud se z obrázku odstraní anotace zbraňové třídy, kterou nelze namapovat, a obrázek zůstane, stane se z něj pozadí a model se učí, že zbraň je pozadím; u nejednoznačných tříd (například obecné `weapon`) se proto zahazoval celý obrázek, zatímco u neškodných tříd (například `person`) jen label. Veřejné datasety zbraní dále vznikají často jako kopie jiných, takže tentýž obrázek se může objevit v trénovacím splitu jednoho a ve validačním splitu druhého zdroje; deduplikace podle hashe obsahu odstranila 5 049 duplicit (10,9 %), při kolizi s předností validačního splitu. Konečně dva exporty míchaly obdélníkové boxy s polygony (549 řádků), které byly převedeny na opsaný obdélník.

Z původních 46 173 kandidátních obrázků zbylo po deduplikaci 41 124 unikátních a po vyřazení obrázků s nezmapovanými labely 24 340. Rozdělení do splitů bylo převzato z původních datasetů, nikoli provedeno náhodně, protože augmentované varianty téhož obrázku se v exportech vyskytují jen v trénovacím splitu a náhodné rozdělení by je rozptýlilo i do validace. Výsledný dataset má 19 924 trénovacích, 3 503 validačních a 913 testovacích obrázků s 28 156 instancemi, z toho 14 218 (50,5 %) `gun`, 10 730 (38,1 %) `melee` a 3 208 (11,4 %) `knife`.

### Zjištěné vady veřejných datasetů

Tři datasety doporučené v rešerši jako zdroje tupých zbraní se při ověření ukázaly jako nepoužitelné, a to z důvodů, které nebyly patrné z popisu ani z metadat. V prvním z nich byly názvy tříd zbytky textu README (například řádek rovnítek nebo věta o exportu z platformy), druhý neobsahoval žádné tupé zbraně a navíc neměl exportovanou verzi a ve třetím byly tři varianty téže třídy pušek lišící se jen tečkou na konci názvu. Ze všech zkoumaných zdrojů tak vyplynulo, že holá tyč, hůl nebo obušek jako samostatně anotovaná třída ve veřejně dostupných datech neexistuje.

Nejzávažnější byl případ největšího nalezeného datasetu (15 981 obrázků, 16 966 boxů) s jedinou třídou `weapon`. Vizuální kontrola vzorku 48 výřezů prokázala, že třída obsahuje směs pistolí, pušek a nožů. Namapování na `gun` by do této třídy vneslo nože, rozdělení by vyžadovalo kompletní přeanotování, a dataset byl proto vyřazen celý. Obecným poučením je, že u datasetů z otevřených platforem nelze spoléhat na metadata: rozhraní platformy u některých datasetů vracelo jiný seznam tříd, než jaký obsahoval skutečný export, a třídy je proto nutné číst z exportované konfigurace a jejich obsah ověřit pohledem.

Sestavený dataset má i vady, které odstranit nešlo. Celkem 7 798 z 10 730 boxů třídy `melee` (72,7 %) pochází z baseballových fotografií, takže hrozí, že se model naučí kontext sportovní scény místo tvaru předmětu. Třída `knife` je zastoupena v poměru 1 : 4,4 vůči třídě `gun`. A především všechna data jsou webové a sportovní fotografie: zbraně zblízka, v dobrém světle, z úrovně očí. Cílová doména je opačná.

## X.4 Trénink

Trénink probíhal na stroji NVIDIA DGX Spark s GPU GB10 a 121 GB paměti sdílené mezi procesorem a grafickým akcelerátorem, s knihovnou Ultralytics [CITACE] ve verzi 8.4.115. Sdílená paměť má nižší propustnost než grafická paměť desktopových karet, takže trénink je omezen propustností a zvětšování dávky ho neurychluje; při dávce 32 a rozlišení 960 px model zpracoval zhruba 14 obrázků za sekundu.

Finální konfigurace používala optimalizátor SGD s počáteční rychlostí učení 0,005, kosinovým útlumem, momentem 0,937, třemi epochami warmupu, dávkou 32 a smíšenou přesností. Augmentace byly nastaveny s ohledem na rozdíl domén: rotace do 10°, změna měřítka 0,6, posun 0,15, horizontální převrácení, mozaika (vypnutá v posledních 10 epochách), mixup 0,1 a posuny barevného tónu, sytosti a jasu. Trénink byl naplánován na 100 epoch s předčasným ukončením po 25 epochách bez zlepšení. Zastavil se po 90 epochách a 43,4 hodinách; jako nejlepší byla vybrána epocha 65. Kritériem výběru přitom nebyla mAP50, ale kombinovaná metrika knihovny (0,1 · mAP50 + 0,9 · mAP50-95), která upřednostňuje přesnou lokalizaci. Nejvyšší mAP50 (0,7804) dosáhla epocha 76, rozdíl proti epoše 65 je však zanedbatelný.

### Divergence optimalizátoru MuSGD

K finální konfiguraci se dospělo až po dvou neúspěšných bězích, jejichž příčina je samostatným zjištěním. Při výchozím nastavení `optimizer='auto'` zvolila použitá verze knihovny optimalizátor MuSGD, založený na metodě Muon [CITACE], s rychlostí učení 0,01. Během warmupu, kdy rychlost učení vystoupala zhruba na trojnásobek, model divergoval: mAP50 klesla z 0,304 v první epoše na 0,0999 ve třetí a validační klasifikační ztráta vzrostla z 2,466 na 6,119, tedy přibližně 2,5krát. Současně rostla i ztráta trénovací, takže nešlo o šum ani o přeučení. Pravděpodobnou příčinou je, že metoda Muon počítá s výrazně menší efektivní rychlostí učení pro maticové parametry než klasický SGD a hodnota 0,01 je pro ni na tomto datasetu příliš vysoká. Explicitní volba SGD s rychlostí učení 0,005 problém odstranila okamžitě: mAP50 po první epoše byla 0,558 proti 0,304 u MuSGD. Průběh obou běhů srovnává obrázek X.1. Krátkodobý propad SGD ve třetí epoše odpovídá vrcholu warmupu, kdy rychlost učení dosáhne cílové hodnoty; od čtvrté epochy metriky rostou. Praktickým důsledkem je doporučení nespoléhat u této verze knihovny na automatickou volbu optimalizátoru.

![Obrázek X.1: Průběh tréninku na webových datech. Výchozí `optimizer='auto'` zvolil MuSGD s rychlostí učení 0,01, který během warmupu divergoval (mAP50 0,30 → 0,10, validační klasifikační ztráta 2,5×). Explicitní SGD s lr0 0,005 konverguje k mAP50 0,78. Propad SGD ve 3. epoše je vrchol warmupu rychlosti učení, ze kterého se model zotavil.](figures/fig1_trenink.png)

## X.5 Vyhodnocení na webových datech

Na validačním splitu (3 503 obrázků, 4 058 instancí) dosáhl model mAP50 0,775 a mAP50-95 0,523 při celkové precision 0,878 a recallu 0,724. Výsledky po třídách shrnuje tabulka X.1.

**Tabulka X.1:** Výsledky výchozího modelu na validačním splitu webových dat.

| Třída | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| `gun` | 0,878 | 0,822 | 0,884 | 0,634 |
| `melee` | 0,871 | 0,763 | 0,830 | 0,533 |
| `knife` | 0,884 | 0,587 | 0,611 | 0,403 |
| celkem | 0,878 | 0,724 | 0,775 | 0,523 |

Testovací split dává lepší čísla (mAP50 0,855, u třídy `knife` dokonce 0,955), jako hlavní výsledek se však neuvádí. Testovací split totiž obsahuje jen tři z pěti zdrojových datasetů, které do splitů přispívají, a chybí v něm zejména nože z nejobtížnějšího zdroje. Není tedy tvrdším, ale užším a snazším měřítkem.

Zřetelnou slabinou je třída `knife`. Precision 0,884 při recallu 0,587 znamená, že model nůž, který označí, označí spolehlivě, ale přibližně 41 % nožů přehlédne. Pro detektor hrozeb je to horší typ chyby, protože nedetekovaná zbraň je propuštěná hrozba. Nabízela se dvě zjevná řešení a obě selhala. Oversampling, při kterém byl každý trénovací obrázek s nožem zařazen třikrát (poměr `gun` : `knife` se tím změnil z 4,4 : 1 na 1,6 : 1), nezvýšil recall vůbec (0,587 → 0,585) a celkovou mAP50 mírně snížil (0,775 → 0,770). Snížení prahu jistoty rovněž nepomáhá: pokles prahu z 0,55 na 0,12 zvedl recall nožů jen z 0,574 na 0,621 za cenu poklesu precision o 0,125. Chybějící nože tedy nejsou slabé detekce těsně pod prahem; model je nevidí vůbec. Oba výsledky ukazují na stejnou příčinu, kterou není nevyváženost tříd, ale nedostatečná rozmanitost trénovacích dat. Duplikace týchž příkladů žádnou novou informaci nepřidává a jediným řešením jsou nová anotovaná data.

Prahy jistoty pro jednotlivé třídy byly odečteny z křivky F1 v závislosti na prahu na validačním splitu (tabulka X.2). Zvolené hodnoty leží záměrně mírně pod maximem F1, protože navazující logika označování ozbrojených osob vyžaduje přítomnost zápěstí u boxu zbraně a část falešných detekcí tak odfiltruje sama. Posun k vyššímu recallu se proto vyplatí.

**Tabulka X.2:** Volba prahů jistoty z křivky F1 na validačním splitu (výchozí model).

| Třída | Práh při max. F1 | Max. F1 | Zvolený práh | Precision | Recall |
|---|---|---|---|---|---|
| `gun` | 0,411 | 0,851 | 0,35 | 0,831 | 0,852 |
| `knife` | 0,388 | 0,709 | 0,30 | ~0,85 | ~0,60 |
| `melee` | 0,494 | 0,817 | 0,40 | 0,806 | 0,792 |

## X.6 Rozdíl domén: test na CCTV záběrech

Žádná metrika z validačního splitu nemůže odpovědět na otázku, jak model funguje v cílové doméně, protože validační data mají stejné rozložení jako trénovací. Model byl proto spuštěn na 208 snímcích z videí 001, 002 a 004 datasetu MED, zpracovaných stejně jako v ostatních částech pipeline, s prahem jistoty 0,25.

Výsledek byl střízlivý. Model vrátil jen 17 detekcí v 17 snímcích (8,2 %): 15 v třídě `melee` s průměrnou jistotou 0,408 a maximem 0,684, dvě v třídě `gun` a žádnou v třídě `knife`. Žádná detekce nepřekročila jistotu 0,8. Teprve při prohlížení snímků se ukázalo, že video 004 zachycuje hromadnou bitku, v níž lidé skutečně drží improvizované zbraně, především tyče. Původní předpoklad, že MED zbraně neobsahuje, byl mylný, a test tak získal reálnou vypovídací hodnotu.

Kvalitativní rozbor ukázal oba typy chyb. Model správně označil podlouhlý předmět v ruce osoby v davu, viděný shora, takže přenos do nové domény alespoň částečně funguje. Ve stejném videu však přehlédl jasně viditelnou dlouhou bílou tyč, kterou jiná osoba drží přes celou šíři těla a která je nejnápadnějším předmětem ve snímku. Falešné detekce mířily na ruce a obnažená předloktí, která mají při tomto rozlišení podobný tvar i jas jako tyč: jsou podlouhlá, světlá a zužují se.

Příčinu selhání lze pojmenovat konkrétně. Třída `melee` byla naučena převážně na baseballových pálkách ze sportovních fotografií, tedy na krátkých kuželovitých předmětech. Dlouhá tenká tyč je jiný tvar a v trénovacích datech chybí. K tomu přistupuje obecný rozdíl mezi fotografií zbraně zblízka a pohledem kamery shora, kde má předmět desítky pixelů a je rozmazaný a částečně zakrytý. Model trénovaný na webových datech tedy na CCTV záběrech není připraven na spolehlivou detekci: nezaplavuje výstup falešnými poplachy, ale většinu hrozeb nenajde.

## X.7 Doménová adaptace

Test v předchozí podkapitole ukázal, že nejúčinnějším krokem nebude další ladění na webových datech, ale doplnění anotovaných snímků přímo z cílové domény a dotrénování modelu na nich.

### Postup anotace

Anotace snímků z bezpečnostní kamery je obtížnější než anotace webových fotografií. Snímky mají po zvětšení rozměr 3416 × 1920 px a typický box zbraně zabírá jen několik procent šířky snímku, takže bez přiblížení nelze objekt ani spolehlivě rozpoznat, ani přesně ohraničit. Anotace proto probíhala s asistencí velkého jazykového modelu s multimodálním vstupem a jeho podřízených agentů [CITACE], a to podle písemné anotační specifikace, která definuje třídy, pravidla pro kreslení boxů a postup v nejasných případech. Výstupy modelu nebyly přebírány automaticky; každý krok byl navržen tak, aby šel ověřit.

Postup měl tři fáze. V první fázi natrénovaný detektor vygeneroval předanotace s nízkým prahem jistoty 0,10. Záměrem byl vysoký recall, tedy co nejvíce kandidátů ke kontrole, nikoli přesnost. Ve druhé fázi agenti každý snímek prošli metodou od hrubého k jemnému: nejprve celý snímek se souřadnicovou mřížkou a předanotacemi, poté přiblížené výřezy kolem jednotlivých kandidátů. Mřížka se ukázala jako zásadní. Bez ní dosahoval odhad boxu jazykovým modelem shody IoU 0,70 s referencí, což je pro trénink příliš volné; s přiblíženým výřezem a mřížkou se souřadnice odečítají, nikoli odhadují, s přesností přibližně ±5 px. U tmavého asfaltu a stínů pomohlo percentilové roztažení kontrastu, které v několika případech ukázalo, že domnělá tyč jsou ve skutečnosti dvě předloktí. Agenti nevraceli přímo labely, ale strukturované verdikty ke každé předanotaci (přijmout, zpřesnit box, zamítnout) spolu se seznamem přehlédnutých objektů a nejasných případů. Nejasný případ se do dat nikdy nezapsal; byl odložen ke kontrole člověkem. Ve třetí fázi byly všechny boxy před zápisem prohlédnuty na kontaktních listech, tedy přehledových tabulích výřezů, na nichž lze rychle porovnat desítky boxů vedle sebe; sporné boxy byly vyloučeny.

Během práce se objevily tři opakující se pasti, které byly následně doplněny do specifikace. První je záměna tyče s obnaženým předloktím: tyč má konstantní šířku, často jinou barvu než kůže a odděluje se od ruky, zatímco paže se zužuje od ramene k ruce. Druhou je záměna tyče s jejím stínem: skutečná tyč mívá po hraně světlý odlesk, stín nikdy, a stín se napojuje na stínovou masu těla. Zdánlivý směr stínů se přitom napříč snímkem mění perspektivou nakloněné kamery, takže jej nelze brát jako konstantu a je nutné ho ověřit lokálně podle nejbližší osoby. Třetí pastí jsou statické artefakty pozadí, jako namalované čáry, praskliny v betonu nebo odložené tašky. Zbraň držená v ruce se v záběru nemůže dlouhodobě nehýbat, takže opakovaná detekce na stejných souřadnicích napříč snímky ukazuje na artefakt; ve specifikaci jich je zdokumentováno jedenáct, osm v trénovacích videích a tři v testovacím videu 005.

Míru, do jaké se lze na výchozí model spolehnout, dobře ilustruje přesnost předanotací. Ze 126 kandidátních boxů navržených při prahu 0,10 byl beze změny přijat jediný, 21 obsahovalo skutečnou tyč, ale s boxem 1,5–2,5krát větším, a 104 byly falešné detekce (předloktí, pěsti, stíny, brašny, boty, značky na asfaltu). Naopak 34 tyčí, které anotátoři našli, model nenavrhl vůbec.

Při přípravě dat se navíc ukázalo, že dříve zvětšené snímky MED nevznikly superrezolucí ESPCN, ale bikubickým zvětšením 4× (odchylka na úrovni šumu komprese JPEG); nové snímky byly proto zvětšeny stejně.

### Anotovaná data

Anotovány byly dvě sady. Řídká sada obsahuje 208 snímků vzorkovaných s frekvencí 2 snímky za sekundu z videí 001, 002 a 004; pouze 10 z nich obsahovalo zbraň (celkem 16 boxů) a 198 bylo negativních. Protože zbraně se objevují jen v krátkém okně bitky ve videu 004, byla doplněna hustá sada 44 snímků vzorkovaných s frekvencí přibližně 6 snímků za sekundu právě z tohoto okna, z toho 22 se zbraní (40 boxů). Hustá extrakce dala 2,5krát více boxů na pětkrát menším počtu snímků. Celkem bylo anotováno 252 snímků s 56 boxy třídy `melee` a 220 negativy.

### Nastavení dotrénování

Při sestavování dotrénovacího datasetu bylo nutné dbát na tři věci. První je způsob rozdělení na trénovací a validační část. Snímky husté sady jsou od sebe vzdáleny jen 0,17 s a sousední snímky jsou téměř totožné. Náhodné rozdělení by tak poslalo prakticky stejný snímek do tréninku i do validace a validace by měřila zapamatování, nikoli schopnost detekce. Data proto byla rozdělena časově: validaci tvoří souvislý blok snímků bitky ve videu 004 s ochrannou mezerou pěti snímků na každé straně, poslední čtvrtina videa 001 a celé video 002. Trénovací část má 209 snímků se 43 boxy, validační 41 snímků s 10 boxy; zbývající dva z 252 anotovaných snímků padly do ochranné mezery a nepoužily se.

Druhou věcí je zapomínání. Snímky MED obsahují jen tyče, nikoli pistole nebo nože, a model dotrénovaný pouze na nich by mohl ztratit schopnost rozpoznat ostatní třídy (tzv. katastrofické zapomínání [CITACE]). Do trénovacích dat bylo proto přimícháno 800 obrázků z původního webového datasetu.

Třetí věcí je rozsah učených parametrů. Dotrénování vycházelo z nejlepších vah hlavního tréninku s nízkou rychlostí učení (SGD, lr0 0,0005, 30 epoch, dávka 16, 960 px). Byly provedeny dva běhy, které se liší pouze tím, zda je zmrazen backbone sítě (vrstvy 0–10), takže tvoří čistou ablaci. Vyhodnocovaly se váhy z poslední epochy, nikoli váhy vybrané podle validace, protože při pouhých 10 validačních boxech by výběr epochy podle téže validace výsledek nadhodnotil.

### Výsledky

Výsledky obou variant shrnuje tabulka X.3. Dotrénování výrazně zlepšilo výkon v cílové doméně: mAP50 na validaci MED vzrostla z 0,138 na 0,498 při učení celé sítě a na 0,623 se zmrazeným backbonem a recall třídy `melee` stoupl z 0,10 na 0,40, resp. 0,50. Cenou je mírný pokles na původních webových datech.

**Tabulka X.3:** Ablace doménové adaptace. Validace MED pochází ze stejné bitky jako trénovací data.

| | Výchozí model | Dotrénování (celá síť) | Dotrénování (zmrazený backbone) |
|---|---|---|---|
| Validace MED: mAP50 | 0,138 | 0,498 | 0,623 |
| Validace MED: recall `melee` | 0,10 | 0,40 | 0,50 |
| Validace MED: precision `melee` | 0,84 | 0,97 | 0,99 |
| Původní validace: mAP50 | 0,777 | 0,745 | 0,755 |
| Původní validace: mAP50-95 | 0,525 | 0,477 | 0,490 |
| `gun` AP50 | 0,890 | 0,867 | 0,881 |
| `knife` AP50 | 0,610 | 0,600 | 0,590 |
| `melee` AP50 | 0,830 | 0,768 | 0,796 |

Zmrazení backbonu vyšlo lépe v obou ohledech: model je přesnější na CCTV a na původních datech zapomíná méně (pokles mAP50 o 0,022 místo 0,032). Dotrénování se zmrazeným backbonem bylo navíc zhruba třikrát rychlejší (přibližně 21 minut proti 61). Zapomínání se tím však neodstranilo, jen zmírnilo. Nejvíce postihlo třídu `melee` (0,830 → 0,796), což je pochopitelné: dotrénování posouvá význam třídy k dlouhým tyčím z MED a model přichází o část schopnosti rozpoznat baseballové pálky z webových fotografií. Lepší výsledek zmrazené varianty na MED je nejspíš důsledkem toho, že s menším počtem učených parametrů se méně přeučí na 43 trénovacích boxech. Rozdíl recallu 0,50 a 0,40 odpovídá ale jediné tyči z deseti, a je tedy na hranici šumu. Především však validace pochází ze stejné bitky, stejné kamery a stejných lidí jako trénovací data a měří interpolaci uvnitř jedné události, nikoli zobecnění na novou scénu. Do pipeline byla nasazena varianta se zmrazeným backbonem (dále v1).

## X.8 Nezávislé ověření, negativní příklady a kalibrace prahu

### Test na jiné bitce

Aby bylo možné ověřit skutečné zobecnění, bylo nutné najít další video se zbraněmi. Všech 28 dosud neprohlédnutých videí MED bylo prohlédnuto na kontaktních listech po šestnácti snímcích a kandidátní úseky v plném rozlišení; automatická metrika pohybu bitku od klidné scény neodlišila (průměrná změna mezi snímky 11,5 v klidné fázi proti 11,0 při bitce). Jediným dalším videem se zbraněmi je video 005: stejný dvůr a kamera jako u videa 004, ale jiná událost, jiní lidé v jiných pozicích a navíc jiné typy předmětů, například červené pruty.

Z videa 005 bylo hustě vzorkováno 48 snímků a anotováno stejnou metodikou. Agenti navrhli 79 boxů, z nichž bylo na kontaktních listech 12 vyloučeno (mimo jiné stín označený jako tyč s vysokou jistotou a pět téměř neviditelných rozmazaných čárek); zůstalo 67 ověřených boxů třídy `melee`. Na videu 005 se nikdy netrénovalo. Výsledky tří modelů shrnuje tabulka X.4; model v2 je popsán v následující části.

**Tabulka X.4:** Nezávislý test na videu 005 (48 snímků, 67 boxů `melee`). Precision a recall jsou uvedeny v bodě maxima F1.

| Model | AP50 | AP50-95 | Precision | Recall |
|---|---|---|---|---|
| Výchozí | 0,219 | 0,042 | 0,44 | 0,25 |
| Dotrénování (celá síť) | 0,640 | 0,302 | 0,87 | 0,51 |
| v1 (zmrazený backbone) | 0,602 | 0,281 | 0,89 | 0,49 |
| v2 (v1 + negativa) | 0,612 | 0,288 | 0,853 | 0,448 |

Dotrénování zobecňuje mimo bitku, na které se model učil: AP50 vzrostla na události, kterou model nikdy neviděl, zhruba trojnásobně (0,22 → 0,60–0,64). Jde o nejsilnější doklad přínosu doménové adaptace v této práci, protože na rozdíl od validace MED (10 boxů ze stejné bitky) jde o 67 boxů z nezávislé scény. Zároveň se zde obrátilo pořadí obou variant dotrénování: na nezávislém testu je mírně lepší učení celé sítě (0,640 proti 0,602). Rozdíl mezi nimi je tedy v rámci šumu a žádná z variant není v cílové doméně prokazatelně lepší. Rozhodujícím kritériem zůstalo menší zapomínání na původních datech, a nasazena proto zůstala varianta se zmrazeným backbonem. Obrázek X.2 shrnuje hlavní kompromis: výrazný zisk na CCTV za cenu malé ztráty na webových datech.

![Obrázek X.2: Zobecnění na CCTV vs. ztráta na webových datech. Doménová adaptace zvýšila AP50 třídy melee na nezávislé bitce z videa 005 téměř trojnásobně (0,219 → 0,612). Na původní webové validaci přitom ztratila jen 0,02–0,03.](figures/fig2_porovnani.png)

Křivky precision–recall na obrázku X.3 ukazují, že žádný z modelů nepřekročí recall 0,75: zbylé tyče jsou rozmazané nebo zakryté a model je nevidí ani s nízkou jistotou.

![Obrázek X.3: PR křivky třídy melee na testu z videa 005 (48 snímků, 67 ručně ověřených tyčí; na videu 005 se netrénovalo). Výchozí model drží precision nad 0,5 jen do recallu ~0,2, dotrénované modely do ~0,6. Žádný model nepřekročí recall 0,75: zbylé tyče jsou rozmazané nebo zakryté a model je nevidí ani s nízkou jistotou.](figures/fig3_pr_krivky.png)

Test zároveň ukázal, že dotrénovaný model je v nové doméně podsebevědomý. Maximum F1 leží u prahu jistoty přibližně 0,05; při tehdejším produkčním prahu 0,40 měl model v1 precision 1,00, ale recall jen 0,31. Prahu se věnuje poslední část této podkapitoly.

### Negativní příklady z rvaček bez zbraní

Nejčastější falešnou detekcí dotrénovaného modelu byla natažená holá paže. Průchod videi MED přitom odhalil scény, v nichž se paže tímto způsobem objevují bez jakékoli zbraně: rvačky pěstmi a pochod s bílými papíry zdviženými nad hlavou, které vizuálně připomínají bílou tyč nad hlavou. Z těchto scén vznikly dvě sady negativních snímků. Šedesát snímků z rvaček ve videích 012, 014 a 016 a z pochodu ve videu 025 bylo přidáno do trénovacích dat; 24 snímků z rvačky ve videu 017 a z pochodu ve videu 026 bylo ponecháno jen pro vyhodnocení.

Negativní snímek, na kterém by byla skrytá zbraň, by model učil opak toho, co má. Proto bylo ověřeno, že snímky zbraň skutečně neobsahují: všechny tři modely na nich byly spuštěny s prahem 0,10 a všech 19 jejich detekcí bylo prohlédnuto. Všechny byly falešné (natažené paže, stíny paží, člověk fotografující mobilem se zdviženými lokty, paže s papírem, nohy, čára na asfaltu).

Dotrénování modelu v2 bylo totožné s v1 (stejný zmrazený backbone, optimalizátor, rychlost učení, počet epoch, rozdělení i stejných 800 přimíchaných webových obrázků); jediným rozdílem bylo 60 negativních snímků navíc. Efekt negativ je tak změřen izolovaně. Výsledky shrnuje tabulka X.5 a obrázek X.4.

**Tabulka X.5:** Vliv negativních snímků z rvaček bez zbraní.

| | v1 | v2 (+ negativa) |
|---|---|---|
| Falešné poplachy na 24 testovacích negativech, práh 0,10 | 7 (ve 4 snímcích) | 3 (ve 2 snímcích) |
| Falešné poplachy na 24 testovacích negativech, práh 0,40 | 2 | 0 |
| Test 005: AP50 `melee` | 0,602 | 0,612 |
| Test 005: precision / recall | 0,885 / 0,493 | 0,853 / 0,448 |
| Původní validace: mAP50 | 0,755 | 0,761 |
| Původní validace: `melee` AP50 | 0,796 | 0,807 |
| Validace MED: mAP50 (10 boxů) | 0,623 | 0,609 |

![Obrázek X.4: Falešné poplachy na 24 testovacích snímcích rvaček bez zbraní v závislosti na prahu. Přidání 60 negativních snímků (v2) snížilo počet falešných poplachů zhruba na polovinu při každém prahu. Při nasazeném prahu 0,20 jsou to 2 poplachy proti 5 u v1.](figures/fig4_falesne_poplachy.png)

Negativa splnila svůj účel. Na scénách bez zbraní model hlásí zhruba poloviční počet falešných poplachů a při tehdejším produkčním prahu 0,40 žádný. Detekce zbraní se přitom nezhoršila: AP50 na nezávislém testu je prakticky shodná a posun precision a recallu je při 67 boxech v rámci šumu. Model navíc o něco méně zapomíná na původních datech. Pokles validace MED (0,623 → 0,609) je při deseti boxech rovněž šum. Testovacích negativ je však jen 24 a falešných poplachů jednotky, takže směr efektu je zřejmý, jeho velikost nikoli. Model v2 byl nasazen do pipeline.

### Kalibrace prahu

Práh 0,40 pro třídu `melee` byl odečten z křivky F1 výchozího modelu na webových datech. Dotrénovaný model však tyče v CCTV záběrech vidí s jistotou často pod 0,2, a práh 0,40 tak propouštěl jen asi šestinu tyčí. Při kontrole zapojení do pipeline to bylo patrné přímo: v deseti snímcích vrcholu bitky ve videu 005, kde jsou přibližně tři tyče na snímek, prošlo prahem pět detekcí. Práh bylo proto nutné přeladit pro model v2.

Validace MED má jen 10 boxů, a pro kalibraci je tedy nepoužitelná. Kalibrovalo se proto na časové polovině testu z videa 005: kalibrační část tvoří 19 snímků s 31 boxy, následuje jeden snímek jako ochranná mezera a ověřovací část tvoří 28 snímků s 34 boxy. Předem zvoleným kritériem bylo maximum F1 na kalibrační polovině. Falešné poplachy byly navíc měřeny na 24 testovacích snímcích rvaček bez zbraní a na 36 snímcích klidné scény z videí 001 a 002. Výsledky shrnuje tabulka X.6 a obrázek X.5.

**Tabulka X.6:** Kalibrace prahu třídy `melee` pro model v2. FP = falešné poplachy.

| Práh | Kalibrační P / R / F1 | Ověřovací P / R / F1 | FP rvačky (snímků z 24) | FP klid |
|---:|---|---|---:|---:|
| 0,027 | 0,91 / 0,65 / 0,75 | 0,63 / 0,46 / 0,53 | 11 (9) | 2 |
| 0,10 | 0,94 / 0,55 / 0,70 | 0,86 / 0,35 / 0,50 | 3 (2) | 1 |
| 0,15 | 0,97 / 0,55 / 0,70 | 0,94 / 0,32 / 0,48 | 2 (2) | 0 |
| **0,20** | **1,00 / 0,54 / 0,70** | **1,00 / 0,31 / 0,47** | **2 (2)** | **0** |
| 0,30 | 1,00 / 0,48 / 0,65 | 1,00 / 0,20 / 0,33 | 1 (1) | 0 |
| 0,40 (původní) | 1,00 / 0,46 / 0,63 | 1,00 / 0,17 / 0,29 | 0 | 0 |

![Obrázek X.5: Kalibrace prahu třídy melee pro model v2. Na kalibrační polovině testu z videa 005 je F1 mezi prahem 0,10 a 0,20 plochá. Zvolen byl nejnižší práh plošiny s precision 1,00 (0,20). Na ověřovací polovině tím recall stoupl z 0,17 (práh 0,40) na 0,31 a precision zůstala 1,00.](figures/fig5_kalibrace.png)

Maximum F1 vyšlo na samém okraji rozsahu, u prahu 0,027. Tento práh byl zamítnut: na snímcích bez zbraní dává 11 falešných poplachů v 9 z 24 snímků, tedy poplach zhruba v každém třetím snímku rvačky, což je pro dohledový systém nepoužitelné, a na ověřovací polovině precision klesá na 0,63. Na kalibrační polovině je F1 mezi prahy 0,10 a 0,20 plochá (0,70) a práh 0,20 je nejnižší bod této plošiny s precision 1,00. Proti původnímu prahu 0,40 se tím na ověřovací polovině zvýšil recall z 0,17 na 0,31 při zachované precision 1,00; na negativních snímcích rvaček přibyly dva falešné poplachy ve 24 snímcích, na klidné scéně žádný. AP50 třídy `melee` činí na kalibrační polovině 0,699 a na ověřovací 0,510; rozdíl mezi polovinami naznačuje, že i v rámci jednoho videa se obtížnost scény výrazně mění. Prahy tříd `gun` a `knife` se neměnily, protože pro ně v CCTV záběrech nejsou data.

Je nutné poctivě uvést, že změna kritéria (od maxima F1 k nejnižšímu prahu plošiny s plnou precision) padla až poté, co byla viděna ověřovací polovina pro prahy 0,027 a 0,40 a výsledky na negativních snímcích. Čísla ověřovací poloviny pro práh 0,20 proto nejsou zcela čistým testem. Samotná volba ale vychází z vlastností kalibrační poloviny (plošina F1 a precision 1,00) a z počtu falešných poplachů, nikoli z maxima na ověřovací polovině; to by naopak vedlo k prahu 0,027. Čistý odhad výkonu při zvoleném prahu by vyžadoval další anotované video.

Obrázek X.6 ukazuje typické chování modelu v2 při prahu 0,20: úspěšnou detekci, přehlédnutí tyče, která splývá s paží, a falešný poplach na rvačce bez zbraní.

![Obrázek X.6: Ukázky výstupu modelu v2 při prahu 0,20. a) obě tyče zachyceny; b) tyč zvednutá nad hlavu splývá s paží a model ji přehlédne; c) falešný poplach na rvačce bez zbraní: natažená paže s předmětem v ruce. Výběr je automatický: snímek s nejvíce správnými detekcemi, snímek s nejvíce přehlédnutími a falešný poplach s nejvyšší jistotou.](figures/fig6_ukazky.png)

### Časové potvrzení poplachů

Dosavadní vyhodnocení pracovalo s jednotlivými snímky. Systém ale zpracovává video a poplach vyhlašuje průběžně, takže rozhodující otázka zní, kolik falešných poplachů vznikne za minutu provozu. Proto byla celá videa zpracována stejně jako v pipeline, tedy při 5 snímcích za sekundu v nativním rozlišení. Simulace zachovává i hysterezi pipeline: jednou vyhlášený poplach trvá ještě 8 snímků (1,6 s). Jako negativa posloužilo sedm videí rvaček bez zbraní (5,5 minuty) a tři klidné scény (3,4 minuty). Pět z rvaček nebylo dosud použito vůbec, ani v tréninku, ani při ladění prahu.

Výsledek ukázal, že test na 24 snímcích falešné poplachy výrazně podhodnotil. Při prahu 0,20 a vyhlášení poplachu po jediné detekci vzniklo na rvačkách 27 poplachů, tedy zhruba pět za minutu, a na klidných scénách dalších 12. Prohlídka ukázala dva zdroje. Prvním jsou natažené paže, stejně jako dříve. Druhým, dosud neznámým, jsou namalované čáry skákacího panáka na asfaltu, které model hlásí opakovaně na stejném místě.

Falešné detekce jsou ale většinou ojedinělé záblesky v jednom snímku, kdežto skutečná tyč je vidět souvisle. Proto bylo testováno pravidlo, že poplach vznikne až tehdy, když detekce na stejném místě přetrvá k po sobě jdoucích snímků (tabulka X.7). Simulace vynechává kontrolu zápěstí, kterou pipeline provádí navíc, takže počty falešných poplachů jsou horním odhadem.

**Tabulka X.7:** Časové potvrzení poplachů (model v2, 5 snímků/s). „Bitka v poplachu“ je podíl snímků bitky ve videu 005, během nichž je poplach aktivní; zpoždění se měří od prvního snímku s tyčí.

| práh | k | bitka v poplachu | zpoždění | poplachy na rvačkách (5,5 min) | poplachy v klidu (3,4 min) |
|---:|---:|---:|---:|---:|---:|
| 0,40 | 1 (původní) | 90 % | 0,2 s | 10 | 8 |
| 0,20 | 1 | 93 % | 0,2 s | 27 | 12 |
| **0,20** | **2** | **72 %** | **0,4 s** | **3** | **5** |
| 0,20 | 3 | 69 % | 0,6 s | 2 | 1 |
| 0,10 | 2 | 90 % | 0,4 s | 10 | 11 |

Požadavek dvou po sobě jdoucích snímků snížil počet falešných poplachů na rvačkách z 27 na 3. Na pěti dosud nepoužitých videích klesl z 18 na 1. Skutečnou bitku přitom systém zachytí o 0,2 s později a poplach je aktivní 72 % jejího trvání místo 93 %. Proti původnímu nastavení (práh 0,40, jeden snímek) má nová kombinace méně falešných poplachů na rvačkách i v klidu (3 proti 10 a 5 proti 8). Zbývající poplachy v klidu působí hlavně namalované čáry, které časový filtr z principu nezachytí, protože jsou vidět trvale. Zkoušené potlačení detekcí, které se opakují na stejném místě, pomohlo jen málo (5 → 3) a nasazeno nebylo. Do pipeline bylo nasazeno potvrzení dvěma snímky; v rámci logiky ARMED se navíc vztahuje ke konkrétní osobě, protože zbraň musí být u jejího zápěstí v obou snímcích.

Čáry na zemi a natažené paže ukazují, jaká negativa má smysl doplnit. Pro další verzi modelu bylo proto vybráno 41 snímků z dalších videí bez zbraní (panika, dýmovnice, klidné scény), na kterých model v2 falešně hlásí tyč. Videa se nepřekrývají se žádnou testovací sadou.

### Zapojení do pipeline

Detektor je zapojen do modulu detekce podezřelých objektů vedle COCO detektoru zavazadel. Test zapojení, který nahrazuje jen detektor osob a odhad skeletu, potvrdil, že se dodržují prahy jednotlivých tříd a že detekce třídy `melee` vstupuje do logiky označování ozbrojených osob; to je ve výchozím nastavení vypnuté a zapíná se přepínačem. Zvětšení vstupního snímku detekci neovlivní: snímek 854 × 480 px a jeho bikubické zvětšení 4× dávají prakticky stejné detekce (14 proti 16 nad prahem 0,05, stejných 5 nad 0,40), protože model vstup stejně převádí na 960 px.

## X.9 Omezení a diskuse

Výsledky této části práce je třeba číst s vědomím několika podstatných omezení.

Prvním a nejzávažnějším je, že v CCTV záběrech je ověřena pouze třída `melee`. Videa MED neobsahují pistole ani nože, takže pro třídy `gun` a `knife` existují jen výsledky na webových datech, u nichž experiment s CCTV jasně ukázal, jak málo vypovídají o cílové doméně. Prahy těchto tříd nebyly pro dotrénovaný model přeladěny a třída `knife` má již na webových datech recall jen 0,587. Pro reálné nasazení by bylo nutné tyto třídy ověřit na záznamech z bezpečnostních kamer, které je obsahují.

Druhým omezením je úzkost cílových dat. Všechny pozitivní příklady z CCTV pocházejí ze dvou bitek na stejném dvoře, natočených stejnou kamerou za podobného osvětlení. Nezávislý test na videu 005 prokázal zobecnění na jinou událost, nikoli na jiné prostředí, úhel kamery, rozlišení nebo noční podmínky. Doménová adaptace tedy nezmenšila rozdíl mezi webovými fotografiemi a CCTV obecně, ale přizpůsobila model jedné konkrétní scéně.

Třetím omezením jsou malé testovací sady. Validace MED má 10 boxů (během tréninku kolísala mAP50 mezi epochami přibližně o 0,1), nezávislý test 67 boxů, po rozdělení pro kalibraci 31 a 34, a testovacích negativ je 24 snímků. Rozdíly mezi variantami dotrénování leží v pásmu šumu a jejich pořadí se mezi validací a testem dokonce obrátilo. Spolehlivé jsou jen velké efekty: zhruba trojnásobné zlepšení AP50 díky doménové adaptaci a zhruba poloviční počet falešných poplachů díky negativům. Ve snímcích, z nichž byly sporné objekty vyloučeny, zůstaly tyto objekty neoznačené a jejich detekce se počítá jako falešná, takže precision je spíše podhodnocená.

Čtvrtým omezením je kalibrace prahu: kritérium se změnilo poté, co byla viděna část ověřovacích dat, a hodnoty na ověřovací polovině proto nejsou zcela nezávislým odhadem. Ani po kalibraci model nezachytí většinu tyčí: na ověřovací polovině je recall 0,31. Detektor je tedy v současném stavu konzervativní, s velmi nízkým počtem falešných poplachů, ale s vysokým podílem přehlédnutí.

Za zmínku stojí i role jazykového modelu při anotaci. Ten sám byl zdrojem chyb (například stín označený jako tyč s vysokou jistotou) a kvalitu dat zajistila až struktura postupu: mřížka místo odhadu, strukturované verdikty místo přímého zápisu, odkládání nejistých případů a vizuální kontrola na kontaktních listech. Šedesát čtyři sporných případů (tyč proti paži nebo stínu) dosud čeká na posouzení člověkem.

Konečně trvá i mezera v dostupných datech. Holá tyč nebo hůl jako anotovaná třída ve veřejných datasetech chybí a třída `melee` je v původních datech ze 72,7 % tvořena baseballovými pálkami. Obecné řešení by vyžadovalo anotaci záznamů z více prostředí, například z datasetů násilí zachyceného bezpečnostními kamerami, jako jsou UCF-Crime, RWF-2000 nebo CCTV-Fights [CITACE].

## X.10 Shrnutí

Do pipeline byl doplněn detektor zbraní založený na modelu YOLO11l se třemi třídami (`gun`, `knife`, `melee`); zavazadla zůstala na COCO detektoru. Model byl natrénován na 24 340 obrázcích ze šesti veřejných datasetů, jejichž sestavení vyžadovalo deduplikaci a vizuální ověření tříd. Automaticky zvolený optimalizátor MuSGD na této úloze divergoval a byl nahrazen SGD. Na validaci webových dat dosáhl model mAP50 0,775, se slabinou u nožů, kterou nevyřešil oversampling ani ladění prahu.

Hlavním zjištěním je, že výkon na webových datech o použitelnosti v kamerových záznamech téměř nevypovídá. Na CCTV záběrech z MED model většinu zbraní nenašel, protože tyče a hole, které se v bitkách skutečně používají, ve veřejných datech chybějí. Doménová adaptace na 252 asistovaně anotovaných snímcích s časovým rozdělením dat, zmrazeným backbonem a přimícháním původních dat zvýšila AP50 třídy `melee` na nezávislé bitce z 0,219 na 0,602 a přidání negativních snímků z rvaček bez zbraní snížilo počet falešných poplachů zhruba na polovinu při AP50 0,612. Ztráta na původních webových datech činila jen 0,02–0,03. Po kalibraci prahu na 0,20 zachytí nasazený model na ověřovací části testu zhruba třetinu tyčí bez jediného falešného poplachu a na snímcích rvaček bez zbraní dává dva falešné poplachy ve 24 snímcích. Výsledky platí pro jednu kameru a jeden typ zbraně a jejich zobecnění na jiná prostředí zůstává otevřenou otázkou pro další práci.

---

## Poznámky pro autora

**Co ověřit:**

- Číslování: kapitola, obrázky a tabulky používají zástupné „X“; nahradit skutečným číslem kapitoly.
- Výchozí model na původní validaci: v podkapitole X.5 je mAP50 0,775 / mAP50-95 0,523 (hlavní vyhodnocení), v tabulce X.3 je 0,777 / 0,525 (opakované vyhodnocení při ablaci). Obojí je převzato ze zdroje; zvážit, zda rozdíl vysvětlit poznámkou pod čarou.
- Kalibrace na stejném objektu: technická zpráva uvádí box natrénovaného modelu 101 × 145 px, odhad hlavního modelu 40 × 68 px (příliš malý) a box subagenta s mřížkou 69 × 90 px (sedí); anotační specifikace naopak uvádí 40 × 68 px jako správnou velikost a 100 × 145 px od modelu. Proto do textu nebyla rozměry zařazena; před případným doplněním rozpor vyjasnit.
- Kalibrační a ověřovací polovina mají dohromady 65 boxů, test 67; zbylé 2 boxy jsou v ochranném snímku 907 (ověřeno). Případně zmínit v textu.
- Recall 0,31 při prahu 0,40 u v1 na celém testu 005 (X.8) a recall 0,31 při prahu 0,20 u v2 na ověřovací polovině jsou dvě různá čísla, která se náhodou shodují; zkontrolovat, že čtenář je nezamění.
- Popisek obrázku X.4 uvádí 5 falešných poplachů u v1 při prahu 0,20; tato hodnota je jen v datech obrázku, ne v textu zprávy (souhlasí s figures_data.json).
- Název „MuSGD“ a jeho vztah k metodě Muon ověřit v dokumentaci Ultralytics.

**Kde chybí citace [CITACE]:**

- Motion Emotion Dataset (MED) — v README projektu je odkaz na repozitář, ověřit primární publikaci.
- YOLO11 / Ultralytics (v README projektu je softwarová citace).
- RT-DETR (citace je v README projektu).
- Dataset COCO (Lin et al.).
- Roboflow Universe a jednotlivé použité datasety (u CC BY 4.0 je uvedení autora povinné).
- Optimalizátor Muon (a případně MuSGD v Ultralytics).
- Katastrofické zapomínání (catastrophic forgetting).
- Použitý jazykový model a agentní anotace (zvážit i obecnou literaturu o asistované/LLM anotaci).
- UCF-Crime, RWF-2000, CCTV-Fights.
- ESPCN je citován jinde v práci (README projektu), zde jen zmíněn.
