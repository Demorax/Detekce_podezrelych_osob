# Vylepšení Skeleton Labeler - Srovnání

## 📊 Srovnání verzí

| Feature | Původní verze | Vylepšená verze |
|---------|--------------|-----------------|
| **Vizualizace** | OpenCV (cv2.imshow) | Matplotlib (anti-aliased, profesionální) |
| **UI Framework** | OpenCV okno + text overlays | Tkinter + Matplotlib canvas |
| **Organizace ovládání** | Všechny instrukce na jednom místě | Záložky pro každý režim |
| **Status informace** | Text na videu | Dedikovaný status bar |
| **Barvy skeletů** | BGR tuple (OpenCV formát) | Matplotlib rainbow colormap |
| **Sledovaná osoba** | Zelená/Žlutá/Červená | Hex barvy + emotikony (✓/?/✗) |
| **Instrukce** | Anglicky | Česky |
| **Segmenty display** | Posledních 5 na videu | Dedikovaný text widget |
| **Save feedback** | Pouze konzole | MessageBox + konzole |
| **Dokumentace** | Minimální | Kompletní docstringy |

## 🎯 Klíčové výhody nové verze

### 1. **Lepší vizuální kvalita**
```python
# Původní (OpenCV)
cv2.line(img, pt1, pt2, (255, 0, 0), 2)  # Aliased čáry
cv2.circle(img, center, 5, color, -1)     # Aliased kruhy

# Nová (Matplotlib)
ax.plot([x1, x2], [y1, y2], color='#4CAF50', linewidth=2, alpha=0.8)  # Anti-aliased
ax.scatter(x, y, s=80, edgecolors='white', linewidth=2, zorder=10)    # Hladké kruhy
```

### 2. **Čistší UI struktura**
```
Původní:                          Nová:
┌─────────────────────┐          ┌──────────────────────────┐
│ Video + Overlays    │          │ Matplotlib Canvas        │
│ - Top bar           │          │ (Čistý obraz)            │
│ - Footer s textem   │          ├──────────────────────────┤
│                     │          │ Status Bar               │
│                     │          │ - Frame | Mode | Play    │
│                     │          ├──────────────────────────┤
└─────────────────────┘          │ Tabs (Select|Track|Label)│
                                 │ - Instrukce pro každý    │
                                 ├──────────────────────────┤
                                 │ Recent Segments          │
                                 └──────────────────────────┘
```

### 3. **Lepší workflow**
- **Status bar** - okamžitě vidíš, kde jsi a co děláš
- **Záložky** - instrukce pro aktuální režim, ne všechny najednou
- **Segments panel** - přehled uložených segmentů bez překrývání videa
- **MessageBox** - vizuální potvrzení při uložení

### 4. **Profesionální vzhled**
- Hex barvy místo BGR tuple
- Material Design color palette:
  - Walking: `#4CAF50` (Green 500)
  - Suspicious: `#FFC107` (Amber 500)
  - Running: `#F44336` (Red 500)
- Unicode symboly: ▶ ⏸ ✓ ? ✗ 🏷

## 🚀 Jak přejít na novou verzi

### Metoda 1: Test vedle sebe
```bash
# Starý labeler
python label_skeletons.py

# Nový labeler
python test_improved_labeler.py
```

### Metoda 2: Nahradit přímo
```bash
# Záloha starého
mv label_skeletons.py label_skeletons_old.py

# Použít nový jako hlavní
mv label_skeletons_improved.py label_skeletons.py
```

## 📝 Stejné funkce (beze změny)

✅ Stejné klávesové zkratky (SPACE, A/D, Y/N, R/P, 1/2/3, L/T, S)
✅ Stejný tracking algoritmus (centroid matching)
✅ Stejný formát výstupu (JSON + NPZ)
✅ Stejná podpora pro 0-9 a A-Z person selection
✅ Stejné tři režimy (select, track, label)
✅ Stejné behavior kategorie (walking, suspicious, running)

## 🎨 Vizuální příklady

### Status Bar
```
┌────────────────────────────────────────────────────────────┐
│ Frame: 15/98 | Mode: TRACK - Kontrola sledování | ⏸ PAUSED │
│ Saved segments: 3                                          │
│ 🏷 Labeling: suspicious (from frame 10)                    │
└────────────────────────────────────────────────────────────┘
```

### Tracking Status Colors
```
🟢 TRACKED ✓  - Zelená  (#4CAF50) - Potvrzené sledování
🟡 TRACKED ?  - Žlutá   (#FFC107) - Nejisté, potřebuje kontrolu
🔴 LOST ✗     - Červená (#F44336) - Ztraceno
⚪ Person X   - Šedá    (#757575) - Nesledovaná osoba (dimmed)
```

### Tabs Organization
```
┌─ Selection Mode ─┬─ Tracking Mode ─┬─ Labeling Mode ─┐
│                  │                 │                 │
│ 0-9: Osoba 0-9   │ SPACE: Play     │ SPACE: Play     │
│ A-Z: Osoba 10-35 │ A/D: Navigace   │ A/D: Navigace   │
│ ESC: Exit        │ Y: Potvrdit     │ 1/2/3: Labels   │
│                  │ N: Špatná osoba │ ENTER: Konec    │
│                  │ R: Znovu vybrat │ S: Uložit       │
│                  │ P: Nová osoba   │ T: Zpět         │
│                  │ L: Začít label  │                 │
└──────────────────┴─────────────────┴─────────────────┘
```

## 🔧 Technické detaily

### Dependencies
Nová verze používá:
- `tkinter` (již nainstalováno s Pythonem)
- `matplotlib` (již máš)
- `numpy`, `cv2`, `scipy` (již máš)

**Žádné nové závislosti!** ✅

### Performance
- Matplotlib rendering: ~10 FPS při přehrávání
- Původní OpenCV: ~30 FPS při přehrávání

**Poznámka**: Pro labeling (kde je pauza mezi framy) je 10 FPS více než dostatečné.

### Kompatibilita
- ✅ Windows, Linux, macOS
- ✅ Stejný formát výstupu
- ✅ Zpětně kompatibilní s existujícími daty
- ✅ Stejné skeleton soubory (.npy)

## 📚 Dokumentace

Všechny metody nyní mají kompletní docstringy:

```python
def draw_skeleton_matplotlib(self, ax, keypoints, color, linewidth=2, alpha=0.8, label=None):
    """
    Vykreslí skeleton pomocí matplotlib (lepší kvalita než OpenCV)

    Args:
        ax: matplotlib axes
        keypoints: numpy array (17, 2)
        color: barva (RGB tuple nebo hex)
        linewidth: šířka čar
        alpha: průhlednost
        label: volitelný label pro osobu
    """
```

## 🎯 Doporučení

**Použij novou verzi pokud:**
- ✅ Chceš profesionálnější vzhled
- ✅ Potřebuješ lepší přehled o stavu
- ✅ Děláš prezentaci nebo screenshots
- ✅ Preferuješ české instrukce

**Zůstaň u staré verze pokud:**
- ⚠️ Potřebuješ maximální rychlost (30 FPS vs 10 FPS)
- ⚠️ Máš hodně slabý počítač
- ⚠️ Nechceš riskovat změnu workflow

---

**Tip**: Vyzkoušej nejdřív test skript a podívej se, jestli ti to vyhovuje!
```bash
python test_improved_labeler.py
```
