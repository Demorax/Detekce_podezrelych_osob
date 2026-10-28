"""
Převod kapitol psaných v Markdownu (styl weapon_training/docs/KAPITOLA_*.md) do Typstu
pro šablonu z typst.app (= nadpisy, #figure s table/image, @citace).

Pravidla:
  # X Název            -> = Název          (prefix "X", "X.1", "Y.2" se odstraní, čísluje Typst)
  ## X.1 Název          -> == Název
  ### Název             -> === Název
  **tučně** / *kurzíva* -> *tučně* / _kurzíva_
  `kód`                 -> `kód`
  [CITACE]              -> @TODO-citace   (doplnit klíč)
  [ODKAZ]               -> @TODO-odkaz
  ![popisek](cesta)     -> #figure(image("cesta"), caption: [popisek])
  **Tabulka X.1:** popis + markdown tabulka -> #figure(table(...), caption: [popis])
  sekce "## Poznámky pro autora" a úvodní blockquote (>) se vynechají (zůstanou jako komentář).
  Odkazy v textu „tabulka X.6“, „obrázek X.4“ zůstávají; nahradit @label ručně.

Použití:
    python docs/typst/md2typ.py weapon_training/docs/KAPITOLA_DETEKCE_ZBRANI.md docs/typst/kapitola_zbrane.typ
"""
import re
import sys
from pathlib import Path


def inline(s):
    s = s.replace("\\", "\\\\")
    s = re.sub(r"\*\*(.+?)\*\*", r"*\1*", s)
    s = re.sub(r"(?<![\w*])\*(?!\*)(.+?)(?<!\*)\*(?![\w*])", r"_\1_", s)
    s = s.replace("[CITACE]", "@TODO-citace").replace("[ODKAZ]", "@TODO-odkaz")
    s = re.sub(r"\[DOPLNIT([^\]]*)\]", r"// TODO: DOPLNIT\1", s)
    # znaky se zvláštním významem v Typstu mimo kód
    out, parts = [], re.split(r"(`[^`]*`)", s)
    for p in parts:
        if p.startswith("`"):
            out.append(p)
        else:
            p = re.sub(r"(?<!\\)#", r"\\#", p)
            p = re.sub(r"(?<!\\)\$", r"\\$", p)
            p = re.sub(r"(?<![\w\\])@(?!TODO)", r"\\@", p)
            p = re.sub(r"(?<!\\)<(?=[^@])", r"\\<", p)
            out.append(p)
    return "".join(out)


def table_block(lines, caption):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    ncol = max(len(r) for r in rows)
    align = []
    for j in range(ncol):
        col = [r[j] for r in rows[1:] if j < len(r) and r[j]]
        align.append("right" if col and all(re.fullmatch(r"[\d,.%± –×/*\-]+", c.replace("**", "")) for c in col) else "left")
    cells = lambda r: ", ".join(f"[{inline(c)}]" for c in r + [""] * (ncol - len(r)))
    body = "\n".join(f"    {cells(r)}," for r in rows[1:])
    return (f"#figure(\n  table(\n    columns: {ncol},\n    align: ({', '.join(align)}),\n"
            f"    table.header({cells(rows[0])}),\n{body}\n  ),\n  caption: [{inline(caption)}],\n)\n")


def convert(md):
    out, i = [], 0
    lines = md.splitlines()
    pending_caption = None
    while i < len(lines):
        l = lines[i]
        if l.startswith(">"):
            out.append("// " + l.lstrip("> ")); i += 1; continue
        m = re.match(r"^(#{1,3}) (?:[XY](?:\.\d+)?\s+)?(.*)", l)
        if m:
            if m.group(2).strip().lower().startswith("poznámky pro autora"):
                out.append("// ---- Poznámky pro autora (z Markdownu) ----")
                out += ["// " + x for x in lines[i + 1:]]; break
            out.append("=" * len(m.group(1)) + " " + inline(m.group(2))); i += 1; continue
        m = re.match(r"^!\[(.*)\]\((.*)\)\s*$", l)
        if m:
            cap = re.sub(r"^Obrázek [XY]\.\d+:\s*", "", m.group(1))
            out.append(f'#figure(\n  image("{m.group(2)}", width: 90%),\n  caption: [{inline(cap)}],\n)'); i += 1; continue
        m = re.match(r"^\*\*Tabulka [XY]\.\d+:\*\*\s*(.*)", l)
        if m:
            pending_caption = m.group(1); i += 1; continue
        if l.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(table_block(lines[i:j], pending_caption or "")); pending_caption = None; i = j; continue
        if l.strip() == "---":
            i += 1; continue
        m = re.match(r"^(\s*)[-*] (.*)", l)
        if m:
            out.append(f"{m.group(1)}- {inline(m.group(2))}"); i += 1; continue
        m = re.match(r"^(\s*)(\d+)\. (.*)", l)
        if m:
            out.append(f"{m.group(1)}+ {inline(m.group(3))}"); i += 1; continue
        out.append(inline(l)); i += 1
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(convert(src.read_text(encoding="utf-8")))
    print(f"-> {dst}")
