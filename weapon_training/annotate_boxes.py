#!/usr/bin/env python3
"""Anotace bounding boxů zbraní — korekce předanotací od modelu.

Doplňuje `label_skeletons.py`, který řeší jinou úlohu (tracking osoby a
labelování chování v čase). Tenhle nástroj anotuje OBJEKTY v jednotlivých
framech pro dotrénování detektoru.

Workflow:
    1. prelabel_frames.py         model předkreslí kandidáty (nízké conf)
    2. annotate_boxes.py  <-- TENTO SKRIPT: smazat špatné, dokreslit chybějící
    3. finetune                   dotrénování na opravených datech

Návrhová rozhodnutí:
  * Závislosti jen tkinter + PIL + numpy — na Macu chybí cv2, a GUI ho
    nepotřebuje. Vykresluje se přes PIL.
  * ZOOM je nutnost, ne komfort: zbraně v MED framech mají desítky pixelů
    na 4K obrázku. Bez přiblížení je nelze anotovat.
  * Rozlišuje se "frame zkontrolován a nic v něm není" od "frame ještě
    nezkontrolován". Prázdný .txt je platné pozadí pro trénink, a je to
    hodnotná informace — proto se stav vede v progress.json.
  * Průběžné ukládání po každé změně framu, žádné "ztratil jsem 2 hodiny práce".

Ovládání je vypsané v pravém panelu a v --help.

Použití:
    python annotate_boxes.py --dir annotate_med
"""
from __future__ import annotations

import argparse
import json
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# Material Design paleta, konzistentní s label_skeletons_improved.py
CLASS_COLORS = ["#F44336", "#FFC107", "#4CAF50", "#2196F3", "#9C27B0", "#FF5722"]
COL_SEL = "#FFFFFF"
COL_PRE = "#9E9E9E"
BG_DARK = "#212121"

HANDLE = 8          # px, oblast pro chycení rohu
MIN_BOX = 4         # px v obrazových souřadnicích


@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    pre: bool = False        # pochází z předanotace a nebyl potvrzen

    def norm(self) -> "Box":
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1
        return self

    def clip(self, w: int, h: int) -> "Box":
        """Omezí box na plochu obrázku.

        Predikce modelu mohou přesahovat okraj a normalizovaný YOLO formát
        navíc zavádí zaokrouhlovací chybu (typicky zlomky pixelu). Zároveň
        chrání před vytažením boxu mimo obraz při editaci.
        """
        self.norm()
        self.x1 = min(max(self.x1, 0.0), float(w))
        self.y1 = min(max(self.y1, 0.0), float(h))
        self.x2 = min(max(self.x2, 0.0), float(w))
        self.y2 = min(max(self.y2, 0.0), float(h))
        return self

    def to_yolo(self, w: int, h: int) -> str:
        cx, cy = (self.x1 + self.x2) / 2 / w, (self.y1 + self.y2) / 2 / h
        bw, bh = (self.x2 - self.x1) / w, (self.y2 - self.y1) / h
        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
        bw, bh = min(max(bw, 0), 1), min(max(bh, 0), 1)
        return f"{self.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    @staticmethod
    def from_yolo(line: str, w: int, h: int, pre: bool) -> "Box | None":
        p = line.split()
        if len(p) != 5:
            return None
        try:
            c = int(float(p[0])); cx, cy, bw, bh = (float(v) for v in p[1:])
        except ValueError:
            return None
        return Box(c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                   (cx + bw / 2) * w, (cy + bh / 2) * h, pre)


@dataclass
class FrameState:
    stem: str
    img_path: Path
    lbl_path: Path
    boxes: list[Box] = field(default_factory=list)
    loaded: bool = False
    dirty: bool = False


class Annotator:
    def __init__(self, root: tk.Tk, work: Path):
        self.root = root
        self.work = work
        self.img_dir = work / "images"
        self.lbl_dir = work / "labels"
        self.progress_path = work / "progress.json"

        cf = work / "classes.txt"
        self.classes = ([l.strip() for l in cf.read_text().splitlines() if l.strip()]
                        if cf.is_file() else ["gun", "knife", "melee"])

        stems = sorted({p.stem for p in self.img_dir.iterdir()
                        if p.suffix.lower() in IMG_EXT}) if self.img_dir.is_dir() else []
        if not stems:
            raise SystemExit(f"Žádné obrázky v {self.img_dir}")
        self.frames: list[FrameState] = []
        for s in stems:
            img = next((self.img_dir / (s + e) for e in IMG_EXT
                        if (self.img_dir / (s + e)).is_file()), None)
            if img:
                self.frames.append(FrameState(s, img, self.lbl_dir / (s + ".txt")))

        self.reviewed: set[str] = set()
        if self.progress_path.is_file():
            try:
                self.reviewed = set(json.loads(self.progress_path.read_text())
                                    .get("reviewed", []))
            except (ValueError, OSError):
                pass

        self.idx = 0
        # skoč na první nezkontrolovaný
        for i, f in enumerate(self.frames):
            if f.stem not in self.reviewed:
                self.idx = i
                break

        self.cur_cls = 0
        self.sel: int | None = None
        self.scale = 1.0
        self.off = [0.0, 0.0]
        self.pil: Image.Image | None = None
        self.tkimg: ImageTk.PhotoImage | None = None
        self._drag = None          # ('new'|'move'|'resize', data)
        self._pan = None
        self._undo: list[list[Box]] = []

        self._build_ui()
        self.root.after(60, self.load_frame)

    # ---------- data ----------
    @property
    def cf(self) -> FrameState:
        return self.frames[self.idx]

    def load_frame(self):
        f = self.cf
        self.pil = Image.open(f.img_path).convert("RGB")
        if not f.loaded:
            f.boxes = []
            if f.lbl_path.is_file():
                pre = f.stem not in self.reviewed
                for line in f.lbl_path.read_text().splitlines():
                    if line.strip():
                        b = Box.from_yolo(line, self.pil.width, self.pil.height, pre)
                        if b:
                            f.boxes.append(b.clip(self.pil.width, self.pil.height))
            f.loaded = True
        self.sel = None
        self._undo = []
        self.fit()

    def save_frame(self, mark_reviewed=True):
        f = self.cf
        if self.pil is None:
            return
        lines = [b.clip(self.pil.width, self.pil.height)
                  .to_yolo(self.pil.width, self.pil.height) for b in f.boxes]
        f.lbl_path.parent.mkdir(parents=True, exist_ok=True)
        f.lbl_path.write_text(("\n".join(lines) + "\n") if lines else "")
        if mark_reviewed:
            self.reviewed.add(f.stem)
            for b in f.boxes:
                b.pre = False
        f.dirty = False
        self.progress_path.write_text(json.dumps(
            {"reviewed": sorted(self.reviewed), "total": len(self.frames)}, indent=1))

    def push_undo(self):
        self._undo.append([Box(b.cls, b.x1, b.y1, b.x2, b.y2, b.pre) for b in self.cf.boxes])
        self._undo = self._undo[-40:]

    # ---------- souřadnice ----------
    def to_canvas(self, x, y):
        return x * self.scale + self.off[0], y * self.scale + self.off[1]

    def to_img(self, cx, cy):
        return (cx - self.off[0]) / self.scale, (cy - self.off[1]) / self.scale

    def fit(self):
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        if self.pil:
            self.scale = min(cw / self.pil.width, ch / self.pil.height)
            self.off = [(cw - self.pil.width * self.scale) / 2,
                        (ch - self.pil.height * self.scale) / 2]
        self.redraw()

    def zoom_at(self, cx, cy, factor):
        ix, iy = self.to_img(cx, cy)
        self.scale = min(max(self.scale * factor, 0.05), 30)
        self.off = [cx - ix * self.scale, cy - iy * self.scale]
        self.redraw()

    # ---------- UI ----------
    def _build_ui(self):
        self.root.title("Anotace boxů — zbraně")
        self.root.geometry("1500x900")
        self.root.configure(bg=BG_DARK)

        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main, bg=BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.status = tk.Label(left, text="", bg="#37474F", fg="white",
                               font=("Helvetica", 12, "bold"), anchor="w",
                               justify=tk.LEFT, padx=8, pady=4)
        self.status.pack(fill=tk.X)

        self.canvas = tk.Canvas(left, bg="#101010", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(main, bg="#263238", width=310)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        tk.Label(right, text="TŘÍDA (1-9 přepne)", bg="#263238", fg="#B0BEC5",
                 font=("Helvetica", 10, "bold")).pack(pady=(10, 4))
        self.cls_frame = tk.Frame(right, bg="#263238")
        self.cls_frame.pack(fill=tk.X, padx=10)
        self.cls_labels = []
        for i, cn in enumerate(self.classes):
            l = tk.Label(self.cls_frame, text=f" {i+1}  {cn}", bg="#263238",
                         fg=CLASS_COLORS[i % len(CLASS_COLORS)], anchor="w",
                         font=("Helvetica", 13))
            l.pack(fill=tk.X)
            self.cls_labels.append(l)

        tk.Label(right, text="BOXY V TOMTO FRAMU", bg="#263238", fg="#B0BEC5",
                 font=("Helvetica", 10, "bold")).pack(pady=(14, 4))
        self.box_list = tk.Listbox(right, bg="#1C262B", fg="#ECEFF1", height=9,
                                   highlightthickness=0, borderwidth=0,
                                   font=("Menlo", 10), selectbackground="#455A64")
        self.box_list.pack(fill=tk.X, padx=10)
        self.box_list.bind("<<ListboxSelect>>", self._on_list_select)

        tk.Label(right, text="OVLÁDÁNÍ", bg="#263238", fg="#B0BEC5",
                 font=("Helvetica", 10, "bold")).pack(pady=(14, 4))
        help_txt = (
            "tažení myší   nový box\n"
            "klik na box   vybrat\n"
            "tažení rohu   změnit velikost\n"
            "tažení uvnitř posunout\n"
            "1-9           třída (i vybraného)\n"
            "DEL / BKSP    smazat vybraný\n"
            "C             smazat vše ve framu\n"
            "\n"
            "D / →         další frame\n"
            "A / ←         předchozí frame\n"
            "X             frame je prázdný → další\n"
            "TAB           další NEzkontrolovaný\n"
            "\n"
            "kolečko       zoom pod kurzorem\n"
            "+ / -         zoom\n"
            "F             přizpůsobit\n"
            "prostřední/   posun obrazu\n"
            "  SHIFT+tažení\n"
            "\n"
            "CTRL+Z        zpět\n"
            "S             uložit\n"
            "Q / ESC       konec (uloží)\n"
        )
        tk.Label(right, text=help_txt, bg="#263238", fg="#CFD8DC", justify=tk.LEFT,
                 anchor="w", font=("Menlo", 10)).pack(fill=tk.X, padx=10)

        self.legend = tk.Label(right, text="", bg="#263238", fg="#78909C",
                               justify=tk.LEFT, anchor="w", font=("Menlo", 9))
        self.legend.pack(fill=tk.X, padx=10, pady=(10, 0))

        self.progress = ttk.Progressbar(right, mode="determinate")
        self.progress.pack(fill=tk.X, padx=10, pady=12)

        c = self.canvas
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_motion)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<ButtonPress-2>", self.on_pan_start)
        c.bind("<B2-Motion>", self.on_pan_move)
        c.bind("<ButtonPress-3>", self.on_pan_start)
        c.bind("<B3-Motion>", self.on_pan_move)
        c.bind("<Shift-ButtonPress-1>", self.on_pan_start)
        c.bind("<Shift-B1-Motion>", self.on_pan_move)
        c.bind("<MouseWheel>", self.on_wheel)
        c.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, 1.15))
        c.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 1 / 1.15))
        c.bind("<Configure>", lambda e: self.redraw())
        self.root.bind("<Key>", self.on_key)

    # ---------- kreslení ----------
    def redraw(self):
        c = self.canvas
        c.delete("all")
        if self.pil is None:
            return
        cw, ch = max(c.winfo_width(), 1), max(c.winfo_height(), 1)
        dw, dh = int(self.pil.width * self.scale), int(self.pil.height * self.scale)
        if dw < 1 or dh < 1:
            return
        # vykresli jen viditelnou část — u 4K obrázku při zoomu 10x by celý
        # resize spolykal stovky MB
        vx0 = max(0, int(-self.off[0] / self.scale))
        vy0 = max(0, int(-self.off[1] / self.scale))
        vx1 = min(self.pil.width, int((cw - self.off[0]) / self.scale) + 1)
        vy1 = min(self.pil.height, int((ch - self.off[1]) / self.scale) + 1)
        if vx1 <= vx0 or vy1 <= vy0:
            return
        crop = self.pil.crop((vx0, vy0, vx1, vy1))
        tw = max(1, int((vx1 - vx0) * self.scale))
        th = max(1, int((vy1 - vy0) * self.scale))
        resample = Image.NEAREST if self.scale > 2 else Image.BILINEAR
        self.tkimg = ImageTk.PhotoImage(crop.resize((tw, th), resample))
        px, py = self.to_canvas(vx0, vy0)
        c.create_image(px, py, image=self.tkimg, anchor="nw")

        for i, b in enumerate(self.cf.boxes):
            x1, y1 = self.to_canvas(b.x1, b.y1)
            x2, y2 = self.to_canvas(b.x2, b.y2)
            col = COL_PRE if b.pre else CLASS_COLORS[b.cls % len(CLASS_COLORS)]
            wd = 3 if i == self.sel else 2
            dash = (4, 3) if b.pre else ()
            c.create_rectangle(x1, y1, x2, y2, outline=col, width=wd, dash=dash)
            name = self.classes[b.cls] if b.cls < len(self.classes) else f"cls{b.cls}"
            c.create_text(x1 + 2, y1 - 9, text=name + (" ?" if b.pre else ""),
                          fill=col, anchor="w", font=("Helvetica", 11, "bold"))
            if i == self.sel:
                for hx, hy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                    c.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4,
                                       outline=COL_SEL, fill=col, width=1)
        self.refresh_panels()

    def refresh_panels(self):
        f = self.cf
        done = f.stem in self.reviewed
        n_pre = sum(1 for b in f.boxes if b.pre)
        mark = "✓ zkontrolován" if done else ("● předanotace" if n_pre else "○ nový")
        self.status.config(
            text=f"[{self.idx+1}/{len(self.frames)}]  {f.stem}   {mark}"
                 f"   |  boxů: {len(f.boxes)}   |  třída: {self.classes[self.cur_cls]}"
                 f"   |  zoom {self.scale:.2f}×   |  hotovo "
                 f"{len(self.reviewed)}/{len(self.frames)}"
                 + ("   *neuloženo" if f.dirty else ""))
        for i, l in enumerate(self.cls_labels):
            l.config(bg="#455A64" if i == self.cur_cls else "#263238")
        self.box_list.delete(0, tk.END)
        for i, b in enumerate(f.boxes):
            nm = self.classes[b.cls] if b.cls < len(self.classes) else f"cls{b.cls}"
            self.box_list.insert(
                tk.END, f"{i:2d} {nm:<6} {int(b.x2-b.x1):4d}×{int(b.y2-b.y1):<4d}"
                        f"{' ?' if b.pre else ''}")
        if self.sel is not None and self.sel < len(f.boxes):
            self.box_list.selection_clear(0, tk.END)
            self.box_list.selection_set(self.sel)
        per = {}
        for fr in self.frames:
            if fr.loaded:
                for b in fr.boxes:
                    nm = self.classes[b.cls] if b.cls < len(self.classes) else "?"
                    per[nm] = per.get(nm, 0) + 1
        self.legend.config(
            text="šedě přerušovaně = předanotace,\nkterou jsi ještě nepotvrdil\n\n"
                 "anotace v načtených framech:\n  "
                 + ("\n  ".join(f"{k}: {v}" for k, v in sorted(per.items())) or "—"))
        self.progress["maximum"] = len(self.frames)
        self.progress["value"] = len(self.reviewed)

    # ---------- myš ----------
    def _hit(self, cx, cy):
        for i in reversed(range(len(self.cf.boxes))):
            b = self.cf.boxes[i]
            x1, y1 = self.to_canvas(b.x1, b.y1)
            x2, y2 = self.to_canvas(b.x2, b.y2)
            for cn, (hx, hy) in enumerate(((x1, y1), (x2, y1), (x1, y2), (x2, y2))):
                if abs(cx - hx) <= HANDLE and abs(cy - hy) <= HANDLE:
                    return i, cn
            if min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2):
                return i, None
        return None, None

    def on_press(self, e):
        if e.state & 0x0001:      # Shift → pan
            return
        i, corner = self._hit(e.x, e.y)
        ix, iy = self.to_img(e.x, e.y)
        if i is None:
            self.push_undo()
            self.cf.boxes.append(Box(self.cur_cls, ix, iy, ix, iy))
            self.sel = len(self.cf.boxes) - 1
            self._drag = ("new", None)
        elif corner is not None:
            self.sel = i
            self.push_undo()
            self._drag = ("resize", corner)
        else:
            self.sel = i
            self.push_undo()
            self._drag = ("move", (ix - self.cf.boxes[i].x1, iy - self.cf.boxes[i].y1))
        self.cf.dirty = True
        self.redraw()

    def on_motion(self, e):
        if self._pan:
            return self.on_pan_move(e)
        if not self._drag or self.sel is None:
            return
        b = self.cf.boxes[self.sel]
        ix, iy = self.to_img(e.x, e.y)
        kind, data = self._drag
        if kind == "new":
            b.x2, b.y2 = ix, iy
        elif kind == "resize":
            if data == 0:   b.x1, b.y1 = ix, iy
            elif data == 1: b.x2, b.y1 = ix, iy
            elif data == 2: b.x1, b.y2 = ix, iy
            else:           b.x2, b.y2 = ix, iy
        else:
            w, h = b.x2 - b.x1, b.y2 - b.y1
            b.x1, b.y1 = ix - data[0], iy - data[1]
            b.x2, b.y2 = b.x1 + w, b.y1 + h
        b.pre = False
        self.redraw()

    def on_release(self, e):
        self._pan = None
        if self._drag and self.sel is not None:
            b = self.cf.boxes[self.sel].norm()
            if (b.x2 - b.x1) < MIN_BOX or (b.y2 - b.y1) < MIN_BOX:
                # omylem kliknutí bez tažení → box nevytvářet
                del self.cf.boxes[self.sel]
                self.sel = None
        self._drag = None
        self.redraw()

    def on_pan_start(self, e):
        self._pan = (e.x, e.y, self.off[0], self.off[1])

    def on_pan_move(self, e):
        if not self._pan:
            return
        x0, y0, ox, oy = self._pan
        self.off = [ox + (e.x - x0), oy + (e.y - y0)]
        self.redraw()

    def on_wheel(self, e):
        self.zoom_at(e.x, e.y, 1.15 if e.delta > 0 else 1 / 1.15)

    def _on_list_select(self, _e):
        s = self.box_list.curselection()
        if s:
            self.sel = s[0]
            self.redraw()

    # ---------- klávesy ----------
    def goto(self, i, mark_empty_reviewed=False):
        if not (0 <= i < len(self.frames)):
            return
        self.save_frame(mark_reviewed=mark_empty_reviewed or bool(self.cf.boxes)
                        or self.cf.stem in self.reviewed)
        self.idx = i
        self.load_frame()

    def on_key(self, e):
        k = e.keysym.lower()
        ctrl = bool(e.state & 0x0004) or bool(e.state & 0x0008)

        if ctrl and k == "z":
            if self._undo:
                self.cf.boxes = self._undo.pop()
                self.sel = None
                self.cf.dirty = True
                self.redraw()
            return
        if k in ("q", "escape"):
            self.save_frame(mark_reviewed=bool(self.cf.boxes))
            n = len(self.reviewed)
            messagebox.showinfo("Uloženo",
                                f"Zkontrolováno {n}/{len(self.frames)} framů.\n"
                                f"Labely: {self.lbl_dir}\nStav: {self.progress_path.name}")
            self.root.quit()
            return
        if k == "s":
            self.save_frame(mark_reviewed=bool(self.cf.boxes))
            self.redraw()
            return
        if k in ("d", "right"):
            self.goto(self.idx + 1)
            return
        if k in ("a", "left"):
            self.goto(self.idx - 1)
            return
        if k == "x":                     # explicitně prázdný frame = platné pozadí
            self.push_undo()
            self.cf.boxes = []
            self.cf.dirty = True
            self.goto(self.idx + 1, mark_empty_reviewed=True)
            return
        if k == "tab":
            for j in list(range(self.idx + 1, len(self.frames))) + list(range(self.idx)):
                if self.frames[j].stem not in self.reviewed:
                    self.goto(j)
                    return
            messagebox.showinfo("Hotovo", "Všechny framy jsou zkontrolované.")
            return
        if k in ("delete", "backspace"):
            if self.sel is not None and self.sel < len(self.cf.boxes):
                self.push_undo()
                del self.cf.boxes[self.sel]
                self.sel = None
                self.cf.dirty = True
                self.redraw()
            return
        if k == "c":
            if self.cf.boxes and messagebox.askyesno("Smazat", "Smazat všechny boxy ve framu?"):
                self.push_undo()
                self.cf.boxes = []
                self.sel = None
                self.cf.dirty = True
                self.redraw()
            return
        if k == "f":
            self.fit()
            return
        if k in ("plus", "equal", "kp_add"):
            self.zoom_at(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, 1.25)
            return
        if k in ("minus", "kp_subtract"):
            self.zoom_at(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, 1 / 1.25)
            return
        if k.isdigit() and k != "0":
            ci = int(k) - 1
            if ci < len(self.classes):
                self.cur_cls = ci
                if self.sel is not None and self.sel < len(self.cf.boxes):
                    self.push_undo()
                    self.cf.boxes[self.sel].cls = ci
                    self.cf.boxes[self.sel].pre = False
                    self.cf.dirty = True
                self.redraw()
            return


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Anotace bounding boxů s korekcí předanotací od modelu.")
    ap.add_argument("--dir", required=True,
                    help="adresář z prelabel_frames.py (images/, labels/, classes.txt)")
    args = ap.parse_args()

    work = Path(args.dir).expanduser().resolve()
    if not (work / "images").is_dir():
        raise SystemExit(f"Chybí {work}/images — spusť nejdřív prelabel_frames.py")

    root = tk.Tk()
    app = Annotator(root, work)
    root.mainloop()
    try:
        app.save_frame(mark_reviewed=False)
    except Exception:
        pass
    print(f"zkontrolováno {len(app.reviewed)}/{len(app.frames)} framů")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
