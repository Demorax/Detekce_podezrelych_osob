#!/usr/bin/env python3
"""Grafy a ukázky detekcí do textu práce (TRAINING_REPORT.md, docs/figures/).

Vše se počítá znovu z vah a anotací, žádná čísla se neopisují ručně, kromě
AP50 na původní validaci. Ta vyžaduje 24k-obrázkový dataset, který je jen
na Sparku; hodnoty jsou z TRAINING_REPORT §5.1 a §6d.

Výstup (PDF pro sazbu + PNG pro náhled):
    fig1_trenink.*          průběh tréninku: SGD vs. divergující MuSGD
    fig2_porovnani.*        tři modely: CCTV test 005 vs. původní validace
    fig3_pr_krivky.*        PR křivky melee na testu 005
    fig4_falesne_poplachy.* falešné poplachy na framech bez zbraní vs. práh (v1 / v2)
    fig5_kalibrace.*        P / R / F1 vs. práh na kalibrační a ověřovací polovině
    fig6_ukazky.*           správná detekce, přehlédnutá tyč, falešný poplach
    figures_data.json       všechna čísla, ze kterých grafy vznikly

Použití:
    python make_thesis_figures.py --device mps
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from PIL import Image

HERE = Path(__file__).resolve().parent
ART = HERE / "docs/artifacts"
OUT = HERE / "docs/figures"
MELEE = 2
CONF_PROD = 0.20     # nasazený práh melee (§6e)
CONF_OLD = 0.40      # původní práh

MODELS = {           # pevné pořadí = pevná barva (kategorická paleta, validováno)
    "výchozí": ("models/weapon_detection_yolo11l/best.pt", "#2a78d6", "-"),
    "v1 (doménová adaptace)": ("models/weapon_detection_yolo11l_med/best.pt", "#eb6834", "--"),
    "v2 (+ negativa)": ("models/weapon_detection_yolo11l_med_v2/best.pt", "#1baf7a", "-"),
}
ORIG_VAL_MELEE_AP50 = {"výchozí": 0.830, "v1 (doménová adaptace)": 0.796, "v2 (+ negativa)": 0.807}

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "legend.frameon": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 200,
})
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def cz(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}".replace(".", ",")


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")


def frame_idx(p: Path) -> int:
    return int(p.stem.split("_")[-1])


def read_csv(p: Path) -> dict[str, list[float]]:
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return {k.strip(): [float(r[k]) for r in rows] for k in rows[0]}


def val_set(model, files: list[Path], device: str, td: Path, tag: str):
    lst = td / f"{tag}.txt"
    lst.write_text("\n".join(map(str, files)) + "\n")
    yml = td / f"{tag}.yaml"
    yml.write_text(f"train: {lst}\nval: {lst}\nnc: 3\nnames: [gun, knife, melee]\n")
    r = model.val(data=str(yml), split="val", imgsz=960, batch=4, conf=0.001,
                  device=device, plots=False, verbose=False)
    i = list(r.box.ap_class_index).index(MELEE)
    return {"ap50": float(r.box.ap50[i]), "px": r.box.px.tolist(),
            "prec_at_recall": r.box.prec_values[i].tolist(),
            "p": r.box.p_curve[i].tolist(), "r": r.box.r_curve[i].tolist(),
            "f1": r.box.f1_curve[i].tolist()}


def melee_confs(model, files: list[Path], device: str) -> list[list[float]]:
    out = []
    for p in files:
        b = model.predict(str(p), conf=0.01, imgsz=960, device=device, verbose=False)[0].boxes
        out.append([float(c) for c, k in zip(b.conf, b.cls) if int(k) == MELEE])
    return out


# ------------------------------------------------------------------ obr. 1
def fig_training(data: dict) -> None:
    sgd = read_csv(ART / "weapons_sgd_lr005/results.csv")
    mu = read_csv(ART / "weapons_yolo11l_960_w6_musgd_diverged/results.csv")
    data["trenink"] = {"sgd_map50": sgd["metrics/mAP50(B)"], "musgd_map50": mu["metrics/mAP50(B)"],
                       "sgd_val_cls": sgd["val/cls_loss"], "musgd_val_cls": mu["val/cls_loss"]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, key, ylab in ((a, "metrics/mAP50(B)", "mAP50 (validace)"),
                          (b, "val/cls_loss", "klasifikační loss (validace)")):
        ax.plot(sgd["epoch"], sgd[key], color="#2a78d6", label="SGD, lr0 = 0,005")
        ax.plot(mu["epoch"], mu[key], color="#eb6834", ls="--", marker="o", ms=4,
                label="MuSGD (optimizer='auto')")
        ax.set_xlabel("epocha"); ax.set_ylabel(ylab)
        ax.yaxis.set_major_formatter(COMMA)
    a.set_ylim(0, None)
    a.annotate("divergence", xy=(3, mu["metrics/mAP50(B)"][2]), xytext=(10, 0.04),
               color=INK2, fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    b.annotate("val loss 2,5×", xy=(3, mu["val/cls_loss"][2]), xytext=(15, mu["val/cls_loss"][2] - 0.4),
               color=INK2, fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    a.legend(loc="center right", fontsize=8)
    fig.suptitle("Průběh tréninku na webových datech (24 340 obrázků)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig1_trenink")


# ------------------------------------------------------------------ obr. 2 + 3
def fig_models(test: dict, data: dict) -> None:
    names = list(MODELS)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    panels = (("CCTV: nezávislá bitka (video 005)", [test[n]["ap50"] for n in names]),
              ("Webová data: původní validace", [ORIG_VAL_MELEE_AP50[n] for n in names]))
    for ax, (title, vals) in zip(axes, panels):
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, height=0.55, color=[MODELS[n][1] for n in names], edgecolor=SURF, lw=2)
        for yy, v in zip(y, vals):
            ax.text(v + 0.015, yy, cz(v, 3), va="center", fontsize=8, color=INK)
        ax.set_yticks(y, names); ax.set_xlim(0, 1); ax.set_title(title, loc="left")
        ax.xaxis.set_major_formatter(COMMA); ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
        ax.set_xlabel("AP50 třídy melee")
    fig.tight_layout()
    save(fig, "fig2_porovnani")

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for n in names:
        _, col, ls = MODELS[n]
        ax.plot(test[n]["px"], test[n]["prec_at_recall"], color=col, ls=ls,
                label=f"{n}  (AP50 {cz(test[n]['ap50'], 3)})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.xaxis.set_major_formatter(COMMA); ax.yaxis.set_major_formatter(COMMA)
    ax.set_title("PR křivky třídy melee — test 005 (67 tyčí)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), fontsize=8)
    fig.tight_layout()
    save(fig, "fig3_pr_krivky")
    data["test005"] = {n: {"ap50": test[n]["ap50"], "ap50_orig_val": ORIG_VAL_MELEE_AP50[n]} for n in names}


# ------------------------------------------------------------------ obr. 4
def fig_false_alarms(neg: dict, n_frames: int, data: dict) -> None:
    ts = np.linspace(0.02, 0.6, 117)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    data["falesne_poplachy"] = {"framu": n_frames}
    for n in ("v1 (doménová adaptace)", "v2 (+ negativa)"):
        _, col, ls = MODELS[n]
        fp = [sum(sum(c >= t for c in fc) for fc in neg[n]) for t in ts]
        ax.step(ts, fp, where="post", color=col, ls=ls, label=n)
        data["falesne_poplachy"][n] = {cz(t): sum(sum(c >= t for c in fc) for fc in neg[n])
                                      for t in (0.10, 0.20, 0.40)}
    for t, lab in ((CONF_PROD, "nasazený práh 0,20"), (CONF_OLD, "původní 0,40")):
        ax.axvline(t, color=INK2, lw=0.8, ls=":")
        ax.text(t + 0.006, ax.get_ylim()[1] * 0.92, lab, fontsize=7.5, color=INK2)
    ax.set_xlabel("práh confidence"); ax.set_ylabel("falešné poplachy melee")
    ax.xaxis.set_major_formatter(COMMA)
    ax.set_title(f"Framy rvaček bez zbraní ({n_frames} framů, testovací)", loc="left")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig4_falesne_poplachy")


# ------------------------------------------------------------------ obr. 5
def fig_calibration(cal: dict, hold: dict, data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    metrics = (("p", "precision", "#2a78d6", "-"), ("r", "recall", "#eb6834", "--"),
               ("f1", "F1", "#1baf7a", "-"))
    for ax, (res, title) in zip(axes, ((cal, "kalibrační polovina (framy ≤ 902)"),
                                       (hold, "ověřovací polovina (framy ≥ 912)"))):
        px = np.array(res["px"])
        for key, lab, col, ls in metrics:
            ax.plot(px, res[key], color=col, ls=ls, label=lab)
        for t in (CONF_PROD, CONF_OLD):
            ax.axvline(t, color=INK2, lw=0.8, ls=":")
        ax.set_xlim(0, 0.6); ax.set_ylim(0, 1.02)
        ax.set_title(title, loc="left"); ax.set_xlabel("práh confidence")
        ax.xaxis.set_major_formatter(COMMA); ax.yaxis.set_major_formatter(COMMA)
    axes[0].text(CONF_PROD + 0.008, 0.06, "0,20", fontsize=7.5, color=INK2)
    axes[0].text(CONF_OLD + 0.008, 0.06, "0,40", fontsize=7.5, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, fontsize=8)
    fig.suptitle("Kalibrace prahu melee na testu 005 (model v2)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig5_kalibrace")
    pick = lambda res, t: {k: res[k][int(np.searchsorted(res["px"], t))] for k in ("p", "r", "f1")}
    data["kalibrace"] = {h: {cz(t): pick(res, t) for t in (0.027, 0.10, 0.20, 0.40)}
                         for h, res in (("kalibracni", cal), ("overovaci", hold))}


# ------------------------------------------------------------------ obr. 6
def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def gt_boxes(lbl: Path, w: int, h: int) -> list[list[float]]:
    out = []
    for line in lbl.read_text().split("\n") if lbl.exists() else []:
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split())
            if int(c) == MELEE:
                out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def fig_examples(model, test_imgs: list[Path], neg_imgs: list[Path], device: str, data: dict) -> None:
    rows = []           # (obrázek, gt, predikce, TP, FN)
    for p in test_imgs:
        r = model.predict(str(p), conf=CONF_PROD, imgsz=960, device=device, verbose=False)[0]
        h, w = r.orig_shape
        pr = [(b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes if int(b.cls[0]) == MELEE]
        gt = gt_boxes(p.parent.parent / "labels" / f"{p.stem}.txt", w, h)
        tp = [g for g in gt if any(iou(g, q) >= 0.5 for q, _ in pr)]
        fn = [g for g in gt if g not in tp]
        rows.append((p, gt, pr, tp, fn))
    best_tp = max(rows, key=lambda x: (len(x[3]), -len(x[4])))
    best_fn = max(rows, key=lambda x: (len(x[4]), -len(x[3])))

    fp_best = None
    for p in neg_imgs:
        r = model.predict(str(p), conf=CONF_PROD, imgsz=960, device=device, verbose=False)[0]
        for b in r.boxes:
            if int(b.cls[0]) == MELEE and (fp_best is None or float(b.conf[0]) > fp_best[2]):
                fp_best = (p, b.xyxy[0].tolist(), float(b.conf[0]))

    def crop(ax, p: Path, gt, pr, title: str, focus):
        im = Image.open(p).convert("RGB")
        xs = [v for bx in focus for v in (bx[0], bx[2])]; ys = [v for bx in focus for v in (bx[1], bx[3])]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = max(max(xs) - min(xs), max(ys) - min(ys), 450) * 0.75
        x0 = min(max(0, cx - half), im.width - 2 * half); y0 = min(max(0, cy - half), im.height - 2 * half)
        x1, y1 = x0 + 2 * half, y0 + 2 * half
        ax.imshow(im.crop((int(x0), int(y0), int(x1), int(y1))))
        for g in gt:
            ax.add_patch(plt.Rectangle((g[0] - x0, g[1] - y0), g[2] - g[0], g[3] - g[1],
                                       fill=False, ec="#ffffff", lw=1.4, ls="--"))
        for q, c in pr:
            ax.add_patch(plt.Rectangle((q[0] - x0, q[1] - y0), q[2] - q[0], q[3] - q[1],
                                       fill=False, ec="#eda100", lw=1.6))
            ax.text(q[0] - x0, q[1] - y0 - 6, cz(c), color=INK, fontsize=7,
                    bbox=dict(fc="#eda100", ec="none", pad=1))
        ax.set_title(title, loc="left", fontsize=9); ax.axis("off")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    p, gt, pr, tp, fn = best_tp
    crop(axes[0], p, gt, pr, f"a) správně: {len(tp)} z {len(gt)} tyčí", tp or gt)
    p, gt, pr, tp, fn = best_fn
    big = max(fn, key=lambda g: (g[2] - g[0]) * (g[3] - g[1]))
    crop(axes[1], p, gt, pr, f"b) přehlédnuto: {len(fn)} z {len(gt)} tyčí ve framu", [big])
    if fp_best:
        p, q, c = fp_best
        crop(axes[2], p, [], [(q, c)], "c) falešný poplach (bez zbraní)", [q])
    fig.text(0.01, -0.02, "plná žlutá = detekce modelu v2 (práh 0,20) s confidence;  "
             "bílá čárkovaná = ruční anotace", fontsize=7.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig6_ukazky")
    data["ukazky"] = {"spravne": best_tp[0].name, "prehlednuto": best_fn[0].name,
                      "falesny_poplach": fp_best[0].name if fp_best else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    test_imgs = sorted((HERE / "annotate_005/images").glob("*.jpg"))
    neg_imgs = sorted((HERE / "annotate_neg_test/images").glob("*.jpg"))
    data: dict = {}

    fig_training(data)

    test, neg, yolo = {}, {}, {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for n, (w, _, _) in MODELS.items():
            yolo[n] = YOLO(str(HERE.parent / w))
            test[n] = val_set(yolo[n], test_imgs, args.device, td, "test")
            neg[n] = melee_confs(yolo[n], neg_imgs, args.device)
        v2 = yolo["v2 (+ negativa)"]
        cal = val_set(v2, [p for p in test_imgs if frame_idx(p) <= 902], args.device, td, "cal")
        hold = val_set(v2, [p for p in test_imgs if frame_idx(p) >= 912], args.device, td, "hold")

    fig_models(test, data)
    fig_false_alarms(neg, len(neg_imgs), data)
    fig_calibration(cal, hold, data)
    fig_examples(v2, test_imgs, neg_imgs, args.device, data)

    (OUT / "figures_data.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(f"hotovo -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
