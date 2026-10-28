"""
Obrázky pro kapitolu o klasifikaci chování (MED dav, RWF, anomálie póz) -> docs/figures/.
Vstup: med_results.json (med_train.py), rwf_lstm_eval.json (eval_rwf_lstm.py),
       med_pose_results.json (med_pose_anomaly.py, pokud existuje).
Styl shodný s weapon_training/make_thesis_figures.py.

Použití:
    python med_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

OUT = Path("docs/figures")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, GREEN, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "legend.frameon": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 200,
})
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
CZ = {"Neutral": "neutrální", "Panic": "panika", "Fight": "rvačka", "Congestion": "kongesce", "Obstacle": "překážka"}


def cz(v, nd=2):
    return f"{v:.{nd}f}".replace(".", ",")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")


def fig_med_ablation(res):
    feats = [("flow", "optický tok"), ("cnn", "CNN"), ("both", "CNN + tok"), ("all", "CNN + tok + osoby")]
    by = {(r["features"], r["temporal"]): r["summary"] for r in res["results"]}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(feats)); w = 0.36
    for off, temporal, col, lab in ((-w / 2, False, INK2, "jeden snímek"), (w / 2, True, BLUE, "okno 3,2 s + BiLSTM")):
        vals = [by[(f, temporal)]["mean_class_acc"]["mean"] for f, _ in feats]
        err = [by[(f, temporal)]["mean_class_acc"]["std"] for f, _ in feats]
        ax.bar(x + off, vals, w, color=col, label=lab, yerr=err, capsize=2, error_kw=dict(lw=0.8, color=INK))
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.012, cz(v), ha="center", fontsize=7.5, color=INK)
    ax.axhline(0.439, color=ORANGE, ls="--", lw=1.2)
    ax.text(-0.45, 0.445, "Rabiee et al. 2016 (0,44)", color=ORANGE, fontsize=7.5, ha="left")
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in feats])
    ax.set_ylim(0.3, 0.6); ax.yaxis.set_major_formatter(COMMA)
    ax.set_ylabel("průměrná přesnost po třídách")
    ax.set_title("Chování davu na MED: vliv příznaků a časového modelu (5 foldů podle videí)", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); save(fig, "fig_med_ablace")


def fig_med_confusion(res):
    best = next(r for r in res["results"] if r["features"] == "all" and r["temporal"])
    cm = np.array(best["confusion_seed0"], dtype=float)
    norm = cm / cm.sum(1, keepdims=True)
    names = [CZ[c] for c in res["classes"]]
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.grid(False)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cz(norm[i, j])}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_yticklabels(names)
    ax.set_xlabel("predikce"); ax.set_ylabel("anotace MED")
    ax.set_title("Matice záměn, CNN + tok + osoby + BiLSTM (seed 0)", loc="left", fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); save(fig, "fig_med_matice")


def fig_rwf(res):
    ck, ho = res["checkpoint"], res["holdout"]["summary"]
    levels = [("window", "okno 3 s"), ("video_mean", "video"), ("video_mean_covered", "video, jen pokrytá")]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(levels)); w = 0.36
    opt = [ck[k]["acc"] for k, _ in levels]
    clean = [ho[f"{k}.acc"]["mean"] for k, _ in levels]
    err = [ho[f"{k}.acc"]["std"] for k, _ in levels]
    ax.bar(x - w / 2, opt, w, color=INK2, label="checkpoint vybraný na testu (optimistické)")
    ax.bar(x + w / 2, clean, w, color=BLUE, yerr=err, capsize=2, error_kw=dict(lw=0.8, color=INK),
           label="výběr na části tréninku, test jednou (čistý odhad)")
    for xs, vals in ((x - w / 2, opt), (x + w / 2, clean)):
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.008, cz(v, 3), ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in levels])
    ax.set_ylim(0.6, 0.9); ax.yaxis.set_major_formatter(COMMA)
    ax.set_ylabel("přesnost (RWF-2000 val)")
    ax.set_title("Rvačka / bez rvačky ze skeletů, BiLSTM + attention", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig_rwf_presnost")


def fig_pose(res):
    keys = [("auc_abnormal", "abnormální (vše)"), ("auc_Panic", "panika"), ("auc_Fight", "rvačka"),
            ("auc_Congestion", "kongesce"), ("auc_Obstacle", "překážka")]
    keys = [(k, n) for k, n in keys if k in res["autoencoder"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(keys)); w = 0.36
    base = [res["baseline_speed"][k] for k, _ in keys]
    ae = [res["autoencoder"][k]["mean"] for k, _ in keys]
    err = [res["autoencoder"][k]["std"] for k, _ in keys]
    ax.bar(x - w / 2, base, w, color=INK2, label="baseline: rychlost osob")
    ax.bar(x + w / 2, ae, w, color=GREEN, yerr=err, capsize=2, error_kw=dict(lw=0.8, color=INK),
           label="GRU autoenkodér (učen jen na normálu)")
    for xs, vals in ((x - w / 2, base), (x + w / 2, ae)):
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.008, cz(v), ha="center", fontsize=7.5, color=INK)
    ax.axhline(0.5, color=ORANGE, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in keys])
    ax.set_ylim(0.3, 1.0); ax.yaxis.set_major_formatter(COMMA)
    ax.set_ylabel("AUC (snímky, proti neutrálním)")
    ax.set_title(f"Anomálie póz jednotlivců na MED, okno {cz(res['window'] / 5, 1)} s", loc="left")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig_med_pozy")


def main():
    med = json.load(open("med_results.json", encoding="utf-8"))
    fig_med_ablation(med); fig_med_confusion(med)
    fig_rwf(json.load(open("rwf_lstm_eval.json", encoding="utf-8")))
    if Path("med_pose_results.json").exists():
        fig_pose(json.load(open("med_pose_results.json", encoding="utf-8")))
    else:
        print("  (med_pose_results.json zatím není, obrázek póz přeskočen)")


if __name__ == "__main__":
    main()
