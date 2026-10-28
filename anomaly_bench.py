"""
Přenos modelu chování davu (med_train.py, natrénovaného na CELÉM MED) na cizí datasety
s anotací po snímcích: UCSD Ped1 / Ped2, CUHK Avenue, UMN. Otázka: pozná model natrénovaný na
jedné kameře abnormální snímky v úplně jiné scéně? Metrika: frame-level AUC normální/abnormální,
jak je u těchto datasetů zvykem (skóre = 1 − P(neutrální)).

Baseline bez učení: průměrná velikost optického toku ve snímku (rychlost scény).
In-domain baseline: PCA + kNN vzdálenost k normálním trénovacím snímkům daného datasetu (bez anotací).
Skóre snímků se vyhlazují mediánem přes 5 vzorků.

Příznaky stejné jako na MED (ResNet-18 + optický tok, med_features.py), bez detekce osob
(model 'both'). Vzorkování ~5 snímků/s: UCSD má ~10 FPS (každý 2. snímek), Avenue 25 FPS
(každý 5.), UMN 30 FPS (každý 6.). Okno BiLSTM 16 vzorků jako na MED.

Výstup: anomaly_bench_results.json, cache příznaků v bench_features/.
Použití:
    python anomaly_bench.py                 # vše
    python anomaly_bench.py --datasets ped2 umn
"""
import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import roc_auc_score

import med_train as MT
from med_features import FLOW_GAP, flow_features

CACHE = Path('bench_features')
UCSD = Path('data/UCSD/UCSD_Anomaly_Dataset.v1p2')
AVENUE = Path('data/Avenue')
UMN = Path('data/UMN')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------- načtení cílových datasetů
def parse_ucsd_gt(mfile):
    """UCSDpedX.m -> list polí 1-based indexů abnormálních snímků pro každý testovací klip."""
    out = []
    for line in Path(mfile).read_text().splitlines():
        if 'gt_frame' in line:
            rng = line.split('[')[1].split(']')[0]
            idx = []
            for part in rng.split(','):
                a, b = part.split(':')
                idx += list(range(int(a), int(b) + 1))
            out.append(np.array(idx))
    return out


def ucsd_clips(ped):
    root = UCSD / f'UCSDped{ped}' / 'Test'
    gts = parse_ucsd_gt(root / f'UCSDped{ped}.m')
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.endswith('_gt'))
    assert len(dirs) == len(gts), (len(dirs), len(gts))
    for d, gt in zip(dirs, gts):
        frames = [cv2.imread(str(p)) for p in sorted(d.glob('*.tif'))]
        lab = np.zeros(len(frames), dtype=np.int64)
        lab[gt - 1] = 1
        yield d.name, frames, lab, 2            # ~10 FPS -> každý 2. snímek


def avenue_clips():
    for i in range(1, 22):
        cap = cv2.VideoCapture(str(AVENUE / 'Avenue Dataset' / 'testing_videos' / f'{i:02d}.avi'))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        m = scipy.io.loadmat(str(AVENUE / 'ground_truth_demo' / 'testing_label_mask' / f'{i}_label.mat'))['volLabel']
        lab = np.array([int(np.any(m[0, k])) for k in range(m.shape[1])], dtype=np.int64)
        n = min(len(frames), len(lab))
        yield f'avenue{i:02d}', frames[:n], lab[:n], 5


def umn_clips():
    z = np.load(UMN / 'umn_labels.npz')
    cap = cv2.VideoCapture(str(UMN / 'Crowd-Activity-All.avi'))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    for c in np.unique(z['clip']):
        idx = np.where(z['clip'] == c)[0]
        yield f'umn{c:02d}', [frames[i] for i in idx], z['label'][idx], 6


DATASETS = {'ped1': lambda: ucsd_clips(1), 'ped2': lambda: ucsd_clips(2), 'avenue': avenue_clips, 'umn': umn_clips}


# ---------------------------------------------------------------- trénovací (jen normální) data
def ucsd_train(ped):
    root = UCSD / f'UCSDped{ped}' / 'Train'
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = [cv2.imread(str(p)) for p in sorted(d.glob('*.tif'))]
        yield d.name, frames, 2


def avenue_train():
    for p in sorted((AVENUE / 'Avenue Dataset' / 'training_videos').glob('*.avi')):
        cap = cv2.VideoCapture(str(p))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        yield p.stem, frames, 5


def umn_train():
    # UMN nemá trénovací část: jako normál slouží prvních 60 % normálních snímků každého klipu
    # (zbytek normálu + panika je test). Vyhodnocení pak používá jen testovou část, viz umn_split.
    z = np.load(UMN / 'umn_labels.npz')
    cap = cv2.VideoCapture(str(UMN / 'Crowd-Activity-All.avi'))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    for c in np.unique(z['clip']):
        idx = np.where(z['clip'] == c)[0]
        n_norm = int((z['label'][idx] == 0).sum())
        cut = idx[0] + int(0.6 * n_norm)
        yield f'umn{c:02d}_train', [frames[i] for i in range(idx[0], cut)], 6


TRAIN_SETS = {'ped1': lambda: ucsd_train(1), 'ped2': lambda: ucsd_train(2), 'avenue': avenue_train, 'umn': umn_train}


# ---------------------------------------------------------------- příznaky
class Extractor:
    def __init__(self):
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        self.cnn = torchvision.models.resnet18(weights=w)
        self.cnn.fc = nn.Identity()
        self.cnn.eval().to(DEV)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    def __call__(self, frames, step):
        gap = max(1, round(FLOW_GAP * step / 6))        # 0,1 s při dané snímkové frekvenci
        idx = [i for i in range(0, len(frames) - gap, step)]
        gray = lambda i: cv2.cvtColor(cv2.resize(frames[i], None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
        flow = np.stack([flow_features(gray(i), gray(i + gap)) for i in idx])
        feats = []
        with torch.no_grad():
            for k in range(0, len(idx), 64):
                batch = [cv2.cvtColor(cv2.resize(frames[i], (455, 256)), cv2.COLOR_BGR2RGB) for i in idx[k:k + 64]]
                x = torch.from_numpy(np.stack(batch)).to(DEV).permute(0, 3, 1, 2).float() / 255
                feats.append(self.cnn((x - self.mean) / self.std).cpu().numpy())
        return np.array(idx), np.concatenate(feats).astype(np.float32), flow


def load_dataset(name, ext):
    out = CACHE / f'{name}.npz'
    if out.exists():
        z = np.load(out, allow_pickle=True)
        return list(z['clips'])
    clips = []
    for cname, frames, lab, step in DATASETS[name]():
        idx, cnn, flow = ext(frames, step)
        clips.append({'name': cname, 'x': np.concatenate([flow, cnn], 1), 'flow_mean': flow[:, 0],
                      'label': lab[idx]})
        print(f'  {cname}: {len(idx)} vzorků, abnorm. {lab[idx].mean():.0%}', flush=True)
    CACHE.mkdir(exist_ok=True)
    np.savez(out, clips=np.array(clips, dtype=object))
    return clips


def load_train(name, ext):
    out = CACHE / f'{name}_train.npz'
    if out.exists():
        return np.load(out)['x']
    xs = []
    for cname, frames, step in TRAIN_SETS[name]():
        _, cnn, flow = ext(frames, step)
        xs.append(np.concatenate([flow, cnn], 1))
        print(f'  {cname}: {len(cnn)} trénovacích vzorků', flush=True)
    x = np.concatenate(xs)
    CACHE.mkdir(exist_ok=True)
    np.savez(out, x=x)
    return x


def one_class_scores(xtr, xte, k=5, n_comp=50):
    """In-domain model normality bez anotací: PCA na trénovacích (normálních) příznacích,
    skóre = průměrná vzdálenost ke k nejbližším trénovacím vzorkům (kNN). Standardní jednoduchá
    baseline pro UCSD/Avenue; dává smysl pro srovnání s literaturou, kde se trénuje jen na normálu."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    pca = PCA(n_components=min(n_comp, xtr.shape[1], len(xtr) - 1)).fit((xtr - mu) / sd)
    ztr, zte = pca.transform((xtr - mu) / sd), pca.transform((xte - mu) / sd)
    nn = NearestNeighbors(n_neighbors=k).fit(ztr)
    d, _ = nn.kneighbors(zte)
    return d.mean(1)


# ---------------------------------------------------------------- model natrénovaný na celém MED
def train_on_med(T, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    data = MT.load('both')
    allx = np.concatenate([d[0] for d in data.values()])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    xs, ys = [], []
    for x, y in data.values():
        w, lab, _ = MT.windows((x - mu) / sd, y, T)
        xs.append(w); ys.append(lab)
    xtr, ytr = torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys))
    cw = torch.tensor(len(ytr) / (len(MT.CLASSES) * np.maximum(np.bincount(ytr.numpy(), minlength=5), 1)),
                      dtype=torch.float32, device=DEV)
    net = MT.Net(xtr.shape[-1], True).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=MT.LR, weight_decay=MT.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MT.EPOCHS)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    for _ in range(MT.EPOCHS):
        net.train()
        for b in torch.randperm(len(ytr)).split(MT.BATCH):
            opt.zero_grad(); loss_fn(net(xtr[b].to(DEV)), ytr[b].to(DEV)).backward(); opt.step()
        sched.step()
    net.eval()
    return net, mu, sd


@torch.no_grad()
def score_clip(net, mu, sd, x, T):
    w, _, _ = MT.windows((x - mu) / sd, np.zeros(len(x), dtype=np.int64), T)
    p = torch.cat([torch.softmax(net(torch.from_numpy(w[k:k + 512]).to(DEV)), 1).cpu()
                   for k in range(0, len(w), 512)]).numpy()
    return 1.0 - p[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=list(DATASETS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--window', type=int, default=16)
    a = ap.parse_args()
    ext = Extractor()
    sets = {n: load_dataset(n, ext) for n in a.datasets}
    models = [train_on_med(a.window, s) for s in a.seeds]
    res = {}
    from scipy.ndimage import median_filter
    for n, clips in sets.items():
        xtr = load_train(n, ext)
        keep = []   # UMN: jen testová část klipu (za trénovacími 60 % normálu)
        for c in clips:
            m = np.ones(len(c['label']), dtype=bool)
            if n == 'umn':
                n_norm = int((c['label'] == 0).sum()); m[:int(0.6 * n_norm)] = False
            keep.append(m)
        y = np.concatenate([c['label'][m] for c, m in zip(clips, keep)])
        def smooth(per_clip, per_video_norm):
            # vyhlazení mediánem; volitelně min-max normalizace skóre v každém videu zvlášť
            # (běžná praxe u UCSD/Avenue/UMN, vyrovná rozdílnou úroveň pohybu mezi scénami)
            out = []
            for sc, m in zip(per_clip, keep):
                sc = median_filter(np.asarray(sc, dtype=np.float64), size=5)
                if per_video_norm:
                    sc = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)
                out.append(sc[m])
            return np.concatenate(out)

        flow_pc = [c['flow_mean'] for c in clips]
        knn_pc = [one_class_scores(xtr, c['x']) for c in clips]
        med_pc = [[score_clip(net, mu, sd, c['x'], a.window) for c in clips] for net, mu, sd in models]
        res[n] = {'n_frames': int(len(y)), 'abnormal_frac': float(y.mean()), 'n_train': int(len(xtr))}
        for tag, norm in (('pooled', False), ('per_video', True)):
            aucs = [roc_auc_score(y, smooth(pc, norm)) for pc in med_pc]
            res[n][tag] = {'auc_flow_baseline': float(roc_auc_score(y, smooth(flow_pc, norm))),
                           'auc_in_domain_knn': float(roc_auc_score(y, smooth(knn_pc, norm))),
                           'auc_med_model': {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs))}}
            r = res[n][tag]
            print(f"{n:<7} {tag:<10} snímků {len(y):>5} abnorm. {y.mean():.0%}  AUC tok {r['auc_flow_baseline']:.3f}  "
                  f"in-domain kNN {r['auc_in_domain_knn']:.3f}  model MED {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    with open('anomaly_bench_results.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print('-> anomaly_bench_results.json')


if __name__ == '__main__':
    main()
