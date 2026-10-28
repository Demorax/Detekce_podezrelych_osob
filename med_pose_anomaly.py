"""
Úloha 2: abnormální pózy jednotlivců na MED, učeno jen z normálních úseků (bez labelů anomálií).

Přístup podle Morais et al. 2019 (Learning Regularity in Skeleton Trajectories): autoenkodér
GRU nad okny skeletových trajektorií jedné osoby. Trénuje se jen na oknech z Neutral snímků
trénovacích videí; skóre anomálie = chyba rekonstrukce okna. Skóre snímku = průměr k nejvyšších
skóre osob v něm (k = 3), vyhlazené mediánem. Osoba, jejíž póza se vymyká naučené normalitě
(běh, pád, zvednuté paže, bitka), dostane vysokou chybu.

Dvě reprezentace jedné osoby (Morais: local + global):
  - lokální: skelet normalizovaný na střed boků a výšku trupu (tvar pózy a její změna),
  - globální: posun středu boků mezi snímky relativně k výšce osoby (rychlost pohybu),
  vstup 14*2 + 2 = 30 rozměrů na snímek, okno T snímků (5 snímků/s).

Vyhodnocení: 5 foldů podle videí (stejné jako med_train.py), AUC na úrovni snímků normální vs.
abnormální na testových videích a AUC zvlášť pro každou abnormální třídu proti Neutral.
Baseline bez učení: skóre = průměrná rychlost k nejrychlejších osob (ukazuje, co přidá naučená
regularita proti prosté rychlosti).

Vstup: data/Motion_Emotion/skeletons/<video>.pkl (med_skeletons.py).
Použití:
    python med_pose_anomaly.py            # -> med_pose_results.json
"""
import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score

from med_features import parse_labels
from med_train import FOLDS

SKEL_DIR = Path('data/Motion_Emotion/skeletons')
CLASSES = {1: 'Panic', 2: 'Fight', 3: 'Congestion', 4: 'Obstacle'}
KP_CONF_MIN = 0.3
warnings.filterwarnings('ignore', 'Mean of empty slice')   # nanmean nad kloubem s nízkou jistotou
TOPK = 3
EPOCHS, BATCH, LR = 30, 128, 1e-3
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HIP, SHO = (6, 7), (0, 1)          # CrowdPose: boky, ramena


def person_sequence(sk):
    """sk (T,14,3) -> (T,30): lokální normalizovaný skelet + globální rychlost. NaN -> 0."""
    xy = sk[..., :2].astype(np.float32).copy()
    xy[sk[..., 2] < KP_CONF_MIN] = np.nan
    hip = np.nanmean(xy[:, HIP], 1)
    sho = np.nanmean(xy[:, SHO], 1)
    torso = np.linalg.norm(sho - hip, axis=-1)
    torso = np.where(np.isfinite(torso) & (torso > 1), torso, np.nan)
    local = (xy - hip[:, None]) / torso[:, None, None]
    vel = np.zeros_like(hip)
    vel[1:] = (hip[1:] - hip[:-1]) / torso[1:, None]
    feat = np.concatenate([local.reshape(len(sk), -1), vel], 1)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def load_videos(T):
    """{v: {'n': počet vzorkovaných snímků, 'label': (n,), 'win': (N,T,30), 'center': (N,), 'speed': (N,)}}"""
    labels = parse_labels()
    data = {}
    for p in sorted(SKEL_DIR.glob('*.pkl')):
        v = int(p.stem)
        with open(p, 'rb') as f:
            r = pickle.load(f)
        idx = np.array(r['sampled_frame_indices'])
        lab = labels[v][np.minimum(idx, len(labels[v]) - 1)]
        wins, centers, speeds = [], [], []
        for tr in r['person_tracks']:
            fr = np.array(tr['frame_indices'])
            if len(fr) < T:
                continue
            seq = person_sequence(tr['skeletons'])
            # jen souvislé úseky (tracker může mít mezery)
            for i in range(0, len(fr) - T + 1):
                if fr[i + T - 1] - fr[i] != T - 1:
                    continue
                w = seq[i:i + T]
                wins.append(w); centers.append(fr[i + T // 2])
                speeds.append(np.linalg.norm(w[1:, -2:], axis=1).mean())
        data[v] = {'n': len(idx), 'label': lab, 'win': np.array(wins, dtype=np.float32),
                   'center': np.array(centers), 'speed': np.array(speeds, dtype=np.float32)}
    return data


class GRUAutoencoder(nn.Module):
    def __init__(self, d_in=30, hidden=64):
        super().__init__()
        self.enc = nn.GRU(d_in, hidden, batch_first=True)
        self.dec = nn.GRU(d_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden, d_in)

    def forward(self, x):
        _, h = self.enc(x)
        # dekodér rekonstruuje okno pozpátku z posledního stavu enkodéru (vstup: posunutý cíl)
        tgt = torch.flip(x, [1])
        inp = torch.cat([torch.zeros_like(tgt[:, :1]), tgt[:, :-1]], 1)
        y, _ = self.dec(inp, h)
        return self.out(y), tgt


def frame_scores(n, centers, per_window, k=TOPK):
    """Skóre snímku = průměr k nejvyšších skóre oken se středem v něm; snímky bez osob = 0."""
    s = np.zeros(n, dtype=np.float32)
    for f in np.unique(centers):
        vals = np.sort(per_window[centers == f])[::-1][:k]
        s[f] = vals.mean()
    return median_filter(s, size=5)


def train_fold(data, test_vids, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train = [v for v in data if v not in test_vids]
    xs = [data[v]['win'][data[v]['label'][data[v]['center']] == 5] for v in train]   # jen Neutral
    x = torch.from_numpy(np.concatenate(xs))
    mu, sd = x.mean((0, 1), keepdim=True), x.std((0, 1), keepdim=True) + 1e-6
    net = GRUAutoencoder().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(EPOCHS):
        net.train()
        for b in torch.randperm(len(x)).split(BATCH):
            xb = ((x[b] - mu) / sd).to(DEV)
            y, tgt = net(xb)
            opt.zero_grad()
            ((y - tgt) ** 2).mean().backward()
            opt.step()
    net.eval()
    out = {}
    with torch.no_grad():
        for v in test_vids:
            w = torch.from_numpy(data[v]['win'])
            err = []
            for b in torch.arange(len(w)).split(1024):
                y, tgt = net(((w[b] - mu) / sd).to(DEV))
                err.append(((y - tgt) ** 2).mean((1, 2)).cpu())
            out[v] = torch.cat(err).numpy() if err else np.zeros(0, dtype=np.float32)
    return out


def evaluate(data, per_video_scores):
    y, s = [], []
    for v, sc in per_video_scores.items():
        y.append(data[v]['label']); s.append(sc)
    y, s = np.concatenate(y), np.concatenate(s)
    keep = y > 0                                  # 'nothing' ignorovat
    y, s = y[keep], s[keep]
    res = {'auc_abnormal': float(roc_auc_score(y != 5, s))}
    for c, name in CLASSES.items():
        m = (y == 5) | (y == c)
        res[f'auc_{name}'] = float(roc_auc_score(y[m] == c, s[m])) if (y[m] == c).any() else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=12)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = ap.parse_args()
    data = load_videos(a.window)
    n_win = sum(len(d['win']) for d in data.values())
    print(f'{len(data)} videí, {n_win} oken délky {a.window} ({a.window / 5:.1f} s)')

    # baseline: rychlost
    base = {v: frame_scores(d['n'], d['center'], d['speed']) for v, d in data.items()}
    rb = evaluate(data, base)
    print('baseline rychlost   ' + '  '.join(f'{k[4:]} {v:.3f}' for k, v in rb.items() if v is not None))

    runs = []
    for s in a.seeds:
        scores = {}
        for fold in FOLDS:
            fold = [v for v in fold if v in data]
            err = train_fold(data, fold, s)
            for v in fold:
                scores[v] = frame_scores(data[v]['n'], data[v]['center'], err[v])
        runs.append(evaluate(data, scores))
    summ = {k: {'mean': float(np.mean([r[k] for r in runs])), 'std': float(np.std([r[k] for r in runs]))}
            for k in runs[0] if runs[0][k] is not None}
    print('GRU autoenkodér     ' + '  '.join(f'{k[4:]} {v["mean"]:.3f}±{v["std"]:.3f}' for k, v in summ.items()))
    with open('med_pose_results.json', 'w', encoding='utf-8') as f:
        json.dump({'window': a.window, 'n_windows': n_win, 'baseline_speed': rb, 'autoencoder': summ,
                   'runs': runs}, f, indent=1, ensure_ascii=False)
    print('-> med_pose_results.json')


if __name__ == '__main__':
    main()
