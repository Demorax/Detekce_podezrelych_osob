"""
Úloha 2 (abnormální pózy jednotlivců) na ShanghaiTech Campus standardním protokolem,
aby šel výsledek přímo srovnat s literaturou skeletové detekce anomálií.

Stejný model jako na MED (med_pose_anomaly.py): GRU autoenkodér nad okny trajektorie jedné osoby
(normalizovaný skelet CrowdPose 14 bodů + rychlost středu boků), trénink jen na 330 trénovacích
(normálních) videích, skóre okna = chyba rekonstrukce, skóre snímku = průměr 3 nejvyšších skóre
osob, vyhlazené mediánem. Skóre vzorkovaných snímků (každý 5.) se interpoluje na všechny snímky.

Metrika: frame-level AUC přes všech 107 testovacích sekvencí najednou (micro), jak ji uvádějí
Morais et al. 2019 (MPED-RNN 0,734), GEPC (0,761), STG-NF (0,859). Baseline bez učení:
rychlost tří nejrychlejších osob.

Vstup: data/ShanghaiTech/skeletons/{training,testing}/*.pkl (shanghai_skeletons.py).
Použití:
    python shanghai_pose_anomaly.py          # -> shanghai_pose_results.json
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score

import med_pose_anomaly as M

SK = Path('data/ShanghaiTech/skeletons')
MASK = Path('data/ShanghaiTech/shanghaitech/testing/test_frame_mask')


def load_split(split, T):
    out = {}
    for p in sorted((SK / split).glob('*.pkl')):
        with open(p, 'rb') as f:
            r = pickle.load(f)
        wins, centers, speeds = [], [], []
        for tr in r['person_tracks']:
            fr = np.array(tr['frame_indices'])
            if len(fr) < T:
                continue
            seq = M.person_sequence(tr['skeletons'])
            for i in range(len(fr) - T + 1):
                if fr[i + T - 1] - fr[i] != T - 1:
                    continue
                w = seq[i:i + T]
                wins.append(w); centers.append(fr[i + T // 2])
                speeds.append(np.linalg.norm(w[1:, -2:], axis=1).mean())
        out[p.stem] = {'n_frames': r['n_frames'], 'idx': np.array(r['sampled_frame_indices']),
                       'win': np.array(wins, dtype=np.float32).reshape(-1, T, 30),
                       'center': np.array(centers, dtype=np.int64), 'speed': np.array(speeds, dtype=np.float32)}
    return out


def to_all_frames(d, per_window):
    """skóre vzorkovaných snímků -> všechny snímky sekvence (lineární interpolace)."""
    n_s = len(d['idx'])
    s = np.zeros(n_s, dtype=np.float32)
    for k in np.unique(d['center']):
        s[k] = np.sort(per_window[d['center'] == k])[::-1][:M.TOPK].mean()
    s = median_filter(s, size=5)
    return np.interp(np.arange(d['n_frames']), d['idx'], s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=12)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = ap.parse_args()
    train, test = load_split('training', a.window), load_split('testing', a.window)
    labels = {k: np.load(MASK / f'{k}.npy').astype(int) for k in test}
    y = np.concatenate([labels[k] for k in test])
    xtr = torch.from_numpy(np.concatenate([d['win'] for d in train.values() if len(d['win'])]))
    print(f'train {len(train)} videí, {len(xtr)} oken; test {len(test)} sekvencí, {len(y)} snímků '
          f'({y.mean():.0%} abnorm.)')

    base = np.concatenate([to_all_frames(d, d['speed']) for d in test.values()])
    res = {'n_train_windows': int(len(xtr)), 'n_test_frames': int(len(y)),
           'auc_speed_baseline': float(roc_auc_score(y, base)), 'runs': []}
    print(f'baseline rychlost: AUC {res["auc_speed_baseline"]:.3f}')

    mu, sd = xtr.mean((0, 1), keepdim=True), xtr.std((0, 1), keepdim=True) + 1e-6
    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        net = M.GRUAutoencoder().to(M.DEV)
        opt = torch.optim.Adam(net.parameters(), lr=M.LR)
        for _ in range(M.EPOCHS):
            net.train()
            for b in torch.randperm(len(xtr)).split(M.BATCH):
                yb, tgt = net(((xtr[b] - mu) / sd).to(M.DEV))
                opt.zero_grad(); ((yb - tgt) ** 2).mean().backward(); opt.step()
        net.eval()
        scores = []
        with torch.no_grad():
            for k, d in test.items():
                w = torch.from_numpy(d['win'])
                err = [((lambda o: ((o[0] - o[1]) ** 2).mean((1, 2)))(net(((w[b] - mu) / sd).to(M.DEV)))).cpu()
                       for b in torch.arange(len(w)).split(2048)] if len(w) else []
                err = torch.cat(err).numpy() if err else np.zeros(0, dtype=np.float32)
                scores.append(to_all_frames(d, err))
        auc = roc_auc_score(y, np.concatenate(scores))
        res['runs'].append({'seed': seed, 'auc': float(auc)})
        print(f'seed {seed}: AUC {auc:.3f}', flush=True)
    aucs = [r['auc'] for r in res['runs']]
    res['auc_autoencoder'] = {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs))}
    print(f'GRU autoenkodér: AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  '
          f'(lit.: MPED-RNN 0,734, GEPC 0,761, STG-NF 0,859)')
    with open('shanghai_pose_results.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
