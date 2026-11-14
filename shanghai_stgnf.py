"""
Úloha 2 (abnormální pózy) na ShanghaiTech: normalizační tok nad skelety podle STG-NF (Hirschorn a Avidan 2023)
místo GRU autoenkodéru (shanghai_pose_anomaly.py, AUC 0,743).

Model (převzato z oficiálního kódu github.com/orhir/STG-NF, výchozí argumenty pro ShanghaiTech):
  vstup okno (C=2: x, y; T snímků; V=14 kloubů CrowdPose), normalizace jako normalize_pose: děleno rozlišením
  (856 × 480), přeškálováno do [-1, 1], odečten průměr přes snímky i klouby, děleno sm. odchylkou y,
  takže v okně zůstává pohyb osoby i její póza; K=8 kroků toku (ActNorm, obrácení kanálů, afinní coupling
  se ST-GCN blokem 1 -> 2 kanály, uniformní graf s max_hop=8, časové jádro T//2+1, škála sigmoid(s+2)),
  prior N(R=3, 1), skóre okna = NLL v bitech na rozměr. Adamax 5e-4, wd 5e-5, exp. pokles 0,99, 8 epoch,
  dávka 256, augmentace identita + zrcadlení x.
Rozdíly proti STG-NF, dané našimi daty: skelety ViTPose (14 bodů CrowdPose) místo AlphaPose (18 bodů),
vzorkování každý 5. snímek (~5 snímků/s) místo všech 24 snímků/s, okno T=12 vzorků (2,4 s) jako u autoenkodéru.

Hodnocení pevně stanovené předem (ShanghaiTech nemá validační sadu s anomáliemi, nic se neladí na testu),
hlásí se všechny kombinace:
  model:   AE (shanghai_pose_anomaly.SeqAE, základ) | tok (STG-NF) | AE + tok (součet z-skóre, průměr
           a sm. odchylka skóre z TRÉNOVACÍCH oken)
  skóre:   nase   průměr 3 nejvyšších skóre osob na snímek, medián 5, interpolace (jako dosud)
           stgnf  maximum přes osoby (= minimum věrohodnosti), snímky bez osoby = nejnormálnější skóre
                  v datasetu, interpolace na všechny snímky, gaussovské vyhlazení s sigma 1..6 postupně
                  (smooth_scores ze STG-NF)
Metrika: frame-level AUC přes všech 107 sekvencí najednou (micro). Lit.: MPED-RNN 0,734, GEPC 0,761, STG-NF 0,859.

Použití:
    python shanghai_stgnf.py                    # -> shanghai_stgnf_results.json
    python shanghai_stgnf.py --skeletons data/ShanghaiTech/skeletons_step2 --train-stride 3         --out shanghai_stgnf_step2_results.json     # skelety po 2 snímcích (12 snímků/s), okno 12 = 1 s
"""
import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

import med_pose_anomaly as M
import shanghai_pose_anomaly as SP

SK = SP.SK
IMG_W, IMG_H = SP.IMG_W, SP.IMG_H
V = 14
# CrowdPose: 0/1 ramena, 2/3 lokty, 4/5 zápěstí, 6/7 boky, 8/9 kolena, 10/11 kotníky, 12 hlava, 13 krk
EDGES = [(12, 13), (13, 0), (13, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7),
         (6, 8), (8, 10), (7, 9), (9, 11)]
K_FLOWS, R_PRIOR, MAX_HOP = 8, 3.0, 8
EPOCHS, BATCH, LR, WD, LR_DECAY = 8, 256, 5e-4, 5e-5, 0.99
DEV = M.DEV


# ---------------------------------------------------------------- data
def load_split(split, T, sk_dir=SK, stride=1):
    """Stejná okna jako shanghai_pose_anomaly.load_split (souvislých T vzorků jedné osoby) ve dvou podobách:
    'win' příznaky autoenkodéru (T, 30), 'pose' surové souřadnice pro tok (2, T, V)."""
    out = {}
    for p in sorted((sk_dir / split).glob('*.pkl')):
        with open(p, 'rb') as f:
            r = pickle.load(f)
        wins, poses, centers = [], [], []
        for tr in r['person_tracks']:
            fr = np.array(tr['frame_indices'])
            if len(fr) < T:
                continue
            sk = np.asarray(tr['skeletons'], dtype=np.float32)
            seq = M.person_sequence(sk)
            for i in range(0, len(fr) - T + 1, stride):
                if fr[i + T - 1] - fr[i] != T - 1:
                    continue
                wins.append(seq[i:i + T]); poses.append(sk[i:i + T, :, :2]); centers.append(fr[i + T // 2])
        out[p.stem] = {'n_frames': r['n_frames'], 'idx': np.array(r['sampled_frame_indices']),
                       'win': np.array(wins, dtype=np.float32).reshape(-1, T, 30),
                       'pose': normalize_pose(np.array(poses, dtype=np.float32).reshape(-1, T, V, 2)),
                       'center': np.array(centers, dtype=np.int64)}
    return out


def normalize_pose(p):
    """(N, T, V, 2) px -> (N, 2, T, V) podle normalize_pose ze STG-NF (vid_res 856 × 480, symm_range)."""
    p = 2 * p / np.array([IMG_W, IMG_H], np.float32) - 1
    p = (p - p.mean((1, 2), keepdims=True)) / (p[..., 1].std((1, 2))[:, None, None, None] + 1e-6)
    return np.ascontiguousarray(p.transpose(0, 3, 1, 2))


# ---------------------------------------------------------------- model (STG-NF)
def uniform_adjacency(edges=EDGES, v=V):
    """ST-GCN Graph(strategy='uniform', max_hop=8): všechny dvojice do 8 hran (vč. smyček), A @ D^-1."""
    adj = np.eye(v)
    for a, b in edges:
        adj[a, b] = adj[b, a] = 1
    reach, hop = np.eye(v), np.eye(v)
    for _ in range(MAX_HOP):
        hop = (hop @ adj > 0).astype(float)
        reach = np.maximum(reach, hop)
    return torch.tensor(reach / reach.sum(0, keepdims=True), dtype=torch.float32)[None]   # (1, V, V)


class STGCN(nn.Module):
    """st_gcn ze STG-NF (residual=False): graf. konvoluce -> BN, ReLU, časová konvoluce, BN -> ReLU."""
    def __init__(self, c_in, c_out, t_kernel):
        super().__init__()
        self.gcn = nn.Conv2d(c_in, c_out, 1)          # kernel_size A = 1 (uniform)
        self.tcn = nn.Sequential(nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                                 nn.Conv2d(c_out, c_out, (t_kernel, 1), padding=((t_kernel - 1) // 2, 0)),
                                 nn.BatchNorm2d(c_out))

    def forward(self, x, A):
        x = torch.einsum('nctv,vw->nctw', self.gcn(x), A[0])
        return torch.relu(self.tcn(x))


class ActNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.register_buffer('inited', torch.tensor(False))

    def forward(self, x):
        if not self.inited:                          # inicializace z první dávky (Glow)
            with torch.no_grad():
                self.bias.data = -x.mean((0, 2, 3), keepdim=True)
                self.logs.data = torch.log(1 / (x.std((0, 2, 3), keepdim=True) + 1e-6))
                self.inited.fill_(True)
        return (x + self.bias) * torch.exp(self.logs), self.logs.sum() * x.shape[2] * x.shape[3]


class FlowStep(nn.Module):
    def __init__(self, t_kernel):
        super().__init__()
        self.actnorm = ActNorm(2)
        self.block = STGCN(1, 2, t_kernel)

    def forward(self, x, A):
        x, logdet = self.actnorm(x)
        x = torch.flip(x, [1])                      # Permute2d(shuffle=False): obrácení kanálů
        z1, z2 = x[:, :1], x[:, 1:]
        h = self.block(z1, A)
        shift, scale = h[:, 0::2], torch.sigmoid(h[:, 1::2] + 2.) + 1e-6
        z2 = (z2 + shift) * scale
        return torch.cat([z1, z2], 1), logdet + torch.log(scale).sum((1, 2, 3))


class STGNF(nn.Module):
    def __init__(self, T, edges=EDGES, v=V):
        super().__init__()
        self.register_buffer('A', uniform_adjacency(edges, v))
        self.steps = nn.ModuleList(FlowStep(T // 2 + 1) for _ in range(K_FLOWS))

    def forward(self, x):
        """-> NLL v bitech na rozměr (vyšší = méně věrohodné = anomálnější)."""
        logdet = torch.zeros(len(x), device=x.device)
        for s in self.steps:
            x, ld = s(x, self.A)
            logdet = logdet + ld
        loglik = (-0.5 * (math.log(2 * math.pi) + (x - R_PRIOR) ** 2)).sum((1, 2, 3))
        return -(logdet + loglik) / (math.log(2.0) * x[0].numel())


def train_flow(x, T, seed, edges=EDGES, v=V):
    torch.manual_seed(seed); np.random.seed(seed)
    net = STGNF(T, edges, v).to(DEV)
    opt = torch.optim.Adamax(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, LR_DECAY)
    flip = torch.tensor([-1., 1.]).view(1, 2, 1, 1)
    xa = torch.cat([x, x * flip])                     # num_transform=2: identita + zrcadlení x
    for ep in range(EPOCHS):
        net.train()
        losses = []
        for b in torch.randperm(len(xa)).split(BATCH):
            loss = net(xa[b].to(DEV)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        print(f'    tok epocha {ep + 1}: NLL {np.mean(losses):.3f}', flush=True)
    return net.eval()


def train_ae(x, seed):
    """Základ ze shanghai_pose_anomaly.run_variant (bez rozšíření), vrací funkci skóre okna."""
    mu, sd = x.mean((0, 1), keepdim=True), x.std((0, 1), keepdim=True) + 1e-6
    torch.manual_seed(seed); np.random.seed(seed)
    net = SP.SeqAE(x.shape[-1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=M.LR)
    for _ in range(M.EPOCHS):
        net.train()
        for b in torch.randperm(len(x)).split(M.BATCH):
            loss = net(((x[b] - mu) / sd).to(DEV))[0].mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    mu, sd = mu.to(DEV), sd.to(DEV)
    return lambda w: net((w.to(DEV) - mu) / sd)[0]


@torch.no_grad()
def score(fn, x):
    return torch.cat([fn(x[b].to(DEV)).float().cpu() for b in torch.arange(len(x)).split(4096)]).numpy() \
        if len(x) else np.zeros(0, np.float32)


# ---------------------------------------------------------------- skóre snímků
def frames_nase(test, s):
    lo = min(v.min() for v in s.values() if len(v))
    return np.concatenate([SP.to_all_frames(d, s[k] - lo) for k, d in test.items()])   # bez osoby = 0 = nejnižší


def frames_stgnf(test, s):
    per = {}
    for k, d in test.items():
        a = np.full(len(d['idx']), np.nan, np.float32)
        for c in np.unique(d['center']):
            a[c] = s[k][d['center'] == c].max()
        per[k] = a
    lo = np.nanmin(np.concatenate(list(per.values())))
    out = []
    for k, d in test.items():
        a = np.interp(np.arange(d['n_frames']), d['idx'], np.nan_to_num(per[k], nan=lo))
        for sig in range(1, 7):                      # smooth_scores ze STG-NF
            a = gaussian_filter1d(a, sigma=sig)
        out.append(a)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=12)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--skeletons', default=str(SK), help='např. data/ShanghaiTech/skeletons_step2')
    ap.add_argument('--train-stride', type=int, default=1)
    ap.add_argument('--out', default='shanghai_stgnf_results.json')
    a = ap.parse_args()
    sk = Path(a.skeletons)
    train, test = load_split('training', a.window, sk, a.train_stride), load_split('testing', a.window, sk)
    y = np.concatenate([np.load(SP.MASK / f'{k}.npy').astype(int) for k in test])
    ae_x = torch.from_numpy(np.concatenate([d['win'] for d in train.values() if len(d['win'])]))
    fl_x = torch.from_numpy(np.concatenate([d['pose'] for d in train.values() if len(d['pose'])]))
    print(f'trénovacích oken {len(fl_x)}, test {len(test)} sekvencí, {len(y)} snímků', flush=True)
    res = {'window': a.window, 'skeletons': a.skeletons, 'train_stride': a.train_stride, 'n_train_windows': int(len(fl_x)), 'n_test_frames': int(len(y)), 'runs': []}
    for seed in a.seeds:
        ae = train_ae(ae_x, seed)
        flow = train_flow(fl_x, a.window, seed)
        s = {'AE': {k: score(ae, torch.from_numpy(d['win'])) for k, d in test.items()},
             'tok': {k: score(flow, torch.from_numpy(d['pose'])) for k, d in test.items()}}
        tr_ae, tr_fl = score(ae, ae_x), score(flow, fl_x)     # normalizace fúze jen z trénovacích oken
        s['AE+tok'] = {k: (s['AE'][k] - tr_ae.mean()) / tr_ae.std() + (s['tok'][k] - tr_fl.mean()) / tr_fl.std()
                       for k in test}
        run = {'seed': seed}
        for m, sm in s.items():
            for name, fn in (('nase', frames_nase), ('stgnf', frames_stgnf)):
                run[f'{m}|{name}'] = float(roc_auc_score(y, fn(test, sm)))
        print(f'seed {seed}: ' + ', '.join(f'{k} {v:.3f}' for k, v in run.items() if k != 'seed'), flush=True)
        res['runs'].append(run)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
    keys = [k for k in res['runs'][0] if k != 'seed']
    res['summary'] = {k: {'mean': float(np.mean([r[k] for r in res['runs']])),
                          'std': float(np.std([r[k] for r in res['runs']]))} for k in keys}
    for k, v in res['summary'].items():
        print(f'{k:16s} AUC {v["mean"]:.3f} ± {v["std"]:.3f}')
    print('lit.: MPED-RNN 0,734, GEPC 0,761, STG-NF 0,859')
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
