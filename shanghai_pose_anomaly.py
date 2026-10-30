"""
Úloha 2 (abnormální pózy jednotlivců) na ShanghaiTech Campus standardním protokolem,
aby šel výsledek přímo srovnat s literaturou skeletové detekce anomálií.

Základ je stejný model jako na MED (med_pose_anomaly.py): GRU autoenkodér nad okny trajektorie jedné osoby
(normalizovaný skelet CrowdPose 14 bodů + rychlost středu boků), trénink jen na 330 trénovacích
(normálních) videích, skóre okna = chyba rekonstrukce, skóre snímku = průměr 3 nejvyšších skóre
osob, vyhlazené mediánem. Skóre vzorkovaných snímků (každý 5.) se interpoluje na všechny snímky.

Rozšíření podle MPED-RNN (Morais et al. 2019), varianty ve VARIANTS:
  box    globální větev: poloha a velikost boxu osoby ve snímku (střed x, y, šířka, výška relativně
         k rozměru snímku); lokální skelet neví, KDE ve scéně se osoba pohybuje a jak je daleko
         (cyklista na chodníku, osoba mimo běžné trasy),
  pred   druhý dekodér předpovídá PRED_LEN dalších snímků; skóre = chyba rekonstrukce + předpovědi
         (okna bez dostatečné budoucnosti mají jen rekonstrukci),
  scena  podmínění identifikátorem kamery (13 scén, první dvě číslice názvu sekvence), aby model
         mohl mít pro každou scénu jinou normalitu polohy (kamera je při nasazení známá).

Metrika: frame-level AUC přes všech 107 testovacích sekvencí najednou (micro), jak ji uvádějí
Morais et al. 2019 (MPED-RNN 0,734), GEPC (0,761), STG-NF (0,859). Baseline bez učení:
rychlost tří nejrychlejších osob. ShanghaiTech nemá validační sadu s anomáliemi; varianty jsou
proto pevně dané předem a hlásí se všechny (žádné ladění hyperparametrů na testu).

Vstup: data/ShanghaiTech/skeletons/{training,testing}/*.pkl (shanghai_skeletons.py).
Použití:
    python shanghai_pose_anomaly.py                  # původní model -> shanghai_pose_results.json
    python shanghai_pose_anomaly.py --variants       # všechny varianty -> shanghai_pose_variants.json
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import median_filter
from sklearn.metrics import roc_auc_score

import med_pose_anomaly as M

SK = Path('data/ShanghaiTech/skeletons')
MASK = Path('data/ShanghaiTech/shanghaitech/testing/test_frame_mask')
IMG_W, IMG_H = 856, 480
N_SCENES = 13
PRED_LEN = 6
VARIANTS = {'zaklad': {}, 'box': {'box': True}, 'box+pred': {'box': True, 'pred': True},
            'box+pred+scena': {'box': True, 'pred': True, 'scene': True}}


def load_split(split, T, box=False, pred=False):
    out = {}
    P = PRED_LEN if pred else 0
    d = 30 + (4 if box else 0)
    for p in sorted((SK / split).glob('*.pkl')):
        with open(p, 'rb') as f:
            r = pickle.load(f)
        wins, futs, fut_ok, centers, speeds = [], [], [], [], []
        for tr in r['person_tracks']:
            fr = np.array(tr['frame_indices'])
            if len(fr) < T:
                continue
            seq = M.person_sequence(tr['skeletons'])
            if box:
                b = np.asarray(tr['bboxes'], dtype=np.float32)
                g = np.stack([(b[:, 0] + b[:, 2]) / 2 / IMG_W, (b[:, 1] + b[:, 3]) / 2 / IMG_H,
                              (b[:, 2] - b[:, 0]) / IMG_W, (b[:, 3] - b[:, 1]) / IMG_H], 1)
                seq = np.concatenate([seq, g], 1)
            for i in range(len(fr) - T + 1):
                if fr[i + T - 1] - fr[i] != T - 1:
                    continue
                w = seq[i:i + T]
                ok = P > 0 and i + T + P <= len(fr) and fr[i + T + P - 1] - fr[i] == T + P - 1
                wins.append(w); centers.append(fr[i + T // 2]); fut_ok.append(ok)
                futs.append(seq[i + T:i + T + P] if ok else np.zeros((P, d), np.float32))
                speeds.append(np.linalg.norm(w[1:, 28:30], axis=1).mean())
        out[p.stem] = {'n_frames': r['n_frames'], 'idx': np.array(r['sampled_frame_indices']),
                       'scene': int(p.stem[:2]) - 1,
                       'win': np.array(wins, dtype=np.float32).reshape(-1, T, d),
                       'fut': np.array(futs, dtype=np.float32).reshape(len(futs), P, d),
                       'fut_ok': np.array(fut_ok, dtype=bool),
                       'center': np.array(centers, dtype=np.int64), 'speed': np.array(speeds, dtype=np.float32)}
    return out


class SeqAE(nn.Module):
    """GRU autoenkodér jako med_pose_anomaly.GRUAutoencoder + volitelně dekodér budoucnosti a podmínění scénou.
    Bez rozšíření je architektura i pořadí inicializace vrstev shodné s původním modelem."""
    def __init__(self, d_in, hidden=64, pred=False, d_cond=0):
        super().__init__()
        self.enc = nn.GRU(d_in + d_cond, hidden, batch_first=True)
        self.dec = nn.GRU(d_in + d_cond, hidden, batch_first=True)
        self.out = nn.Linear(hidden, d_in)
        self.pred = pred
        if pred:
            self.dec_f = nn.GRU(d_in + d_cond, hidden, batch_first=True)
            self.out_f = nn.Linear(hidden, d_in)

    @staticmethod
    def _cat(x, c):
        return x if c is None else torch.cat([x, c[:, None].expand(-1, x.shape[1], -1)], 2)

    def forward(self, x, cond=None, fut=None):
        """-> (chyba rekonstrukce, chyba předpovědi) na okno."""
        _, h = self.enc(self._cat(x, cond))
        tgt = torch.flip(x, [1])
        inp = torch.cat([torch.zeros_like(tgt[:, :1]), tgt[:, :-1]], 1)
        y, _ = self.dec(self._cat(inp, cond), h)
        rec = ((self.out(y) - tgt) ** 2).mean((1, 2))
        if not self.pred:
            return rec, torch.zeros_like(rec)
        finp = torch.cat([x[:, -1:], fut[:, :-1]], 1)          # teacher forcing od posledního snímku okna
        yf, _ = self.dec_f(self._cat(finp, cond), h)
        return rec, ((self.out_f(yf) - fut) ** 2).mean((1, 2))


def to_all_frames(d, per_window):
    """skóre vzorkovaných snímků -> všechny snímky sekvence (lineární interpolace)."""
    n_s = len(d['idx'])
    s = np.zeros(n_s, dtype=np.float32)
    for k in np.unique(d['center']):
        s[k] = np.sort(per_window[d['center'] == k])[::-1][:M.TOPK].mean()
    s = median_filter(s, size=5)
    return np.interp(np.arange(d['n_frames']), d['idx'], s)


def stack(split):
    ds = [d for d in split.values() if len(d['win'])]
    cat = lambda k: torch.from_numpy(np.concatenate([d[k] for d in ds]))
    scene = torch.from_numpy(np.concatenate([np.full(len(d['win']), d['scene']) for d in ds]))
    return cat('win'), cat('fut'), cat('fut_ok'), scene


def run_variant(train, test, y, seeds, box=False, pred=False, scene=False):
    xtr, ftr, oktr, sctr = stack(train)
    mu, sd = xtr.mean((0, 1), keepdim=True), xtr.std((0, 1), keepdim=True) + 1e-6
    eye = torch.eye(N_SCENES)

    def errors(net, x, f, ok, sc):
        c = eye[sc].to(M.DEV) if scene else None
        rec, pe = net(((x - mu) / sd).to(M.DEV), c, ((f - mu) / sd).to(M.DEV))
        return rec, pe * ok.to(M.DEV)

    runs = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        net = SeqAE(xtr.shape[-1], pred=pred, d_cond=N_SCENES if scene else 0).to(M.DEV)
        opt = torch.optim.Adam(net.parameters(), lr=M.LR)
        for _ in range(M.EPOCHS):
            net.train()
            for b in torch.randperm(len(xtr)).split(M.BATCH):
                rec, pe = errors(net, xtr[b], ftr[b], oktr[b], sctr[b])
                loss = rec.mean() + (pe.sum() / oktr[b].sum().clamp(min=1) if pred else 0)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        scores = []
        with torch.no_grad():
            for d in test.values():
                if not len(d['win']):
                    scores.append(to_all_frames(d, np.zeros(0, np.float32)))
                    continue
                x, f, ok = map(torch.from_numpy, (d['win'], d['fut'], d['fut_ok']))
                sc = torch.full((len(x),), d['scene'], dtype=torch.long)
                err = torch.cat([sum(errors(net, x[b], f[b], ok[b], sc[b])).cpu()
                                 for b in torch.arange(len(x)).split(2048)]).numpy()
                scores.append(to_all_frames(d, err))
        auc = float(roc_auc_score(y, np.concatenate(scores)))
        runs.append({'seed': seed, 'auc': auc})
        print(f'  seed {seed}: AUC {auc:.3f}', flush=True)
    aucs = [r['auc'] for r in runs]
    return {'n_train_windows': int(len(xtr)), 'runs': runs,
            'auc': {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=12)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--variants', action='store_true', help='spustit všechny varianty z VARIANTS')
    a = ap.parse_args()
    names = list(VARIANTS) if a.variants else ['zaklad']
    res = {'window': a.window, 'pred_len': PRED_LEN, 'variants': {}}
    y = None
    for name in names:
        cfg = VARIANTS[name]
        train = load_split('training', a.window, cfg.get('box', False), cfg.get('pred', False))
        test = load_split('testing', a.window, cfg.get('box', False), cfg.get('pred', False))
        if y is None:
            labels = {k: np.load(MASK / f'{k}.npy').astype(int) for k in test}
            y = np.concatenate([labels[k] for k in test])
            base = np.concatenate([to_all_frames(d, d['speed']) for d in test.values()])
            res.update(n_test_frames=int(len(y)), auc_speed_baseline=float(roc_auc_score(y, base)))
            print(f'test {len(test)} sekvencí, {len(y)} snímků ({y.mean():.0%} abnorm.); '
                  f'baseline rychlost: AUC {res["auc_speed_baseline"]:.3f}')
        print(f'varianta {name}:', flush=True)
        res['variants'][name] = run_variant(train, test, y, a.seeds, **cfg)
        r = res['variants'][name]['auc']
        print(f'{name}: AUC {r["mean"]:.3f} ± {r["std"]:.3f}', flush=True)
    print('lit.: MPED-RNN 0,734, GEPC 0,761, STG-NF 0,859')
    out = 'shanghai_pose_variants.json' if a.variants else 'shanghai_pose_results.json'
    if not a.variants:      # zachovat formát původního výstupu
        v = res['variants']['zaklad']
        res = {'n_train_windows': v['n_train_windows'], 'n_test_frames': res['n_test_frames'],
               'auc_speed_baseline': res['auc_speed_baseline'], 'runs': v['runs'], 'auc_autoencoder': v['auc']}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
