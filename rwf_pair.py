"""
Úloha 3 (interakce mezi jednotlivci): model rvaček nad DVOJICÍ osob místo jedné.

Původní model (run_train_lstm.py / eval_rwf_lstm.py) dostává okno skeletu jedné osoby, takže pozná
„prudký pohyb jednotlivce“, ne interakci. Tady má každé okno osoby A navíc nejbližšího partnera B
ve stejných snímcích (track s překryvem aspoň polovinu okna, nejmenší průměrná vzdálenost středů boků):
  pair  skelet A normalizovaný na vlastní boky a trup (jako dřív) + skelet B ve STEJNÉ souřadné
        soustavě (relativně k bokům a trupu A), rychlosti obou, příznak přítomnosti B
        = 14*2*2 + 14*2*2 + 1 = 113 rozměrů na snímek,
  dist  jen A (56) + poloha a rychlost středu boků B relativně k A + příznak = 61 rozměrů
        (ablace: stačí vědět, kde je soused, nebo je potřeba jeho póza?).
Okna, osoby a pokrytí videí jsou totožné s původním modelem (okno 10, krok 3), mění se jen vstup;
B chybí (žádný soused) -> nuly a příznak 0. Augmentace (zrcadlení, rotace, měřítko, šum, výpadek
kloubů) se aplikuje na A i B společně, aby se zachoval jejich vzájemný vztah.

Protokol je stejný jako eval_rwf_lstm.py --holdout: výběr epochy na 15 % trénovacích videí,
oficiální val RWF-2000 jen jako test, 3 seedy; srovnání s jednou osobou: 0,826 ± 0,005 (okno 10, krok 3).

Použití:
    python rwf_pair.py                          # obě varianty -> rwf_pair_results.json
    python rwf_pair.py --modes pair --seeds 0
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import eval_rwf_lstm as ER
import run_train_lstm as T

HIP, SHO = (6, 7), (0, 1)
LR_PAIRS_AB = T.LEFT_RIGHT_PAIRS + [(l + 14, r + 14) for l, r in T.LEFT_RIGHT_PAIRS]
MAX_REL = 10.0                      # soused dál než 10 délek trupu je pro interakci irelevantní
MODES = {'pair': 113, 'dist': 61}


def _hip(sk):
    return (sk[:, HIP[0]] + sk[:, HIP[1]]) / 2


def load_videos(split):
    """Jako ER.load_videos, ale okno = {'a': (T,14,2), 'b': (T,14,2), 'has_b': (T,)} v pixelech."""
    L, S = T.SEQUENCE_LENGTH, T.STEP_SIZE
    vids = []
    for cls_dir, label in [('NonFight', 0), ('Fight', 1)]:
        for p in sorted(glob.glob(os.path.join(T.RWF_SKELETONS_DIR, split, cls_dir, '*.pkl'))):
            with open(p, 'rb') as f:
                r = T.pickle.load(f)
            tracks = [(np.asarray(tr['frame_indices']), np.nan_to_num(tr['skeletons'][..., :2].astype(np.float32)))
                      for tr in r.get('person_tracks', [])]
            lookup = [{int(fi): k for k, fi in enumerate(fr)} for fr, _ in tracks]
            wins, track_ids = [], []
            for ti, (fr, sk) in enumerate(tracks):
                for i in range(0, len(sk) - L + 1, S):
                    a, fa = sk[i:i + L], fr[i:i + L]
                    ha = _hip(a)
                    best, best_d = None, np.inf
                    for tj, (_, skb) in enumerate(tracks):
                        if tj == ti:
                            continue
                        idx = [lookup[tj].get(int(f), -1) for f in fa]
                        ok = np.array(idx) >= 0
                        if ok.sum() < L / 2:
                            continue
                        d = np.linalg.norm(_hip(skb[np.array(idx)[ok]]) - ha[ok], axis=1).mean()
                        if d < best_d:
                            best, best_d = (tj, np.array(idx), ok), d
                    b, has_b = np.zeros_like(a), np.zeros(L, np.float32)
                    if best is not None:
                        tj, idx, ok = best
                        b[ok] = tracks[tj][1][idx[ok]]
                        has_b[ok] = 1.0
                    wins.append({'a': a, 'b': b, 'has_b': has_b})
                    track_ids.append(ti)
            vids.append({'id': f'{cls_dir}/{os.path.basename(p)[:-4]}', 'label': label,
                         'wins': wins, 'tracks': track_ids})
    return vids


def augment_joint(sk):
    """T.augment_skeleton pro spojené pole A+B (T,28,2) se stejnou transformací pro obě osoby."""
    sk = sk.copy()
    if random.random() < 0.5:
        sk[..., 0] = -sk[..., 0]
        for l, r in LR_PAIRS_AB:
            sk[:, [l, r]] = sk[:, [r, l]]
    angle = np.random.uniform(-0.14, 0.14)
    c, s = np.cos(angle), np.sin(angle)
    sk = sk @ np.array([[c, -s], [s, c]], dtype=sk.dtype).T
    sk *= np.random.uniform(0.9, 1.1)
    sk += np.random.normal(0, 0.02, sk.shape).astype(sk.dtype)
    if T.KEYPOINT_DROPOUT_PROB > 0:
        sk[np.random.random(sk.shape[:2]) < T.KEYPOINT_DROPOUT_PROB] = 0
    return sk


def make_dataset(mode):
    class PairDataset(Dataset):
        def __init__(self, vids, augment=False):
            self.samples = [(w, v['label']) for v in vids for w in v['wins']]
            self.augment = augment

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            w, lbl = self.samples[i]
            a, b, has_b = w['a'], w['b'], w['has_b']
            # souřadná soustava A (stejně jako T.normalize_skeleton), B do ní
            ha, hs = _hip(a), (a[:, SHO[0]] + a[:, SHO[1]]) / 2
            torso = np.maximum(np.linalg.norm(hs - ha, axis=-1), 1e-6)[:, None, None]
            ab = np.concatenate([a, b], 1)
            ab = (ab - ha[:, None]) / torso
            ab[:, 14:][has_b == 0] = 0
            ab[:, 14:] = np.clip(ab[:, 14:], -MAX_REL, MAX_REL)    # A beze změny proti původnímu modelu
            if self.augment:
                ab = augment_joint(ab)
                ab[:, 14:][has_b == 0] = 0
            vel = np.zeros_like(ab)
            vel[1:] = ab[1:] - ab[:-1]
            vel[:, 14:][(has_b == 0) | (np.roll(has_b, 1) == 0)] = 0     # rychlost B jen mezi platnými snímky
            vel[0] = 0
            fa = np.concatenate([ab[:, :14], vel[:, :14]], -1).reshape(len(a), -1)       # 56, jako původní model
            if mode == 'pair':
                fb = np.concatenate([ab[:, 14:], vel[:, 14:]], -1).reshape(len(a), -1)   # 56
            else:
                hb, vb = _hip(ab[:, 14:]), _hip(vel[:, 14:])
                fb = np.concatenate([hb, vb], 1)                                          # 4
            x = np.concatenate([fa, fb, has_b[:, None]], 1).astype(np.float32)
            return torch.from_numpy(x), torch.tensor(lbl, dtype=torch.long)
    return PairDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='+', default=list(MODES), choices=list(MODES))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--seq-len', type=int, default=10)
    ap.add_argument('--step', type=int, default=3)
    ap.add_argument('--out', default='rwf_pair_results.json')
    a = ap.parse_args()
    T.SEQUENCE_LENGTH, T.STEP_SIZE = a.seq_len, a.step
    train, val = load_videos('train'), load_videos('val')
    has = np.concatenate([w['has_b'].mean(keepdims=True) for v in val for w in v['wins']])
    print(f'okno {a.seq_len}, krok {a.step}; val oken {len(has)}, se sousedem {np.mean(has > 0):.0%}', flush=True)
    out = {'seq_len': a.seq_len, 'step': a.step, 'val_windows_with_partner': float(np.mean(has > 0)), 'modes': {}}
    keys = ('window', 'video_mean', 'video_max_track')
    for mode in a.modes:
        ds = make_dataset(mode)
        runs = []
        for s in a.seeds:
            model, info = ER.train_holdout(s, train, 0.15, T.DEFAULT_EPOCHS, ds_cls=ds, input_dim=MODES[mode])
            r = ER.evaluate(model, val, ds)
            runs.append({**info, **r})
            print(f'{mode} seed {s}: epocha {info["best_epoch"]}, ' +
                  ', '.join(f'{k} acc {r[k]["acc"]:.3f} AUC {r[k]["roc_auc"]:.3f}' for k in keys), flush=True)
        summ = {f'{k}.{m}': {'mean': float(np.mean([x[k][m] for x in runs])), 'std': float(np.std([x[k][m] for x in runs]))}
                for k in keys for m in ('acc', 'macro_f1', 'fight_f1', 'roc_auc')}
        out['modes'][mode] = {'runs': runs, 'summary': summ}
        v = summ['video_mean.acc']
        print(f'{mode}: přesnost po videích {v["mean"]:.3f} ± {v["std"]:.3f} (jedna osoba 0,826 ± 0,005)', flush=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
