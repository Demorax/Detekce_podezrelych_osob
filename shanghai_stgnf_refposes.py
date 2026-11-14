"""
Úloha 2: kolik z rozdílu proti STG-NF (0,859) dělají skelety a kolik model. Stejný tok jako shanghai_stgnf.py,
ale nad referenčními skelety autorů STG-NF (AlphaPose + sledování, COCO 17 bodů + krk, všech 24 snímků/s;
data/STG-NF/ShanghaiTech, viz data/ZDROJE.md). Varianty pevně předem, 3 seedy, hodnocení STG-NF:
  A_24fps  plná snímková frekvence, okno 24 snímků, trénink krok 6, test krok 1 (výchozí nastavení STG-NF),
  B_5fps   každý 5. snímek, okno 12 vzorků, krok 1 (stejné vzorkování jako naše skelety ViTPose).
  C_12fps  každý 2. snímek, okno 12 vzorků (1 s jako A), trénink krok 3 (6 snímků jako A): stačí
           pro naše skelety hustší extrakce po 2 snímcích (poloviční čas proti všem snímkům)?
Srovnání: B_5fps vs. naše skelety (shanghai_stgnf.py, tok 0,787) = vliv kvality skeletů,
          A_24fps vs. B_5fps = vliv snímkové frekvence, A_24fps vs. článek (0,859) = věrnost reimplementace.

Použití:
    python shanghai_stgnf_refposes.py           # -> shanghai_stgnf_refposes_results.json
    python shanghai_stgnf_refposes.py --variants C_12fps --out shanghai_stgnf_refposes_c.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import shanghai_stgnf as S

REF = Path('data/STG-NF/ShanghaiTech')
V18 = 18
# COCO 17 (0 nos, 1-2 oči, 3-4 uši, 5-6 ramena, 7-8 lokty, 9-10 zápěstí, 11-12 boky, 13-14 kolena,
# 15-16 kotníky) + 17 krk = průměr ramen; hrany jako graf OpenPose 18 ve STG-NF (bez hrany bok-bok)
EDGES18 = [(0, 1), (0, 2), (1, 3), (2, 4), (17, 0), (17, 5), (17, 6), (5, 7), (7, 9), (6, 8), (8, 10),
           (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
VARIANTS = {'A_24fps': {'step': 1, 'T': 24, 'train_stride': 6}, 'B_5fps': {'step': 5, 'T': 12, 'train_stride': 1},
            'C_12fps': {'step': 2, 'T': 12, 'train_stride': 3}}


def load(split, step, T, stride):
    """{seq: {'n_frames', 'idx', 'pose' (N,2,T,18), 'center' (N,) index do idx}}; okna jen ze souvislých snímků."""
    out = {}
    for p in sorted((REF / 'pose' / split).glob('*_alphapose_tracked_person.json')):
        seq = p.name.split('_alphapose')[0]
        with open(p, encoding='utf-8') as f:
            persons = json.load(f)
        if split == 'test':
            n = len(np.load(REF / 'gt' / 'test_frame_mask' / f'{seq}.npy'))
        else:
            n = 1 + max((int(k) for pp in persons.values() for k in pp), default=0)
        idx = np.arange(0, n, step)
        poses, centers = [], []
        for pp in persons.values():
            keys = {int(k): k for k in pp}                  # klíče snímků mají v různých souborech různou délku
            fr = np.array(sorted(k for k in keys if k % step == 0))
            if len(fr) < T:
                continue
            kp = np.array([pp[keys[k]]['keypoints'] for k in fr],
                          np.float32).reshape(len(fr), 17, 3)[..., :2]
            kp = np.concatenate([kp, kp[:, 5:7].mean(1, keepdims=True)], 1)         # + krk
            s = fr // step
            for i in range(0, len(s) - T + 1, stride):
                if s[i + T - 1] - s[i] != T - 1:
                    continue
                poses.append(kp[i:i + T]); centers.append(s[i + T // 2])
        out[seq] = {'n_frames': n, 'idx': idx,
                    'pose': S.normalize_pose(np.array(poses, np.float32).reshape(-1, T, V18, 2)),
                    'center': np.array(centers, np.int64)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS))
    ap.add_argument('--out', default='shanghai_stgnf_refposes_results.json')
    a = ap.parse_args()
    res = {'variants': {}}
    for name in a.variants:
        cfg = VARIANTS[name]
        train = load('train', cfg['step'], cfg['T'], cfg['train_stride'])
        test = load('test', cfg['step'], cfg['T'], 1)
        y = np.concatenate([np.load(REF / 'gt' / 'test_frame_mask' / f'{k}.npy').astype(int) for k in test])
        x = torch.from_numpy(np.concatenate([d['pose'] for d in train.values() if len(d['pose'])]))
        print(f'{name}: trénovacích oken {len(x)}, test {len(test)} sekvencí, {len(y)} snímků', flush=True)
        runs = []
        for seed in a.seeds:
            flow = S.train_flow(x, cfg['T'], seed, EDGES18, V18)
            sc = {k: S.score(flow, torch.from_numpy(d['pose'])) for k, d in test.items()}
            auc = float(roc_auc_score(y, S.frames_stgnf(test, sc)))
            runs.append({'seed': seed, 'auc': auc})
            print(f'  {name} seed {seed}: AUC {auc:.3f}', flush=True)
        aucs = [r['auc'] for r in runs]
        res['variants'][name] = {**cfg, 'n_train_windows': int(len(x)), 'runs': runs,
                                 'auc': {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs))}}
        print(f'{name}: AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}', flush=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
    print('srovnání: naše skelety ViTPose 5 snímků/s 0,787; STG-NF článek 0,859')


if __name__ == '__main__':
    main()
