"""
Úloha 3 (interakce / rvačky) na RWF-2000: obrazová větev nad celým videem a fúze se skeletovým modelem dvojic.

Skeletový model (rwf_pair.py, 0,833) je omezený kvalitou skeletů (přesnost po oknech ~0,72, 56/400 videí bez
okna). Obrazová větev vidí přímo pixely: MViT-v2-S (Li et al. 2022, předtrénovaný na Kinetics-400,
torchvision), tedy prostorově-časový transformer nad 16 snímky rovnoměrně z celého klipu (~5 s), 224 × 224.
Doučuje se celá síť s novou hlavou Fight/NonFight.

Protokol stejný jako u skeletových modelů: výběr epochy na 15 % trénovacích videí (stratifikovaně podle seedu),
oficiální val RWF-2000 (400 videí) jen jako test, 3 seedy.
Fúze (pevná, bez ladění na testu): průměr pravděpodobnosti Fight z obrazu a ze skeletů (rwf_pair_dist_probs.json,
stejný seed); u videí bez skeletového okna jen obraz.

Vstup: data/RWF-2000/videos/{train,val}/{Fight,NonFight}/*.avi (download_rwf.py).
Použití:
    python rwf_video.py                     # -> rwf_video_results.json
    python rwf_video.py --seeds 0 --epochs 2   # rychlá kontrola
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

from eval_rwf_lstm import metrics

VID = Path('data/RWF-2000/videos')
N_FRAMES, SIZE = 16, 224
MEAN, STD = np.array([0.45] * 3, np.float32), np.array([0.225] * 3, np.float32)
EPOCHS, BATCH, LR, WD, PATIENCE = 15, 8, 1e-4, 0.05, 5
DEV = 'cuda'


def list_videos(split):
    return [{'path': str(p), 'id': f'{c}/{p.stem}', 'label': lab}
            for c, lab in (('NonFight', 0), ('Fight', 1)) for p in sorted((VID / split / c).glob('*.avi'))]


def read_clip(path, train):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return np.zeros((N_FRAMES, SIZE, SIZE, 3), np.uint8)
    n = len(frames)
    edges = np.linspace(0, n, N_FRAMES + 1)
    if train:       # náhodný snímek v každém z 16 úseků
        idx = [int(np.random.uniform(a, max(a + 1e-3, b - 1e-3))) for a, b in zip(edges[:-1], edges[1:])]
    else:           # střed úseku
        idx = ((edges[:-1] + edges[1:]) / 2).astype(int)
    clip = np.stack([frames[min(i, n - 1)] for i in idx])
    h, w = clip.shape[1:3]
    if train:       # náhodný výřez 60–100 % plochy, stejný pro celý klip, a zrcadlení
        s = np.sqrt(np.random.uniform(0.6, 1.0))
        ch, cw = int(h * s), int(w * s)
        y0, x0 = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)
        clip = clip[:, y0:y0 + ch, x0:x0 + cw]
        if random.random() < 0.5:
            clip = clip[:, :, ::-1]
    return np.stack([cv2.resize(np.ascontiguousarray(f), (SIZE, SIZE)) for f in clip])


class ClipDataset(Dataset):
    def __init__(self, vids, train=False):
        self.vids, self.train = vids, train

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, i):
        v = self.vids[i]
        x = (read_clip(v['path'], self.train)[..., ::-1].astype(np.float32) / 255 - MEAN) / STD   # BGR -> RGB
        return torch.from_numpy(np.ascontiguousarray(x.transpose(3, 0, 1, 2))), v['label']      # (C,T,H,W)


def build_model():
    m = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
    m.head[-1] = nn.Linear(m.head[-1].in_features, 2)
    return m.to(DEV)


@torch.no_grad()
def predict(model, vids):
    model.eval()
    out = []
    for x, _ in DataLoader(ClipDataset(vids), batch_size=16, num_workers=6):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(torch.softmax(model(x.to(DEV)).float(), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def split_select(vids, seed, frac=0.15):
    rng = random.Random(seed)
    tr, sel = [], []
    for c in (0, 1):
        vs = [v for v in vids if v['label'] == c]
        rng.shuffle(vs)
        k = int(round(len(vs) * frac))
        sel += vs[:k]; tr += vs[k:]
    return tr, sel


def train(seed, vids, epochs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tr, sel = split_select(vids, seed)
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=epochs * (len(tr) // BATCH), pct_start=0.1)
    loader = DataLoader(ClipDataset(tr, train=True), batch_size=BATCH, shuffle=True, num_workers=6,
                        drop_last=True, persistent_workers=True)
    ysel = np.array([v['label'] for v in sel])
    best, best_state, best_ep, stale = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(model(x.to(DEV)).float(), y.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        acc = float(((predict(model, sel) >= 0.5) == ysel).mean())
        print(f'  seed {seed} epocha {ep}: loss {np.mean(losses):.3f}, výběrová přesnost {acc:.3f}', flush=True)
        if acc > best:
            best, best_ep, stale = acc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {'seed': seed, 'select_acc': best, 'best_epoch': best_ep, 'epochs_run': ep,
                   'n_train_videos': len(tr), 'n_select_videos': len(sel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--skeleton-probs', default='rwf_pair_dist_probs.json')
    ap.add_argument('--save-model', default='models/behavior/fight_video.pt', help='uloží model prvního seedu')
    ap.add_argument('--out', default='rwf_video_results.json')
    a = ap.parse_args()
    trv, val = list_videos('train'), list_videos('val')
    y = [v['label'] for v in val]
    print(f'train {len(trv)} videí, test (val) {len(val)} videí', flush=True)
    sk = {}
    if Path(a.skeleton_probs).exists():
        with open(a.skeleton_probs, encoding='utf-8') as f:
            for r in json.load(f)['modes']['dist']['runs']:
                sk[r['seed']] = dict(zip(r['video_ids'], r['video_prob_mean']))
    res = {'model': 'MViT-v2-S Kinetics-400', 'frames': N_FRAMES, 'size': SIZE, 'runs': []}
    for s in a.seeds:
        model, info = train(s, trv, a.epochs)
        if s == a.seeds[0] and a.save_model:
            Path(a.save_model).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': model.state_dict(), 'arch': 'mvit_v2_s', 'frames': N_FRAMES, 'size': SIZE,
                        'mean': MEAN.tolist(), 'std': STD.tolist(), 'info': info}, a.save_model)
        p = predict(model, val)
        run = {**info, 'video': metrics(y, (p >= 0.5).astype(int), p),
               'video_ids': [v['id'] for v in val], 'video_prob': [round(float(x), 5) for x in p]}
        line = f'seed {s}: obraz acc {run["video"]["acc"]:.3f} AUC {run["video"]["roc_auc"]:.3f}'
        if s in sk:
            missing = [v['id'] for v in val if v['id'] not in sk[s]]
            ps = np.array([sk[s].get(v['id'], 0.0) for v in val])
            pf = np.where(ps > 0, (p + ps) / 2, p)            # bez skeletového okna (prob 0) jen obraz
            run['fusion'] = metrics(y, (pf >= 0.5).astype(int), pf)
            run['skeleton'] = metrics(y, (ps >= 0.5).astype(int), ps)
            run['ids_missing_in_skeleton'] = len(missing)
            line += (f' | skelety acc {run["skeleton"]["acc"]:.3f} | fúze acc {run["fusion"]["acc"]:.3f} '
                     f'AUC {run["fusion"]["roc_auc"]:.3f} (chybí id {len(missing)})')
        print(line, flush=True)
        res['runs'].append(run)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
    for k in ('video', 'skeleton', 'fusion'):
        vals = [r[k] for r in res['runs'] if k in r]
        if vals:
            res[f'summary_{k}'] = {m: {'mean': float(np.mean([v[m] for v in vals])), 'std': float(np.std([v[m] for v in vals]))}
                                   for m in ('acc', 'macro_f1', 'fight_f1', 'roc_auc')}
            print(f'{k}: acc {res[f"summary_{k}"]["acc"]["mean"]:.3f} ± {res[f"summary_{k}"]["acc"]["std"]:.3f}, '
                  f'AUC {res[f"summary_{k}"]["roc_auc"]["mean"]:.3f}')
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
