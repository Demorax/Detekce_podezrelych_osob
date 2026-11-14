"""
Úloha 1 (chování davu) na MED: obrazový prostorově-časový model místo ručních příznaků (CNN snímku + optický tok).

Na RWF-2000 doučený MViT-v2-S (rwf_video.py) výrazně předčil skeletový model, proto stejný přístup na MED:
MViT-v2-S (Kinetics-400) nad 16 snímky po 5 snímcích/s (3,2 s, tedy stejné okno jako model davu T=16)
se středem v každém vzorkovaném snímku s platným labelem, 5 tříd (Neutral, Panic, Fight, Congestion, Obstacle).

Protokol shodný s med_train.py: 5 foldů podle videí (med_train.FOLDS), počet epoch pevný předem (EPOCHS),
nic se neladí na testu, váhy tříd podle četnosti v tréninku, predikce všech foldů se sloučí a metriky
(med_train.evaluate: průměrná přesnost po třídách, macro-F1, AUC normální/abnormální) se počítají
na všech 31 videích. Trénuje se na každém 2. okně (sousední okna se z 15/16 překrývají), testuje na všech.

Krok 1 (CPU): snímky 5 snímků/s ze všech videí do med_video_frames/<v>.npy (224 × 224 RGB, ~1 GB).
Krok 2 (GPU): trénink a vyhodnocení.
Použití:
    python med_video.py --prepare
    python med_video.py --seeds 0           # -> med_video_results.json
    python med_video.py --final             # model pro demo bez demo videí -> models/behavior/crowd_video.pt
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

import med_train as MT

FEAT = Path('med_features')
FRAMES = Path('med_video_frames')
N_FRAMES, SIZE = 16, 224
MEAN, STD = np.array([0.45] * 3, np.float32), np.array([0.225] * 3, np.float32)
EPOCHS, BATCH, LR, WD, TRAIN_STRIDE = 4, 8, 1e-4, 0.05, 2
DEV = 'cuda'


def prepare():
    FRAMES.mkdir(exist_ok=True)
    for p in sorted(FEAT.glob('*.npz')):
        out = FRAMES / f'{p.stem}.npy'
        if out.exists():
            continue
        idx = set(np.load(p)['frame'].tolist())
        cap, i, fr = cv2.VideoCapture(f'data/Motion_Emotion/{p.stem}.mp4'), 0, {}
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i in idx:
                fr[i] = cv2.cvtColor(cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            i += 1
        np.save(out, np.stack([fr[k] for k in sorted(idx)]))
        print(f'{p.stem}: {len(idx)} snímků', flush=True)


def load():
    data = {}
    for p in sorted(FEAT.glob('*.npz')):
        lab = np.array([MT.MED_TO_CLS.get(int(l), -1) for l in np.load(p)['label']], dtype=np.int64)
        data[int(p.stem)] = (np.load(FRAMES / f'{p.stem}.npy', mmap_mode='r'), lab)
    return data


class ClipDataset(Dataset):
    """Okno 16 vzorkovaných snímků se středem v k (okraje doplněny opakováním, jako med_train.windows)."""
    def __init__(self, data, items, train=False):
        self.data, self.items, self.train = data, items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        v, k = self.items[i]
        fr, lab = self.data[v]
        idx = np.clip(np.arange(k - N_FRAMES // 2, k + N_FRAMES - N_FRAMES // 2), 0, len(fr) - 1)
        clip = fr[idx]
        if self.train:
            s = np.sqrt(np.random.uniform(0.7, 1.0))
            c = int(SIZE * s)
            y0, x0 = np.random.randint(0, SIZE - c + 1, 2)
            clip = np.stack([cv2.resize(f[y0:y0 + c, x0:x0 + c], (SIZE, SIZE)) for f in clip])
            if random.random() < 0.5:
                clip = clip[:, :, ::-1]
        x = (clip.astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(3, 0, 1, 2))), int(lab[k])


def items(data, vids, stride=1):
    return [(v, int(k)) for v in vids for k in np.where(data[v][1] >= 0)[0][::stride]]


def run_fold(data, test_vids, seed, return_model=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tr = items(data, [v for v in data if v not in test_vids], TRAIN_STRIDE)
    te = items(data, test_vids)
    ytr = np.array([data[v][1][k] for v, k in tr])
    w = torch.tensor(len(ytr) / (len(MT.CLASSES) * np.maximum(np.bincount(ytr, minlength=len(MT.CLASSES)), 1)),
                     dtype=torch.float32, device=DEV)
    model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
    model.head[-1] = nn.Linear(model.head[-1].in_features, len(MT.CLASSES))
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loader = DataLoader(ClipDataset(data, tr, train=True), batch_size=BATCH, shuffle=True, num_workers=4,
                        drop_last=True, persistent_workers=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * len(loader), pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    for ep in range(EPOCHS):
        model.train()
        losses = []
        for x, y in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = loss_fn(model(x.to(DEV)).float(), y.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        print(f'    epocha {ep + 1}: loss {np.mean(losses):.3f}', flush=True)
    model.eval()
    if return_model:
        return model
    probs = []
    with torch.no_grad():
        for x, _ in DataLoader(ClipDataset(data, te), batch_size=16, num_workers=4):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                probs.append(torch.softmax(model(x.to(DEV)).float(), 1).cpu().numpy())
    yte = np.array([data[v][1][k] for v, k in te])
    return yte, np.concatenate(probs), te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--out', default='med_video_results.json')
    ap.add_argument('--final', action='store_true')
    a = ap.parse_args()
    if a.prepare:
        prepare()
        return
    data = load()
    if a.final:
        from train_demo_models import DEMO_VIDEOS
        model = run_fold(data, DEMO_VIDEOS, 0, return_model=True)
        torch.save({'state_dict': model.state_dict(), 'arch': 'mvit_v2_s', 'frames': N_FRAMES, 'size': SIZE,
                    'mean': MEAN.tolist(), 'std': STD.tolist(), 'classes': MT.CLASSES, 'sample_fps': 5,
                    'train_videos': sorted(v for v in data if v not in DEMO_VIDEOS),
                    'excluded_demo_videos': DEMO_VIDEOS}, 'models/behavior/crowd_video.pt')
        print('-> models/behavior/crowd_video.pt')
        return
    res = {'model': 'MViT-v2-S Kinetics-400', 'frames': N_FRAMES, 'epochs': EPOCHS, 'train_stride': TRAIN_STRIDE,
           'classes': MT.CLASSES, 'folds': MT.FOLDS, 'runs': []}
    for s in a.seeds:
        ys, ps, ids = [], [], []
        for fi, fold in enumerate(MT.FOLDS):
            print(f'seed {s}, fold {fi + 1}/5 (test videa {fold})', flush=True)
            y, p, te = run_fold(data, fold, s)
            ys.append(y); ps.append(p); ids += te
            r = MT.evaluate(y, p) if len(set(y)) > 1 else {}
            print(f'  fold acc {(p.argmax(1) == y).mean():.3f}, tř.acc {r.get("mean_class_acc", float("nan")):.3f}',
                  flush=True)
        y, p = np.concatenate(ys), np.concatenate(ps)
        r = MT.evaluate(y, p)
        r['seed'] = s
        r['pred'] = {'ids': ids, 'prob': np.round(p, 4).tolist()}
        res['runs'].append(r)
        print(f'seed {s}: acc {r["acc"]:.3f}, prům. přesnost po třídách {r["mean_class_acc"]:.3f}, '
              f'macro-F1 {r["macro_f1"]:.3f}, AUC abnorm. {r["abnormal_auc"]:.3f}, F1 {r["f1"]}', flush=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
    print('srovnání: CNN + tok + osoby, Transformer T=16: prům. přesnost po tř. 0,549, macro-F1 0,525, AUC 0,818')


if __name__ == '__main__':
    main()
