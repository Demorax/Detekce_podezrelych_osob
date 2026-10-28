"""
Klasifikace chování davu na MED: prostorové příznaky (CNN, optický tok) + časový model (BiLSTM).

Vstup: med_features/*.npz (med_features.py). Třídy podle oficiální anotace MED:
Neutral, Panic, Fight, Congestion, Obstacle (snímky „nothing“ se ignorují).

Vyhodnocení: 5 foldů podle VIDEÍ (žádné video není zároveň v tréninku a testu), každý fold
obsahuje všechny abnormální třídy. Počet epoch je pevný předem, nic se neladí na testu.
Predikce všech foldů se sloučí a metriky se počítají na všech 31 videích; opakuje se
pro několik seedů (průměr ± sd).

Varianty (ablace):
    --features flow | cnn | both
    --temporal  (BiLSTM přes okno --window snímků)  vs. bez něj (MLP na jednom snímku)

Použití:
    python med_train.py --all            # celá ablační tabulka -> med_results.json
    python med_train.py --features both --temporal
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

FEAT_DIR = Path('med_features')
CLASSES = ['Neutral', 'Panic', 'Fight', 'Congestion', 'Obstacle']
MED_TO_CLS = {5: 0, 1: 1, 2: 2, 3: 3, 4: 4}          # MED 0 = nothing -> ignorovat
FOLDS = [[1, 4, 12, 18, 21, 27], [2, 5, 13, 19, 22, 28], [3, 6, 14, 15, 20, 23, 29],
         [7, 10, 16, 24, 25, 30], [8, 9, 11, 17, 26, 31]]
EPOCHS, BATCH, LR, WD = 25, 64, 1e-3, 1e-3
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load(features):
    data = {}
    for p in sorted(FEAT_DIR.glob('*.npz')):
        d = np.load(p)
        x = {'flow': d['flow'], 'cnn': d['cnn'],
             'both': np.concatenate([d['flow'], d['cnn']], 1),
             # + osoby z YOLO11x (med_person_features.py)
             'all': np.concatenate([d['flow'], d['cnn'], d['person']], 1)}[features]
        y = np.array([MED_TO_CLS.get(int(l), -1) for l in d['label']], dtype=np.int64)
        data[int(p.stem)] = (x.astype(np.float32), y)
    return data


def windows(x, y, T):
    """Okna délky T se středem v každém snímku s platným labelem (okraje doplněny opakováním)."""
    pad = np.concatenate([np.repeat(x[:1], T // 2, 0), x, np.repeat(x[-1:], T - 1 - T // 2, 0)])
    keep = np.where(y >= 0)[0]
    return np.stack([pad[i:i + T] for i in keep]), y[keep], keep


class Net(nn.Module):
    def __init__(self, d_in, temporal, hidden=64, drop=0.3):
        super().__init__()
        self.temporal = temporal
        self.proj = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(drop))
        if temporal:
            self.rnn = nn.LSTM(128, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * hidden, len(CLASSES))
        else:
            self.head = nn.Sequential(nn.Linear(128, 2 * hidden), nn.ReLU(), nn.Dropout(drop),
                                      nn.Linear(2 * hidden, len(CLASSES)))

    def forward(self, x):                 # x: (B, T, D); bez časového modelu T = 1
        h = self.proj(x)
        if self.temporal:
            h, _ = self.rnn(h)
            return self.head(h.mean(1))
        return self.head(h[:, 0])


def run_fold(data, test_vids, temporal, T, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train_vids = [v for v in data if v not in test_vids]
    allx = np.concatenate([data[v][0] for v in train_vids])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    T = T if temporal else 1

    def build(vids):
        xs, ys, ids = [], [], []
        for v in vids:
            x, y = data[v]
            w, lab, keep = windows((x - mu) / sd, y, T)
            xs.append(w); ys.append(lab); ids += [(v, int(k)) for k in keep]
        return torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys)), ids

    xtr, ytr, _ = build(train_vids)
    xte, yte, ids = build(test_vids)
    w = torch.tensor(len(ytr) / (len(CLASSES) * np.maximum(np.bincount(ytr.numpy(), minlength=len(CLASSES)), 1)),
                     dtype=torch.float32, device=DEV)
    net = Net(xtr.shape[-1], temporal).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(ytr))
        for k in range(0, len(perm), BATCH):
            b = perm[k:k + BATCH]
            opt.zero_grad()
            loss_fn(net(xtr[b].to(DEV)), ytr[b].to(DEV)).backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        prob = torch.cat([torch.softmax(net(xte[k:k + 512].to(DEV)), 1).cpu()
                          for k in range(0, len(xte), 512)]).numpy()
    return yte.numpy(), prob, ids


def evaluate(y, prob):
    pred = prob.argmax(1)
    f1 = f1_score(y, pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    abn = (y != 0).astype(int)
    # mean_class_acc = průměr recallu přes třídy, metrika článku MED (Rabiee et al. 2016)
    return {'acc': float((pred == y).mean()), 'mean_class_acc': float(balanced_accuracy_score(y, pred)),
            'macro_f1': float(f1.mean()),
            'f1': {c: float(v) for c, v in zip(CLASSES, f1)},
            'abnormal_auc': float(roc_auc_score(abn, 1 - prob[:, 0])),
            'abnormal_f1': float(f1_score(abn, (pred != 0).astype(int))),
            'confusion': confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist()}


def experiment(features, temporal, T, seeds):
    data = load(features)
    runs = []
    for s in seeds:
        ys, ps = [], []
        for fold in FOLDS:
            y, p, _ = run_fold(data, fold, temporal, T, s)
            ys.append(y); ps.append(p)
        runs.append(evaluate(np.concatenate(ys), np.concatenate(ps)))
    summ = {k: {'mean': float(np.mean([r[k] for r in runs])), 'std': float(np.std([r[k] for r in runs]))}
            for k in ('acc', 'mean_class_acc', 'macro_f1', 'abnormal_auc', 'abnormal_f1')}
    summ['f1'] = {c: float(np.mean([r['f1'][c] for r in runs])) for c in CLASSES}
    name = f"{features}{' + BiLSTM' if temporal else ' (1 snímek)'}"
    print(f"{name:<22} acc {summ['acc']['mean']:.3f}  tř.acc {summ['mean_class_acc']['mean']:.3f}  macro-F1 {summ['macro_f1']['mean']:.3f} ± "
          f"{summ['macro_f1']['std']:.3f}  abnorm. AUC {summ['abnormal_auc']['mean']:.3f}  F1/třída "
          + ' '.join(f"{c[:4]} {v:.2f}" for c, v in summ['f1'].items()), flush=True)
    return {'name': name, 'features': features, 'temporal': temporal, 'window': T, 'summary': summ,
            'confusion_seed0': runs[0]['confusion']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='both', choices=['flow', 'cnn', 'both', 'all'])
    ap.add_argument('--temporal', action='store_true')
    ap.add_argument('--window', type=int, default=16)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--all', action='store_true', help='celá ablační tabulka')
    a = ap.parse_args()
    if a.all:
        res = [experiment(f, t, a.window, a.seeds) for f in ('flow', 'cnn', 'both', 'all') for t in (False, True)]
        with open('med_results.json', 'w', encoding='utf-8') as fh:
            json.dump({'classes': CLASSES, 'folds': FOLDS, 'epochs': EPOCHS, 'results': res}, fh,
                      indent=1, ensure_ascii=False)
        print('-> med_results.json')
    else:
        experiment(a.features, a.temporal, a.window, a.seeds)


if __name__ == '__main__':
    main()
