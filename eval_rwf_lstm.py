"""
Vyhodnocení BiLSTM+Attention na RWF-2000 (Fight / NonFight).

run_train_lstm.py vybírá nejlepší checkpoint podle val_acc na oficiálním val splitu RWF-2000,
který je zároveň jedinou nezávislou sadou (v literatuře se na něm reportuje test). Číslo 0,756
je proto mírně optimistické a je jen po oknech (15 snímků), ne po videích.

Tento skript:
  1. checkpoint best_lstm_pytorch.pt: metriky po oknech + po videích (agregace oken),
  2. --holdout: znovu natrénuje stejný model se stejnými hyperparametry, ale checkpoint vybírá
     na 15 % trénovacích VIDEÍ (stratifikovaně) a oficiální val použije jen jednou jako test.
     Opakuje pro několik seedů a reportuje průměr ± směrodatnou odchylku.

Video-level: pravděpodobnost Fight = průměr přes všechna okna všech osob ve videu (mean)
nebo maximum přes osoby z průměru jejich oken (max_track). Videa bez okna (žádná osoba
sledovaná ≥ 15 snímků) se počítají jako NonFight a hlásí se zvlášť.

Použití:
    python eval_rwf_lstm.py                       # jen checkpoint
    python eval_rwf_lstm.py --holdout --seeds 0 1 2
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset

import run_train_lstm as T

OUT_JSON = 'rwf_lstm_eval.json'


def load_videos(split):
    """[(video_id, label, [okna (15,14,2), ...]), ...] — okna stejně jako SkeletonDataset."""
    vids = []
    for cls_dir, label in [('NonFight', 0), ('Fight', 1)]:
        for p in sorted(glob.glob(os.path.join(T.RWF_SKELETONS_DIR, split, cls_dir, '*.pkl'))):
            with open(p, 'rb') as f:
                r = T.pickle.load(f)
            wins, track_ids = [], []
            for ti, tr in enumerate(r.get('person_tracks', [])):
                sk = tr['skeletons'][..., :2]
                for i in range(0, len(sk) - T.SEQUENCE_LENGTH + 1, T.STEP_SIZE):
                    wins.append(sk[i:i + T.SEQUENCE_LENGTH])
                    track_ids.append(ti)
            vids.append({'id': f'{cls_dir}/{os.path.basename(p)[:-4]}', 'label': label,
                         'wins': wins, 'tracks': track_ids})
    return vids


class WinDataset(Dataset):
    def __init__(self, vids, augment=False):
        self.samples = [(w, v['label']) for v in vids for w in v['wins']]
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sk, lbl = self.samples[i]
        sk = T.normalize_skeleton(np.nan_to_num(sk.astype(np.float32)))
        if self.augment:
            sk = T.augment_skeleton(sk)
        sk = T.add_velocity_features(sk).reshape(T.SEQUENCE_LENGTH, -1)
        return torch.from_numpy(sk).float(), torch.tensor(lbl, dtype=torch.long)


@torch.no_grad()
def predict(model, vids):
    """Pravděpodobnost Fight pro každé okno, ve stejném pořadí jako WinDataset(vids)."""
    model.eval()
    probs = []
    for x, _ in DataLoader(WinDataset(vids), batch_size=256, shuffle=False):
        probs.append(torch.softmax(model(x.to(T.DEVICE)), 1)[:, 1].cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0)


def metrics(y, p, prob=None):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1], zero_division=0)
    m = {'n': int(len(y)), 'acc': float(np.mean(np.array(y) == np.array(p))),
         'macro_f1': float(f1_score(y, p, average='macro')),
         'fight_precision': float(pr[1]), 'fight_recall': float(rc[1]), 'fight_f1': float(f1[1]),
         'nonfight_recall': float(rc[0]),
         'confusion': confusion_matrix(y, p, labels=[0, 1]).tolist()}
    if prob is not None and len(set(y)) == 2:
        m['roc_auc'] = float(roc_auc_score(y, prob))
    return m


def evaluate(model, vids):
    prob = predict(model, vids)
    y_win = [v['label'] for v in vids for _ in v['wins']]
    res = {'window': metrics(y_win, (prob >= 0.5).astype(int), prob)}
    k = 0
    agg = {'mean': [], 'max_track': []}
    n_empty = 0
    for v in vids:
        n = len(v['wins'])
        pv, k = prob[k:k + n], k + n
        if n == 0:
            n_empty += 1
            agg['mean'].append(0.0); agg['max_track'].append(0.0)
            continue
        agg['mean'].append(float(pv.mean()))
        tr = np.array(v['tracks'])
        agg['max_track'].append(float(max(pv[tr == t].mean() for t in set(v['tracks']))))
    y_vid = [v['label'] for v in vids]
    for name, pv in agg.items():
        pv = np.array(pv)
        res[f'video_{name}'] = metrics(y_vid, (pv >= 0.5).astype(int), pv)
    res['videos_without_window'] = n_empty
    res['videos_without_window_fight'] = sum(1 for v in vids if not v['wins'] and v['label'] == 1)
    cov = [i for i, v in enumerate(vids) if v['wins']]
    pv = np.array(agg['mean'])[cov]
    res['video_mean_covered'] = metrics([y_vid[i] for i in cov], (pv >= 0.5).astype(int), pv)
    return res


def train_holdout(seed, train_vids, val_frac, epochs):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    by_cls = {0: [], 1: []}
    for v in train_vids:
        by_cls[v['label']].append(v)
    tr, sel = [], []
    rng = random.Random(seed)
    for c in by_cls.values():
        c = c[:]; rng.shuffle(c)
        n_sel = int(round(len(c) * val_frac))
        sel += c[:n_sel]; tr += c[n_sel:]

    ds = WinDataset(tr, augment=True)
    counts = np.maximum(np.bincount([s[1] for s in ds.samples], minlength=2), 1)
    w = torch.tensor(len(ds) / (2 * counts), dtype=torch.float32, device=T.DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    model = T.BiLSTMAttention(input_dim=T.N_KEYPOINTS * 4).to(T.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=T.LR, weight_decay=T.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(ds, batch_size=T.BATCH_SIZE, shuffle=True, num_workers=0)
    sel_loader = DataLoader(WinDataset(sel), batch_size=256, shuffle=False)

    best, best_state, stale, best_ep = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        T.train_epoch(model, loader, opt, loss_fn, T.DEVICE)
        _, acc, _, _ = T.eval_model(model, sel_loader, loss_fn, T.DEVICE)
        sched.step()
        if acc > best:
            best, best_state, stale, best_ep = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0, ep
        else:
            stale += 1
            if stale >= T.EARLY_STOPPING_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {'seed': seed, 'select_acc': best, 'best_epoch': best_ep, 'epochs_run': ep,
                   'n_train_videos': len(tr), 'n_select_videos': len(sel)}


def fmt(name, m):
    auc = f" AUC={m['roc_auc']:.3f}" if 'roc_auc' in m else ''
    return (f"  {name:<16} n={m['n']:<5} acc={m['acc']:.3f} macroF1={m['macro_f1']:.3f} "
            f"Fight P/R/F1={m['fight_precision']:.3f}/{m['fight_recall']:.3f}/{m['fight_f1']:.3f}{auc}  "
            f"CM={m['confusion']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=T.MODEL_SAVE_PATH)
    ap.add_argument('--holdout', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--epochs', type=int, default=T.DEFAULT_EPOCHS)
    # kratší okno = víc videí s aspoň jedním oknem (při 15 snímcích jich 84/400 nemá žádné);
    # checkpoint je natrénovaný na 15, při jiné délce se proto vyhodnocuje jen --holdout
    ap.add_argument('--seq-len', type=int, default=T.SEQUENCE_LENGTH)
    ap.add_argument('--step', type=int, default=T.STEP_SIZE)
    ap.add_argument('--out', default=OUT_JSON)
    args = ap.parse_args()
    T.SEQUENCE_LENGTH, T.STEP_SIZE = args.seq_len, args.step
    print(f'okno {T.SEQUENCE_LENGTH} snímků ({T.SEQUENCE_LENGTH / 5:.1f} s), krok {T.STEP_SIZE}')

    val = load_videos('val')
    out = {'seq_len': T.SEQUENCE_LENGTH, 'step': T.STEP_SIZE}

    if args.seq_len == 15:
        model = T.BiLSTMAttention(input_dim=T.N_KEYPOINTS * 4).to(T.DEVICE)
        ckpt = torch.load(args.checkpoint, map_location=T.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        r = evaluate(model, val)
        out['checkpoint'] = {'path': args.checkpoint, 'epoch': ckpt.get('epoch'), **r}
        print(f"Checkpoint {args.checkpoint} (epocha {ckpt.get('epoch')}, vybrán na val -> optimistické):")
        for k in ('window', 'video_mean', 'video_max_track', 'video_mean_covered'):
            print(fmt(k, r[k]))
        print(f"  videí bez okna: {r['videos_without_window']} / {len(val)} "
              f"(z toho Fight {r['videos_without_window_fight']})")

    if args.holdout:
        train = load_videos('train')
        runs = []
        for s in args.seeds:
            model, info = train_holdout(s, train, args.val_frac, args.epochs)
            r = evaluate(model, val)
            runs.append({**info, **r})
            print(f"\nHoldout seed {s}: výběr na {info['n_select_videos']} trénovacích videích, "
                  f"epocha {info['best_epoch']}/{info['epochs_run']}, select_acc={info['select_acc']:.3f}")
            for k in ('window', 'video_mean', 'video_max_track', 'video_mean_covered'):
                print(fmt(k, r[k]))
            print(f"  videí bez okna: {r['videos_without_window']} / {len(val)}")
        summ = {}
        for k in ('window', 'video_mean', 'video_max_track', 'video_mean_covered'):
            for metric in ('acc', 'macro_f1', 'fight_f1', 'roc_auc'):
                vals = [x[k][metric] for x in runs if metric in x[k]]
                if vals:
                    summ[f'{k}.{metric}'] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        out['holdout'] = {'val_frac': args.val_frac, 'runs': runs, 'summary': summ}
        print('\nHoldout souhrn (oficiální RWF val jako čistý test, průměr ± sd přes seedy):')
        for k, v in summ.items():
            print(f"  {k:<28} {v['mean']:.3f} ± {v['std']:.3f}")

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f'\n-> {args.out}')


if __name__ == '__main__':
    main()
