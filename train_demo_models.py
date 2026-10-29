"""
Natrénuje a uloží modely chování pro demo (bod 4 plánu praktické části).

Videa, která se v demu ukazují (DEMO_VIDEOS), se z tréninku vyřazují, aby demo neběželo na
trénovacích datech:
  - model davu (med_train.Net, příznaky CNN + tok + osoby, Transformer, okno 16) na MED bez demo videí,
  - autoenkodér póz (med_pose_anomaly.GRUAutoencoder) na Neutral oknech MED bez demo videí,
    práh anomálie = 99. percentil chyby rekonstrukce na trénovacích oknech,
  - model rvaček (run_train_lstm.BiLSTMAttention) na RWF-2000 s oknem 10 a krokem 3
    (nejlepší konfigurace bodu 2), výběr epochy na 15 % trénovacích videí (eval_rwf_lstm.train_holdout).

Výstup: models/behavior/{crowd.pt, pose_ae.pt, fight.pt} (state_dict + normalizace + metadata).
Použití:
    python train_demo_models.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import med_pose_anomaly as MP
import med_train as MT

DEMO_VIDEOS = [5, 10, 15]
OUT = Path('models/behavior')
DEV = MT.DEV


def train_crowd(arch='transformer', T=16, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    data = {v: d for v, d in MT.load('all').items() if v not in DEMO_VIDEOS}
    allx = np.concatenate([d[0] for d in data.values()])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    xs, ys = [], []
    for x, y in data.values():
        w, lab, _ = MT.windows((x - mu) / sd, y, T)
        xs.append(w); ys.append(lab)
    xtr, ytr = torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys))
    cw = torch.tensor(len(ytr) / (len(MT.CLASSES) * np.maximum(np.bincount(ytr.numpy(), minlength=5), 1)),
                      dtype=torch.float32, device=DEV)
    net = MT.Net(xtr.shape[-1], True, arch=arch).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=MT.LR, weight_decay=MT.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MT.EPOCHS)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    for _ in range(MT.EPOCHS):
        net.train()
        for b in torch.randperm(len(ytr)).split(MT.BATCH):
            opt.zero_grad(); loss_fn(net(xtr[b].to(DEV)), ytr[b].to(DEV)).backward(); opt.step()
        sched.step()
    torch.save({'state_dict': net.state_dict(), 'mu': mu, 'sd': sd, 'arch': arch, 'window': T,
                'd_in': int(xtr.shape[-1]), 'classes': MT.CLASSES, 'features': 'flow48+cnn512+person24',
                'train_videos': sorted(data), 'excluded_demo_videos': DEMO_VIDEOS}, OUT / 'crowd.pt')
    print(f'crowd: {len(data)} videí, {len(ytr)} oken -> {OUT / "crowd.pt"}')


def train_pose(T=12, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    data = {v: d for v, d in MP.load_videos(T).items() if v not in DEMO_VIDEOS}
    x = torch.from_numpy(np.concatenate([d['win'][d['label'][d['center']] == 5] for d in data.values()]))
    mu, sd = x.mean((0, 1), keepdim=True), x.std((0, 1), keepdim=True) + 1e-6
    net = MP.GRUAutoencoder().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=MP.LR)
    for _ in range(MP.EPOCHS):
        net.train()
        for b in torch.randperm(len(x)).split(MP.BATCH):
            y, tgt = net(((x[b] - mu) / sd).to(DEV))
            opt.zero_grad(); ((y - tgt) ** 2).mean().backward(); opt.step()
    net.eval()
    with torch.no_grad():
        err = torch.cat([((lambda o: ((o[0] - o[1]) ** 2).mean((1, 2)))(net(((x[b] - mu) / sd).to(DEV)))).cpu()
                         for b in torch.arange(len(x)).split(4096)]).numpy()
    thr = float(np.percentile(err, 99))
    torch.save({'state_dict': net.state_dict(), 'mu': mu, 'sd': sd, 'window': T, 'threshold_p99': thr,
                'train_videos': sorted(data), 'excluded_demo_videos': DEMO_VIDEOS}, OUT / 'pose_ae.pt')
    print(f'pose AE: {len(x)} normálních oken, práh p99 = {thr:.3f} -> {OUT / "pose_ae.pt"}')


def train_fight(seq_len=10, step=3, seed=0):
    import run_train_lstm as RL
    import eval_rwf_lstm as ER
    RL.SEQUENCE_LENGTH, RL.STEP_SIZE = seq_len, step
    train = ER.load_videos('train')
    model, info = ER.train_holdout(seed, train, 0.15, RL.DEFAULT_EPOCHS)
    torch.save({'state_dict': model.state_dict(), 'seq_len': seq_len, 'step': step, 'info': info,
                'input': 'CrowdPose14 xy normalizované na boky/trup + rychlost (56)'}, OUT / 'fight.pt')
    print(f'fight: okno {seq_len}, výběr epochy {info["best_epoch"]}, select_acc {info["select_acc"]:.3f} '
          f'-> {OUT / "fight.pt"}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_crowd(); train_pose(); train_fight()
    with open(OUT / 'README.json', 'w', encoding='utf-8') as f:
        json.dump({'demo_videos_excluded_from_training': DEMO_VIDEOS,
                   'crowd': 'med_train.Net(arch=transformer, T=16), příznaky all',
                   'pose_ae': 'med_pose_anomaly.GRUAutoencoder, okno 12, práh p99 trénovací chyby',
                   'fight': 'run_train_lstm.BiLSTMAttention, RWF-2000, okno 10, krok 3'}, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
