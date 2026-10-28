"""
Příznaky pro klasifikaci chování davu na MED (oficiální anotace po snímcích).

Pro každé video z data/Motion_Emotion se vzorkuje 5 snímků/s a pro každý vzorkovaný snímek:
  - cnn  (512): ResNet-18 předtrénovaná na ImageNetu, globální průměr poslední vrstvy
                (prostorová informace: hustota davu, rozložení osob, objekty ve scéně),
  - flow (46):  optický tok Farneback mezi snímkem t a t+3 (0,1 s) v polovičním rozlišení:
                statistiky velikosti, podíl pohybujících se pixelů, histogram směrů vážený
                velikostí, entropie směrů, koherence pohybu a průměrná velikost v mřížce 4×4,
  - label:      oficiální anotace MED (dataset_frames_abnormal_labeling.m),
                0 nothing, 1 Panic, 2 Fight, 3 Congestion, 4 Obstacle, 5 Neutral.

Výstup: med_features/<video>.npz (frame, cnn, flow, label).

Použití:
    python med_features.py
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

VIDEO_DIR = Path('data/Motion_Emotion')
LABEL_FILE = VIDEO_DIR / 'Motion Emotion Dataset(MED) annotation' / 'dataset_frames_abnormal_labeling.m'
OUT_DIR = Path('med_features')
SAMPLE_FPS = 5
FLOW_GAP = 3            # snímky mezi dvojicí pro optický tok (0,1 s při 30 FPS)
MOVING_THR = 1.0        # px / FLOW_GAP snímků v polovičním rozlišení
GRID = 4


def parse_labels(path=LABEL_FILE):
    """{1: np.array(label po snímcích), ...} z MATLAB skriptu MED."""
    vids = {}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        m = re.match(r'\s*video(\d+)\s*=\s*zeros\(1,(\d+)\)', line)
        if m:
            vids[int(m.group(1))] = np.zeros(int(m.group(2)), dtype=np.int64)
            continue
        m = re.match(r'\s*video(\d+)\(1,(\d+):(\w+)\)\s*=\s*(\d+)', line)
        if m:
            v, a, b, lab = m.groups()
            arr = vids[int(v)]
            arr[int(a) - 1:len(arr) if b == 'end' else int(b)] = int(lab)
    return vids


def flow_features(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy)
    moving = mag > MOVING_THR
    f = [mag.mean(), mag.std(), *np.percentile(mag, [50, 90, 99]), moving.mean()]
    hist, _ = np.histogram(ang, bins=8, range=(0, 2 * np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-6)
    f += list(hist)
    f.append(-(hist * np.log(hist + 1e-9)).sum())                      # entropie směrů
    # koherence: |průměrný vektor| / průměrná velikost (1 = všichni stejným směrem)
    f.append(np.hypot(fx[moving].mean(), fy[moving].mean()) / (mag[moving].mean() + 1e-6)
             if moving.any() else 0.0)
    h, w = mag.shape
    cells = [mag[i * h // GRID:(i + 1) * h // GRID, j * w // GRID:(j + 1) * w // GRID]
             for i in range(GRID) for j in range(GRID)]
    f += [c.mean() for c in cells]
    f += [(c > MOVING_THR).mean() for c in cells]
    return np.array(f, dtype=np.float32)                               # 6 + 8 + 2 + 16 + 16 = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', type=int, nargs='*', default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    labels = parse_labels()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    cnn = torchvision.models.resnet18(weights=weights)
    cnn.fc = torch.nn.Identity()
    cnn.eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    for v in args.videos or sorted(labels):
        out = OUT_DIR / f'{v:03d}.npz'
        if out.exists():
            print(f'{v:03d}: už hotovo')
            continue
        cap = cv2.VideoCapture(str(VIDEO_DIR / f'{v:03d}.mp4'))
        frames = []
        ok, fr = cap.read()
        while ok:
            frames.append(fr)
            ok, fr = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = int(round(fps / SAMPLE_FPS))
        lab = labels[v]
        idx = [i for i in range(0, len(frames) - FLOW_GAP, step) if i < len(lab)]
        gray = lambda i: cv2.cvtColor(cv2.resize(frames[i], None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
        flow = np.stack([flow_features(gray(i), gray(i + FLOW_GAP)) for i in idx])
        feats = []
        with torch.no_grad():
            for k in range(0, len(idx), 64):
                batch = [cv2.cvtColor(cv2.resize(frames[i], (455, 256)), cv2.COLOR_BGR2RGB) for i in idx[k:k + 64]]
                x = torch.from_numpy(np.stack(batch)).to(dev).permute(0, 3, 1, 2).float() / 255
                feats.append(cnn((x - mean) / std).cpu().numpy())
        np.savez(out, frame=np.array(idx), cnn=np.concatenate(feats).astype(np.float32), flow=flow,
                 label=lab[idx])
        print(f'{v:03d}: {len(idx)} vzorků, třídy {np.bincount(lab[idx], minlength=6).tolist()}', flush=True)


if __name__ == '__main__':
    main()
