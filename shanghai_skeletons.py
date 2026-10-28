"""
Skelety se sledováním pro ShanghaiTech Campus (benchmark skeletové anomálie, úloha 2 ze zadání).

Stejná pipeline jako pro MED a RWF (RT-DETR -> ViTPose 14 bodů CrowdPose -> centroidový tracker
z run_rwf_batch.py). ShanghaiTech má 24 FPS; vzorkuje se každý 5. snímek (~5 snímků/s jako MED).
  - testing:  data/ShanghaiTech/shanghaitech/testing/frames/<seq>/*.jpg  (107 sekvencí)
  - training: data/ShanghaiTech/shanghaitech/training/videos/<seq>.avi   (330 videí, jen normál)
Výstup: data/ShanghaiTech/skeletons/<split>/<seq>.pkl se strukturou jako MED/RWF pickly.

Použití:
    python shanghai_skeletons.py --split testing
    python shanghai_skeletons.py --split training
"""
import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import run_rwf_batch as R
from suspicious_detector import (RTDETRDetector, VitPoseExtractor, _RTDETR_CONFIG, _RTDETR_MODEL,
                                 _VITPOSE_CONFIG, _VITPOSE_MODEL)

ROOT = Path('data/ShanghaiTech/shanghaitech')
OUT = Path('data/ShanghaiTech/skeletons')
STEP = 5


def sequences(split):
    if split == 'testing':
        for d in sorted((ROOT / 'testing' / 'frames').iterdir()):
            files = sorted(d.glob('*.jpg'))
            yield d.name, len(files), (lambda i, files=files: cv2.imread(str(files[i])))
    else:
        for v in sorted((ROOT / 'training' / 'videos').glob('*.avi')):
            cap = cv2.VideoCapture(str(v))
            frames = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            if not frames:
                print(f'{v.stem}: nečitelné, přeskočeno', flush=True)
                continue
            yield v.stem, len(frames), (lambda i, frames=frames: frames[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['testing', 'training'], required=True)
    a = ap.parse_args()
    out_dir = OUT / a.split
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rtdetr = RTDETRDetector(_RTDETR_CONFIG, _RTDETR_MODEL, device=dev)
    vitpose = VitPoseExtractor(_VITPOSE_CONFIG, _VITPOSE_MODEL, device=dev)
    for name, n, get in sequences(a.split):
        out = out_dir / f'{name}.pkl'
        if out.exists():
            continue
        t0 = time.time()
        idx = list(range(0, n, STEP))
        sks, bbs = [], []
        for i in idx:
            frame = get(i)
            try:
                boxes = rtdetr.detect(frame, conf_threshold=R.RTDETR_CONF)
                if len(boxes) == 0:
                    sks.append(np.array([])); bbs.append(np.array([])); continue
                sks.append(vitpose.extract_keypoints(frame, boxes)); bbs.append(boxes)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); sks.append(np.array([])); bbs.append(np.array([]))
        res = {'sequence': name, 'n_frames': n, 'sampled_frame_indices': idx,
               'n_persons_per_frame': [len(s) for s in sks], 'person_tracks': R.track_persons(sks, bbs)}
        with open(out, 'wb') as f:
            pickle.dump(res, f)
        print(f'{name}: {len(idx)} vzorků, {len(res["person_tracks"])} tracků, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
