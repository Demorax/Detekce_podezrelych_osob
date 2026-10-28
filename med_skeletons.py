"""
Skelety osob se sledováním v čase pro všechna MED videa (vstup pro úlohu 2: anomálie póz).

Stejná pipeline jako pro RWF-2000 (run_rwf_batch.py): RT-DETR osoby -> ViTPose 14 bodů CrowdPose
-> hladový centroidový tracker, 5 snímků/s. Výstup data/Motion_Emotion/skeletons/<video>.pkl se
stejnou strukturou jako RWF pickly (sampled_frame_indices, person_tracks[{frame_indices,
skeletons (T,14,3), bboxes}]), label = None (anotace se bere až při tréninku z med_features.parse_labels).

Použití:
    python med_skeletons.py            # všech 31 videí, hotová přeskočí
    python med_skeletons.py --videos 1 5
"""
import argparse
import pickle
import time
from pathlib import Path

import torch

import run_rwf_batch as R
from suspicious_detector import (RTDETRDetector, VitPoseExtractor, _RTDETR_CONFIG, _RTDETR_MODEL,
                                 _VITPOSE_CONFIG, _VITPOSE_MODEL)

VIDEO_DIR = Path('data/Motion_Emotion')
OUT_DIR = VIDEO_DIR / 'skeletons'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', type=int, nargs='*', default=list(range(1, 32)))
    a = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rtdetr = RTDETRDetector(_RTDETR_CONFIG, _RTDETR_MODEL, device=dev)
    vitpose = VitPoseExtractor(_VITPOSE_CONFIG, _VITPOSE_MODEL, device=dev)
    for v in a.videos:
        out = OUT_DIR / f'{v:03d}.pkl'
        if out.exists():
            print(f'{v:03d}: už hotovo', flush=True)
            continue
        t0 = time.time()
        res = R.process_video(rtdetr, vitpose, str(VIDEO_DIR / f'{v:03d}.mp4'), label=None)
        with open(out, 'wb') as f:
            pickle.dump(res, f)
        n_fr = len(res['sampled_frame_indices'])
        print(f'{v:03d}: {n_fr} snímků, {len(res["person_tracks"])} tracků, '
              f'{sum(res["n_persons_per_frame"]) / max(n_fr, 1):.1f} osob/snímek, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
