"""
Příznaky z detekce osob pro MED (doplněk k med_features.py, stejné vzorkované snímky).

YOLO11x (COCO, třída person) na každém vzorkovaném snímku:
počet osob, průměrná a směrodatná výška boxu (relativně k výšce snímku), rozptyl středů,
podíl plochy pokryté boxy, hustota osob v mřížce 4×4, průměrná jistota detekce = 24 rozměrů.
Cílí hlavně na kongesci (hustý, pomalý dav), kterou CNN a optický tok rozlišují špatně.

Výstup: pole `person` doplněné do med_features/<video>.npz.

Použití:
    python med_person_features.py
"""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

FEAT_DIR = Path('med_features')
VIDEO_DIR = Path('data/Motion_Emotion')
GRID = 4


def person_features(boxes, w, h):
    f = np.zeros(8 + GRID * GRID, dtype=np.float32)
    n = len(boxes)
    f[0] = n
    if n:
        bh = (boxes[:, 3] - boxes[:, 1]) / h
        cx = (boxes[:, 0] + boxes[:, 2]) / 2 / w
        cy = (boxes[:, 1] + boxes[:, 3]) / 2 / h
        f[1:8] = [bh.mean(), bh.std(), cx.std(), cy.std(),
                  ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum() / (w * h),
                  boxes[:, 4].mean(), np.log1p(n)]
        gi = np.clip((cy * GRID).astype(int), 0, GRID - 1)
        gj = np.clip((cx * GRID).astype(int), 0, GRID - 1)
        np.add.at(f, 8 + gi * GRID + gj, 1)
    return f


def main():
    model = YOLO('models/yolo/yolo11x.pt')
    for p in sorted(FEAT_DIR.glob('*.npz')):
        d = dict(np.load(p))
        if 'person' in d:
            continue
        cap = cv2.VideoCapture(str(VIDEO_DIR / f'{p.stem}.mp4'))
        want, feats, i = set(d['frame'].tolist()), {}, 0
        ok, fr = cap.read()
        while ok:
            if i in want:
                r = model.predict(fr, classes=[0], conf=0.25, imgsz=1280, verbose=False)[0]
                b = np.concatenate([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]], 1)
                feats[i] = person_features(b, fr.shape[1], fr.shape[0])
            ok, fr = cap.read()
            i += 1
        d['person'] = np.stack([feats[k] for k in d['frame'].tolist()])
        np.savez(p, **d)
        print(f'{p.stem}: průměrně {d["person"][:, 0].mean():.1f} osob', flush=True)


if __name__ == '__main__':
    main()
