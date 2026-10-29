"""
ABODA s detektorem statických objektů (static_object_detector.py) místo COCO detekce zavazadel.

Osoby: YOLO11x (třída person, jeden průchod, imgsz 1280), 5 snímků/s, stejné trackery a pravidla
opuštění jako pipeline (abandon_sec 5 jako původní ABODA běh). ABODA nemá veřejný GT po snímcích,
proto se ke každému alertu ukládá snímek s vyznačeným objektem a vlastníkem k vizuálnímu ověření.

Výstup: reference_img/aboda_static/<video>_alerts.json, <video>_alert<k>.jpg, souhrn.json
Použití:
    python run_aboda_static.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from abandoned_object_detector import PersonTracker, ObjectTracker, update_associations, check_abandonment
from static_object_detector import StaticObjectDetector

OUT = Path('reference_img/aboda_static')
SAMPLE_FPS = 5
ABANDON_SEC = 5


def run(video, yolo):
    cap = cv2.VideoCapture(f'data/ABODA/{video}.avi')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / SAMPLE_FPS)))
    sod, pt, ot = None, PersonTracker(max_dist=100, max_missed=5), ObjectTracker(iou_threshold=0.2, max_missed=15)
    alerts, i, si = [], 0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            if sod is None:
                sod = StaticObjectDetector(f.shape, fps=SAMPLE_FPS)
            r = yolo.predict(f, classes=[0], conf=0.3, imgsz=1280, verbose=False)[0]
            pb = np.concatenate([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]], 1)
            dets = sod.update(f, pb)
            ptr, otr = pt.update(si, pb), ot.update(si, dets)
            update_associations(si, ptr, otr, radius_factor=1.5)
            for a in check_abandonment(si, SAMPLE_FPS, otr, abandon_sec=ABANDON_SEC, min_ownership_frames=1):
                a['video_frame'], a['time_sec'] = i, i / fps
                v = f.copy()
                for b in pb:
                    cv2.rectangle(v, tuple(map(int, b[:2])), tuple(map(int, b[2:4])), (0, 255, 0), 1)
                b = a['box']
                cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
                cv2.putText(v, f'{video} t={i / fps:.1f}s', (5, 20), 0, 0.6, (0, 255, 255), 2)
                cv2.imwrite(str(OUT / f'{video}_alert{len(alerts)}.jpg'), v)
                alerts.append(a)
            si += 1
        i += 1
    return alerts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    yolo = YOLO('models/yolo/yolo11x.pt')
    summary = []
    for k in range(1, 12):
        v = f'video{k}'
        alerts = run(v, yolo)
        with open(OUT / f'{v}_alerts.json', 'w', encoding='utf-8') as f:
            json.dump(alerts, f, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
        summary.append({'video': v, 'alerts': len(alerts), 'times': [round(a['time_sec'], 1) for a in alerts]})
        print(f'{v}: {len(alerts)} alertů {summary[-1]["times"]}', flush=True)
    with open(OUT / 'souhrn.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
