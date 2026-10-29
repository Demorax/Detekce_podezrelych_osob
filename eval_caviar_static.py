"""
Vyhodnocení detektoru statických objektů (static_object_detector.py) na CAVIAR videích.

Osoby se berou z GT (aby se měřila jen detekce předmětu, ne detekce osob), objekty z detektoru
statických objektů nad skutečným videem, pravidla opuštění a trackery stejné jako v pipeline
(eval_caviar_rules.py). Srovnání: stejná pravidla s GT boxy objektů (horní mez, 4/4).
Video se zpracovává 5 snímků/s jako pipeline.

Metriky: TP/FP/FN/TN po scénářích jako eval_caviar_rules (alert v okně GT události = TP,
alert ve scénáři bez opuštění = FP) a latence.

Použití:
    python eval_caviar_static.py
    python eval_caviar_static.py --abandon-sec 2 --radius-factor 1.5
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from abandoned_object_detector import (PersonTracker, ObjectTracker, update_associations,
                                       check_abandonment)
from eval_caviar_rules import parse_caviar_xml, to_person_boxes
from static_object_detector import StaticObjectDetector

SAMPLE_FPS = 5


def run(xml_path, abandon_sec, radius_factor, min_own):
    frames_data, gt_events = parse_caviar_xml(xml_path)
    by_frame = {fd['frame']: fd for fd in frames_data}
    video = next(Path(xml_path).parent.glob('*.mpg'))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = int(round(fps / SAMPLE_FPS))
    sod, pt, ot = None, PersonTracker(max_dist=80 / step * 2, max_missed=10), ObjectTracker(iou_threshold=0.2, max_missed=15)
    alerts, n_static_frames, i, si = [], 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            if sod is None:
                sod = StaticObjectDetector(frame.shape, fps=SAMPLE_FPS)
            fd = by_frame.get(i, {'persons': []})
            persons = to_person_boxes(fd['persons'])
            dets = sod.update(frame, persons if len(persons) else None)
            n_static_frames += bool(dets)
            ptr = pt.update(si, persons if len(persons) else np.zeros((0, 5)))
            otr = ot.update(si, dets)
            update_associations(si, ptr, otr, radius_factor=radius_factor)
            for a in check_abandonment(si, SAMPLE_FPS, otr, abandon_sec=abandon_sec, min_ownership_frames=min_own):
                a['video_frame'] = i
                alerts.append(a)
            si += 1
        i += 1
    res = {'video': Path(xml_path).parent.name, 'alerts': len(alerts),
           'first_alert_frame': alerts[0]['video_frame'] if alerts else None,
           'sampled_frames_with_static': n_static_frames}
    if gt_events:
        gt = gt_events[0]
        res['gt_start'] = gt['start_frame']
        tp = [a for a in alerts if gt['start_frame'] <= a['video_frame'] <= gt['end_frame'] + abandon_sec * fps]
        res['outcome'] = 'TP' if tp else ('FP' if alerts else 'FN')
        if tp:
            res['latency_sec'] = (tp[0]['video_frame'] - gt['start_frame']) / fps
    else:
        res['outcome'] = 'FP' if alerts else 'TN'
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abandon-sec', type=float, default=2.0)
    ap.add_argument('--radius-factor', type=float, default=1.5)
    ap.add_argument('--min-own', type=int, default=1)
    a = ap.parse_args()
    xmls = sorted(p for p in Path('data/CAVIAR').glob('*/[!.]*.xml') if list(p.parent.glob('*.mpg')))
    rows = []
    for x in xmls:
        r = run(str(x), a.abandon_sec, a.radius_factor, a.min_own)
        rows.append(r)
        lat = f"{r['latency_sec']:+.1f} s" if 'latency_sec' in r else ''
        print(f"{r['video']:<22} alertů {r['alerts']:>2}  statických snímků {r['sampled_frames_with_static']:>4}  "
              f"-> {r['outcome']} {lat}", flush=True)
    c = {k: sum(r['outcome'] == k for r in rows) for k in ('TP', 'FP', 'FN', 'TN')}
    p = c['TP'] / max(c['TP'] + c['FP'], 1); rc = c['TP'] / max(c['TP'] + c['FN'], 1)
    lats = [r['latency_sec'] for r in rows if 'latency_sec' in r]
    print(f"\nTP={c['TP']} FP={c['FP']} FN={c['FN']} TN={c['TN']}  precision {p:.0%}  recall {rc:.0%}"
          f"  latence {np.mean(lats) if lats else float('nan'):.1f} s")
    with open('caviar_static_results.json', 'w', encoding='utf-8') as f:
        json.dump({'params': vars(a), 'counts': c, 'precision': p, 'recall': rc, 'rows': rows}, f,
                  indent=1, ensure_ascii=False, default=float)


if __name__ == '__main__':
    main()
