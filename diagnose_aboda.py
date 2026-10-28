"""
ABODA: plná pipeline (stejné volby jako run_aboda_batch.py) + diagnostika, PROČ alert (ne)přišel.

Původní batch (2026-05-09) dal alert jen ve video1. Běžel ještě s chybou trackeru
(CODE_REVIEW_PIPELINE.md bod 1: track se po jediném vynechaném framu už nespároval, taška
tak ztratila vlastníka). Tady se pro každý track zavazadla zaznamená:
  - kolik sampled framů byl detekován,
  - v kolika měl vlastníka (osoba v radiu),
  - nejdelší dobu bez vlastníka (max_alone_sec) — alert vyžaduje >= abandon_sec.
Tím se selhání rozdělí na: nic nedetekováno / bez vlastníka / vlastník nikdy neodešel na dost dlouho.

Použití:
    python diagnose_aboda.py                       # všech 11 videí
    python diagnose_aboda.py --videos video4 video9
Výstup: reference_img/aboda_alerts_v2/<video>_alerts.json + <video>_diag.json + souhrn.json
"""
import argparse
import json
import os
import types
from collections import Counter
from pathlib import Path

import abandoned_object_detector as aod
from suspicious_detector import SuspiciousDetector

OUT_DIR = Path('reference_img/aboda_alerts_v2')
CARRY = {'backpack', 'handbag', 'suitcase'}


def run_video(video, detector, abandon_sec):
    stats = {}
    frames_with_carry = [0]
    orig_check = aod.check_abandonment

    def check(frame_idx, fps, object_tracks, abandon_sec, min_ownership_frames):
        if any(o['label'] in CARRY for o in object_tracks.values()):
            frames_with_carry[0] += 1
        for oid, o in object_tracks.items():
            s = stats.setdefault(oid, {'label': o['label'], 'first': frame_idx, 'n_frames': 0,
                                       'owner_frames': 0, 'max_alone_sec': 0.0})
            s['last'] = frame_idx
            s['n_frames'] += 1
            if o['owner_history'] and o['owner_history'][-1][1] is not None:
                s['owner_frames'] += 1
            elif o['last_owner_frame'] is not None:
                s['max_alone_sec'] = max(s['max_alone_sec'], (frame_idx - o['last_owner_frame']) / fps)
        return orig_check(frame_idx, fps, object_tracks, abandon_sec, min_ownership_frames)

    aod.check_abandonment = check
    try:
        args = types.SimpleNamespace(
            sample_fps=aod.DEFAULT_SAMPLE_FPS, abandon_sec=abandon_sec,
            radius_factor=aod.DEFAULT_OWNERSHIP_RADIUS_FACTOR,
            max_person_dist=aod.DEFAULT_MAX_PERSON_TRACK_DIST, max_obj_missed=aod.DEFAULT_MAX_OBJ_MISSED,
            obj_iou=aod.DEFAULT_OBJ_IOU_THRESHOLD, min_ownership_frames=aod.DEFAULT_MIN_OWNERSHIP_FRAMES,
            conf_obj=0.10, conf_weapon=None, imgsz=1280, use_better_obj=True, upscale=True,
            save_video=False, output_dir=str(OUT_DIR), armed=False)
        alerts = aod.process_video(f'data/ABODA/{video}.avi', detector, args) or []
    finally:
        aod.check_abandonment = orig_check

    carry = {k: v for k, v in stats.items() if v['label'] in CARRY}
    long_tracks = [v for v in carry.values() if v['n_frames'] >= 10]   # >= 2 s při 5 FPS
    if not carry:
        reason = 'žádné zavazadlo nedetekováno'
    elif alerts:
        reason = 'alert'
    elif not any(v['owner_frames'] for v in long_tracks):
        reason = 'zavazadlo bez vlastníka (nikdy osoba v radiu)'
    elif max((v['max_alone_sec'] for v in long_tracks), default=0) < abandon_sec:
        reason = f'vlastník neodešel na >= {abandon_sec} s (nebo track zanikl dřív)'
    else:
        reason = 'jiné (viz tracky)'
    return alerts, {
        'video': video, 'alerts': len(alerts), 'reason': reason,
        'sampled_frames_with_carry': frames_with_carry[0],
        'carry_tracks': len(carry), 'carry_tracks_ge_2s': len(long_tracks),
        'labels': dict(Counter(v['label'] for v in carry.values())),
        'max_alone_sec': max((v['max_alone_sec'] for v in carry.values()), default=0.0),
        'tracks': sorted(carry.values(), key=lambda v: -v['n_frames'])[:15],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', nargs='*', default=[f'video{i}' for i in range(1, 12)])
    ap.add_argument('--abandon-sec', type=float, default=5)   # jako run_aboda_batch.py
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = SuspiciousDetector(skip_weapon_model=True)
    summary = []
    for v in a.videos:
        done = OUT_DIR / f'{v}_diag.json'
        if done.exists():   # hotové video z dřívějšího (přerušeného) běhu jen převzít do souhrnu
            with open(done, encoding='utf-8') as f:
                diag = json.load(f)
            summary.append({k: diag[k] for k in diag if k != 'tracks'})
            print(f'>>> {v}: převzato z {done.name}', flush=True)
            continue
        alerts, diag = run_video(v, detector, a.abandon_sec)
        for name, obj in ((f'{v}_alerts.json', alerts), (f'{v}_diag.json', diag)):
            with open(OUT_DIR / name, 'w', encoding='utf-8') as f:
                json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
        summary.append({k: diag[k] for k in diag if k != 'tracks'})
        print(f'>>> {v}: {diag["alerts"]} alertů — {diag["reason"]} '
              f'(tracky zavazadel {diag["carry_tracks"]}, ≥2 s {diag["carry_tracks_ge_2s"]}, '
              f'max bez vlastníka {diag["max_alone_sec"]:.1f} s)', flush=True)
        with open(OUT_DIR / 'souhrn.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
