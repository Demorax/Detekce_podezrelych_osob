"""
Přepočítá pole `behaviors` v už uložených *_data.npz ze segmentů v *_tracking.json.

Starý export labeleru (label_skeletons.py / label_skeletons_improved.py) dával každému
framu mimo označený segment behavior = 0 (walking), takže neolabelované framy vypadaly
jako chůze. Opravený export dává -1 (UNLABELED). Tento skript to samé udělá zpětně
pro existující data a vypíše statistiku tříd před a po.

Skelety se nemění, jen `behaviors` (pořadí framů = tracked_path v JSON, stejně jako při exportu).

Použití:
    python fix_labeled_behaviors.py                 # jen statistika, nic nezapisuje
    python fix_labeled_behaviors.py --write         # přepíše behaviors v npz
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

UNLABELED = -1
NAMES = {UNLABELED: 'neolabelováno', 0: 'walking', 1: 'suspicious', 2: 'running'}
DIRS = ['labeled_behaviors', 'labeled_behaviors_05', 'labeled_behaviors_improved']


def relabel(meta):
    out = []
    for frame_num, _ in meta['tracked_path']:
        b = UNLABELED
        for seg in meta['behavior_segments']:
            if seg['start'] <= frame_num <= seg['end']:
                b = seg['behavior']
                break
        out.append(b)
    return np.array(out)


def fmt(c):
    return ', '.join(f'{NAMES[k]} {c.get(k, 0)}' for k in (0, 1, 2, UNLABELED))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()

    for d in DIRS:
        tot_old, tot_new = Counter(), Counter()
        for npz_path in sorted(glob.glob(os.path.join(d, '*_data.npz'))):
            js = npz_path.replace('_data.npz', '_tracking.json')
            with open(js, encoding='utf-8') as f:
                meta = json.load(f)
            data = dict(np.load(npz_path))
            old, new = data['behaviors'], relabel(meta)
            assert len(old) == len(new), f'{npz_path}: délka {len(old)} != tracked_path {len(new)}'
            # framy uvnitř segmentů se nesmí změnit, jen ty mimo (0 -> -1)
            inside = new != UNLABELED
            assert (old[inside] == new[inside]).all(), f'{npz_path}: nesouhlasí framy v segmentech'
            tot_old.update(old.tolist()); tot_new.update(new.tolist())
            segs = [(s['start'], s['end'], s['behavior']) for s in meta['behavior_segments']]
            print(f'  {os.path.basename(npz_path):<28} {len(new):>3} framů, segmenty {segs}, '
                  f'neolabelováno {int((~inside).sum())}')
            if args.write:
                data['behaviors'] = new.astype(old.dtype)
                np.savez(npz_path, **data)
        if tot_old:
            print(f'{d}:\n    dřív: {fmt(tot_old)}\n    teď:  {fmt(tot_new)}')
    if not args.write:
        print('\n(nic nezapsáno, spusť s --write)')


if __name__ == '__main__':
    main()
