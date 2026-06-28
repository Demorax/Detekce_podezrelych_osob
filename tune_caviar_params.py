"""
Grid search nad parametry abandoned-object detector na CAVIAR GT.
Hledá optimální (abandon_sec, radius_factor, min_ownership_frames) podle F1.

Rychlé (sekundy per kombinace) díky rule-only eval bez YOLO detection.
"""
import itertools
import json
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_caviar_rules import evaluate_video

# Grid
ABANDON_SEC_GRID = [2, 3, 4, 5, 7, 10]
RADIUS_FACTOR_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
MIN_OWN_FRAMES_GRID = [1, 2, 3]

# Helper: aggregate counts over all videos for one config
def evaluate_all(abandon_sec, radius_factor, min_own_frames):
    xml_paths = sorted(Path('data/CAVIAR').glob('*/[!.]*.xml'))
    results = []
    for xp in xml_paths:
        r = evaluate_video(
            str(xp),
            abandon_sec=abandon_sec,
            radius_factor=radius_factor,
            min_ownership_frames=min_own_frames,
            max_obj_missed=15,
            verbose=False,
        )
        results.append(r)

    counts = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'FP_BASELINE': 0}
    latencies = []
    for r in results:
        counts[r['outcome']] = counts.get(r['outcome'], 0) + 1
        if r['outcome'] == 'TP' and 'latency_sec' in r:
            latencies.append(r['latency_sec'])

    tp, fp_bl, fn = counts['TP'], counts.get('FP_BASELINE', 0), counts['FN']
    # CAVIAR má FP_BASELINE (alert tam kde GT žádná abandonment) — to je vlastně FP
    fp = counts['FP'] + fp_bl
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    mean_latency = float(np.mean(latencies)) if latencies else None

    return {
        'config': f'abandon={abandon_sec}, radius={radius_factor}, min_own={min_own_frames}',
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': counts['TN'],
        'precision': precision, 'recall': recall, 'f1': f1,
        'mean_latency_sec': mean_latency,
    }


def main():
    combinations = list(itertools.product(ABANDON_SEC_GRID, RADIUS_FACTOR_GRID, MIN_OWN_FRAMES_GRID))
    print(f'Grid search: {len(combinations)} kombinací\n')

    all_results = []
    for i, (asec, rfact, mof) in enumerate(combinations, 1):
        try:
            r = evaluate_all(asec, rfact, mof)
            all_results.append(r)
            if i % 10 == 0 or i == len(combinations):
                print(f'  [{i}/{len(combinations)}] done')
        except Exception as e:
            print(f'  [{i}/{len(combinations)}] ERROR: {e}')

    # Top by F1
    print('\n=== Top 15 by F1 ===')
    print(f'{"config":<55} {"TP":>3} {"FP":>3} {"FN":>3} {"prec":>6} {"recall":>7} {"F1":>6} {"lat":>5}')
    print('-' * 100)
    sorted_results = sorted(all_results, key=lambda r: (-r['f1'], r['mean_latency_sec'] or 999))
    for r in sorted_results[:15]:
        lat = f"{r['mean_latency_sec']:.1f}" if r['mean_latency_sec'] is not None else '-'
        print(f'{r["config"]:<55} {r["TP"]:>3} {r["FP"]:>3} {r["FN"]:>3} '
              f'{r["precision"]:>6.2%} {r["recall"]:>7.2%} {r["f1"]:>6.3f} {lat:>5}')

    print('\n=== Pareto front (precision = 100%) ===')
    pareto = [r for r in all_results if r['precision'] == 1.0]
    pareto = sorted(pareto, key=lambda r: -r['recall'])
    print(f'{"config":<55} {"TP":>3} {"FN":>3} {"recall":>7} {"F1":>6} {"lat":>5}')
    print('-' * 100)
    for r in pareto[:10]:
        lat = f"{r['mean_latency_sec']:.1f}" if r['mean_latency_sec'] is not None else '-'
        print(f'{r["config"]:<55} {r["TP"]:>3} {r["FN"]:>3} '
              f'{r["recall"]:>7.2%} {r["f1"]:>6.3f} {lat:>5}')

    # Save
    with open('caviar_tuning_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved: caviar_tuning_results.json ({len(all_results)} combinations)')


if __name__ == '__main__':
    main()
