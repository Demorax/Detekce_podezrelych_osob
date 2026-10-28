"""
Leave-one-video-out vyhodnocení pravidel pro opuštěná zavazadla na CAVIAR (GT boxy).

tune_caviar_params.py ladí parametry na všech videích a stejná videa pak reportuje, takže
„100 % precision / 100 % recall“ je výsledek na trénovacích datech. Tady se pro každé video
vybere konfigurace jen podle OSTATNÍCH videí (F1, při shodě nižší latence, pak pořadí v gridu)
a vyhodnotí se na vynechaném videu. Součet přes všechna vynechaná videa je odhad výkonu na
nových datech. Vypisuje se i, které konfigurace se vybíraly (stabilita volby).

Použití:
    python caviar_loo.py
"""
import itertools
import json
from collections import Counter
from pathlib import Path

from eval_caviar_rules import evaluate_video
from tune_caviar_params import ABANDON_SEC_GRID, MIN_OWN_FRAMES_GRID, RADIUS_FACTOR_GRID

OUT_JSON = 'caviar_loo_results.json'


def score(outcomes, latencies):
    tp = outcomes.count('TP')
    fp = outcomes.count('FP') + outcomes.count('FP_BASELINE')
    fn = outcomes.count('FN')
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-6)
    lat = sum(latencies) / len(latencies) if latencies else float('inf')
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': outcomes.count('TN'),
            'precision': p, 'recall': r, 'f1': f1, 'mean_latency_sec': lat}


def main():
    xmls = sorted(Path('data/CAVIAR').glob('*/[!.]*.xml'))
    configs = list(itertools.product(ABANDON_SEC_GRID, RADIUS_FACTOR_GRID, MIN_OWN_FRAMES_GRID))
    print(f'{len(xmls)} videí, {len(configs)} konfigurací')

    # výsledek každé konfigurace na každém videu (pravidla jsou deterministická)
    res = {}
    for c in configs:
        a, r, m = c
        res[c] = [evaluate_video(str(x), abandon_sec=a, radius_factor=r, min_ownership_frames=m,
                                 max_obj_missed=15) for x in xmls]

    held, chosen = [], []
    for i, x in enumerate(xmls):
        def train_key(c):
            rs = [v for j, v in enumerate(res[c]) if j != i]
            s = score([v['outcome'] for v in rs], [v['latency_sec'] for v in rs if v['outcome'] == 'TP'])
            return (-s['f1'], s['mean_latency_sec'], configs.index(c))
        best = min(configs, key=train_key)
        v = res[best][i]
        held.append(v)
        chosen.append(best)
        lat = f"{v['latency_sec']:+.1f} s" if v['outcome'] == 'TP' else ''
        print(f'  {v["video"]:<22} vybráno abandon={best[0]}, radius={best[1]}, min_own={best[2]}'
              f'  -> {v["outcome"]} {lat}')

    s = score([v['outcome'] for v in held], [v['latency_sec'] for v in held if v['outcome'] == 'TP'])
    print(f"\nLOO: TP={s['TP']} FP={s['FP']} FN={s['FN']} TN={s['TN']}  "
          f"precision={s['precision']:.0%} recall={s['recall']:.0%} F1={s['f1']:.3f} "
          f"latence {s['mean_latency_sec']:.1f} s")
    print('vybírané konfigurace:', dict(Counter(f'{a}/{r}/{m}' for a, r, m in chosen)))

    # citlivost: kolik konfigurací z gridu dává na všech videích stejné F1 jako nejlepší
    full = {c: score([v['outcome'] for v in res[c]],
                     [v['latency_sec'] for v in res[c] if v['outcome'] == 'TP']) for c in configs}
    best_f1 = max(v['f1'] for v in full.values())
    n_best = sum(v['f1'] == best_f1 for v in full.values())
    print(f'na všech videích má F1={best_f1:.3f} celkem {n_best} z {len(configs)} konfigurací')

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'loo': s, 'per_video': [{'video': v['video'], 'outcome': v['outcome'],
                                            'latency_sec': v.get('latency_sec'), 'config': list(c)}
                                           for v, c in zip(held, chosen)],
                   'n_configs_at_best_f1': n_best, 'n_configs': len(configs)}, f, indent=1,
                  ensure_ascii=False)
    print(f'-> {OUT_JSON}')


if __name__ == '__main__':
    main()
