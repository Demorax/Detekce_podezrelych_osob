"""
Falešné rvačky obrazového modelu RWF-2000 (fight_video.pt) v MED demu: kalibrace prahu na scéně.

Model rvaček je doučený na RWF-2000 (bližší kamery, velké postavy). Na MED (shora snímaný dav, malé postavy)
dával s prahem 0,5 v demu 010 dva falešné poplachy. Místo ladění na demu se práh kalibruje na normálním
provozu stejného typu kamery, stejně jako práh autoenkodéru póz (p99 normálních oken): na 28 MED videích
mimo demo (5, 10, 15) se spočítá P(rvačka) přesně jako v demu (posledních 5 s, 16 snímků, 1x za sekundu)
a práh = 99. percentil P na oknech bez rvačky. Na tato videa model nikdy netrénoval, takže kalibrace je čistá.

Pravidla poplachu (pevně předem, potvrzení 2 vyhodnocení po sobě a odstup 10 s jako v demu):
  R0  práh 0,5 (dosavadní),
  R1  práh p99 normálních oken MED (kalibrovaný),
  R2  práh 0,5 a zároveň model davu (MED, třída Fight) v tomtéž vzorku P(Fight) > 0,5 (hradlo podle scény).
      Na kalibračních videích se pro hradlo berou predikce modelu davu z křížové validace (med_video_results.json,
      seed 0, model video neviděl); okno davu je tam středové, v demu kauzální -> P v čase k = predikce se
      středem k - 7. Na demu skutečné P(Fight) z demo_outputs (crowd_video.pt demo videa neviděl).
  R3  pro srovnání jen model davu: P(Fight) > 0,5 se stejným potvrzením (bez obrazového modelu RWF).
Hodnotí se na kalibračních videích (falešné poplachy za hodinu mimo rvačku, zachycené úseky rvaček)
a pak na demo videích přehráním skutečných P(rvačka) uložených v demo_outputs/full_v2/*_events.json.
Poplach se počítá jako správný, když v okně [t - 5 s, t] je label Fight (okno modelu je 5 s).

Použití:
    python med_fight_transfer.py        # -> med_fight_transfer_results.json
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torchvision.models.video import mvit_v2_s

import med_train as MT

FEAT, FRAMES, DEMO = Path('med_features'), Path('med_video_frames'), Path('demo_outputs/full_v2')
DEMO_VIDS = [5, 10, 15]
FPS, WIN, EVERY, CONFIRM, COOLDOWN = 5, 25, 5, 2, 10.0     # stejné jako demo_pipeline.py
FIGHT = MT.CLASSES.index('Fight')
DEV = 'cuda'


def labels(v):
    return np.array([MT.MED_TO_CLS.get(int(l), -1) for l in np.load(FEAT / f'{v:03d}.npz')['label']])


@torch.no_grad()
def probs_from_frames(model, ck, v):
    """P(rvačka) v každém 5. vzorku k, okno posledních WIN vzorků (od poloviny plnosti), jako video_fight_step."""
    fr = np.load(FRAMES / f'{v:03d}.npy', mmap_mode='r')
    mean = torch.tensor(ck['mean'], device=DEV).view(3, 1, 1, 1)
    std = torch.tensor(ck['std'], device=DEV).view(3, 1, 1, 1)
    out = {}
    for k in range(0, len(fr), EVERY):
        buf = fr[max(0, k - WIN + 1):k + 1]
        if len(buf) < WIN // 2:
            continue
        edges = np.linspace(0, len(buf), ck['frames'] + 1)
        clip = np.stack([buf[min(int(i), len(buf) - 1)] for i in (edges[:-1] + edges[1:]) / 2])
        x = torch.from_numpy(clip).to(DEV).permute(3, 0, 1, 2).float() / 255
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out[k] = float(torch.softmax(model(((x - mean) / std)[None]).float(), 1)[0, 1])
    return out


def alarms(p, thr):
    """p: {k: P}; poplach při CONFIRM vyhodnoceních nad prahem po sobě, odstup COOLDOWN s."""
    res, run, last = [], 0, -1e9
    for k in sorted(p):
        run = run + 1 if p[k] > thr else 0
        if run == CONFIRM and k / FPS - last >= COOLDOWN:
            res.append(k); last = k / FPS
    return res


def judge(al, lab):
    """Každý poplach: správný, když v posledních 5 s je Fight. Úsek rvačky zachycen, když v něm (nebo do 5 s
    po jeho konci) padne poplach."""
    ok = [bool((lab[max(0, k - WIN + 1):k + 1] == FIGHT).any()) for k in al]
    seg, segs = None, []
    for k, l in enumerate(list(lab) + [-2]):
        if l == FIGHT and seg is None:
            seg = k
        elif l != FIGHT and seg is not None:
            segs.append((seg, k - 1)); seg = None
    hit = [any(a <= k <= b + WIN for k in al) for a, b in segs]
    return {'alarms_s': [round(k / FPS, 1) for k in al], 'correct': ok, 'false': int(len(ok) - sum(ok)),
            'fight_segments': len(segs), 'fight_segments_hit': int(sum(hit))}


def main():
    ck = torch.load('models/behavior/fight_video.pt', map_location=DEV, weights_only=False)
    model = mvit_v2_s()
    model.head[-1] = torch.nn.Linear(model.head[-1].in_features, 2)
    model.load_state_dict(ck['state_dict'])
    model.eval().to(DEV)
    vids = sorted(int(p.stem) for p in FEAT.glob('*.npz'))
    calib = [v for v in vids if v not in DEMO_VIDS]
    P, L = {}, {}
    for v in vids:
        P[v], L[v] = probs_from_frames(model, ck, v), labels(v)
        print(f'{v:03d}: {len(P[v])} vyhodnocení, max P {max(P[v].values()):.2f}', flush=True)

    def flat(vs, cond):
        return np.array([p for v in vs for k, p in P[v].items() if cond(L[v][k])])
    normal = flat(calib, lambda l: l >= 0 and l != FIGHT)
    fight = flat(calib, lambda l: l == FIGHT)
    y = np.r_[np.zeros(len(normal)), np.ones(len(fight))]
    auc = float(roc_auc_score(y, np.r_[normal, fight]))
    thr = {'R0': 0.5, 'R1': float(np.percentile(normal, 99))}
    print(f'kalibrace ({len(calib)} videí): AUC rvačka vs. ostatní {auc:.3f}, '
          f'normálních oken {len(normal)}, rvaček {len(fight)}; práh p99 = {thr["R1"]:.3f}')
    for q in (95, 99.5):
        print(f'  (citlivost, nepoužito) p{q} = {np.percentile(normal, q):.3f}')

    res = {'calibration_videos': calib, 'auc_fight_vs_rest': auc, 'n_normal': int(len(normal)),
           'n_fight': int(len(fight)), 'thresholds': thr, 'rules': {}}
    oof = json.load(open('med_video_results.json', encoding='utf-8'))['runs'][0]['pred']
    crowd_fight = {}
    for (v, k), pr in zip(oof['ids'], oof['prob']):
        crowd_fight.setdefault(v, {})[k] = pr[FIGHT]
    gate = {v: {k: p if crowd_fight.get(v, {}).get(k - 7, 0.0) > 0.5 else 0.0 for k, p in P[v].items()}
            for v in calib}
    demo_P, demo_gate = {}, {}
    for v in DEMO_VIDS:
        ev = json.load(open(DEMO / f'{v:03d}_events.json', encoding='utf-8'))['frames']
        demo_P[v] = {k: e['fight_video'] for k, e in enumerate(ev) if k % EVERY == 0 and e['fight_video'] is not None}
        demo_gate[v] = {k: p if ev[k]['crowd'] and ev[k]['crowd']['Fight'] > 0.5 else 0.0 for k, p in demo_P[v].items()}
        common = [k for k in demo_P[v] if k in P[v]]
        diff = np.abs(np.array([demo_P[v][k] - P[v][k] for k in common]))
        print(f'demo {v:03d}: P z dema vs. ze snímků MED, průměrný rozdíl {diff.mean():.3f}, max {diff.max():.3f}')
    rules = {n: (t, P, demo_P) for n, t in thr.items()}
    rules['R2'] = (0.5, gate, demo_gate)
    crowd_only = {v: {k: crowd_fight.get(v, {}).get(k - 7, 0.0) for k in P[v]} for v in calib}
    demo_crowd = {}
    for v in DEMO_VIDS:
        ev = json.load(open(DEMO / f'{v:03d}_events.json', encoding='utf-8'))['frames']
        demo_crowd[v] = {k: ev[k]['crowd']['Fight'] if ev[k]['crowd'] else 0.0 for k in demo_P[v]}
    rules['R3'] = (0.5, crowd_only, demo_crowd)
    for name, (t, PP, DP) in rules.items():
        cal = [judge(alarms(PP[v], t), L[v]) for v in calib]
        n_false = sum(c['false'] for c in cal)
        hours = sum(int(((L[v] >= 0) & (L[v] != FIGHT)).sum()) for v in calib) / FPS / 3600
        segs, hit = sum(c['fight_segments'] for c in cal), sum(c['fight_segments_hit'] for c in cal)
        demo = {f'{v:03d}': judge(alarms(DP[v], t), L[v]) for v in DEMO_VIDS}
        res['rules'][name] = {'threshold': t, 'calib_false_alarms': n_false,
                              'calib_false_per_hour': n_false / hours, 'calib_fight_segments': segs,
                              'calib_fight_segments_hit': hit, 'demo': demo}
        print(f'{name} (práh {t:.3f}): kalibrace {n_false} falešných ({n_false / hours:.1f}/h), '
              f'rvaček zachyceno {hit}/{segs}')
        for v, d in demo.items():
            print(f'   demo {v}: poplachy {d["alarms_s"]}, správně {d["correct"]}, rvačky {d["fight_segments_hit"]}/'
                  f'{d["fight_segments"]}')
    with open('med_fight_transfer_results.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
