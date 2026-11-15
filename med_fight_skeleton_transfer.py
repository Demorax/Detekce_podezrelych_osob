"""
Falešné rvačky skeletového modelu z RWF-2000 (fight.pt, BiLSTM s attention) v MED demu: totéž jako
med_fight_transfer.py pro obrazový model. V demu v3 hlásí rvačku při panice (005 27,8 s, 010 75,7 s).

Skóre scény přesně jako v demu: každý vzorek (5 snímků/s) P(rvačka) pro každou osobu se souvislou historií
aspoň 10 vzorků (posledních 10 skeletů), skóre scény = průměr přes osoby; poplach po 3 vzorcích nad prahem
po sobě, odstup 10 s. Skelety MED ze stejné pipeline (med_skeletons.py: RT-DETR + ViTPose, 5 snímků/s,
data/Motion_Emotion/skeletons/), model na MED nikdy netrénoval. Kalibrace na 28 videích mimo demo (5, 10, 15).

Pravidla (pevně předem, stejná jako pro obrazový model):
  S0  práh 0,5 (dosavadní),
  S1  práh p99 skóre scény na vzorcích bez rvačky,
  S2  práh 0,5 a zároveň model davu P(Fight) > 0,5 v tomtéž vzorku (predikce z křížové validace, kauzální
      posun -7; v demu skutečné hodnoty),
  S3  pro srovnání jen model davu P(Fight) > 0,5 se stejným potvrzením 3 vzorky.
Poplach správný, když v posledních 2 s (okno modelu) nebo 5 s je label Fight -> bere se 5 s jako u obrazu.

Použití:
    python med_fight_skeleton_transfer.py      # -> med_fight_skeleton_transfer_results.json
"""
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import med_fight_transfer as FT
import run_train_lstm as RL

SKEL, DEMO = Path('data/Motion_Emotion/skeletons'), Path('demo_outputs/full_v3')
CONFIRM, COOLDOWN, FPS, FIGHT = 3, 10.0, FT.FPS, FT.FIGHT
DEV = 'cuda'


@torch.no_grad()
def scene_scores(model, T, v, n):
    """{k: průměr P(rvačka) přes osoby se souvislými posledními T vzorky končícími v k}."""
    d = pickle.load(open(SKEL / f'{v:03d}.pkl', 'rb'))
    per_k = {}
    for tr in d['person_tracks']:
        fi, sk = np.asarray(tr['frame_indices']), np.asarray(tr['skeletons'], np.float32)
        if len(fi) < T:
            continue
        ends = [i for i in range(T - 1, len(fi)) if fi[i] - fi[i - T + 1] == T - 1]
        if not ends:
            continue
        x = np.stack([RL.add_velocity_features(RL.normalize_skeleton(np.nan_to_num(sk[i - T + 1:i + 1, :, :2])))
                      .reshape(T, -1) for i in ends]).astype(np.float32)
        p = torch.softmax(model(torch.from_numpy(x).to(DEV)), 1)[:, 1].cpu().numpy()
        for i, pi in zip(ends, p):
            per_k.setdefault(int(fi[i]), []).append(float(pi))
    return {k: float(np.mean(v_)) for k, v_ in per_k.items() if k < n}


def alarms(p, thr):
    res, run, last, prev = [], 0, -1e9, None
    for k in sorted(p):
        if prev is not None and k != prev + 1:
            run = 0                                   # vzorek bez skóre přeruší řadu jako v demu
        run = run + 1 if p[k] > thr else 0
        if run == CONFIRM and k / FPS - last >= COOLDOWN:
            res.append(k); last = k / FPS
        prev = k
    return res


def main():
    f = torch.load('models/behavior/fight.pt', map_location=DEV, weights_only=False)
    model = RL.BiLSTMAttention(input_dim=RL.N_KEYPOINTS * 4).to(DEV).eval()
    model.load_state_dict(f['state_dict'])
    T = f['seq_len']
    vids = sorted(int(p.stem) for p in SKEL.glob('*.pkl'))
    calib = [v for v in vids if v not in FT.DEMO_VIDS]
    P, L = {}, {}
    for v in calib:
        L[v] = FT.labels(v)
        P[v] = scene_scores(model, T, v, len(L[v]))
        print(f'{v:03d}: {len(P[v])} vzorků se skóre, max {max(P[v].values(), default=0):.2f}', flush=True)
    normal = np.array([p for v in calib for k, p in P[v].items() if L[v][k] >= 0 and L[v][k] != FIGHT])
    fight = np.array([p for v in calib for k, p in P[v].items() if L[v][k] == FIGHT])
    auc = float(roc_auc_score(np.r_[np.zeros(len(normal)), np.ones(len(fight))], np.r_[normal, fight]))
    thr = {'S0': 0.5, 'S1': float(np.percentile(normal, 99))}
    print(f'kalibrace ({len(calib)} videí): AUC rvačka vs. ostatní {auc:.3f}, normálních vzorků {len(normal)}, '
          f'rvaček {len(fight)}; p99 = {thr["S1"]:.3f}')

    oof = json.load(open('med_video_results.json', encoding='utf-8'))['runs'][0]['pred']
    cf = {}
    for (v, k), pr in zip(oof['ids'], oof['prob']):
        cf.setdefault(v, {})[k] = pr[FIGHT]
    gate = {v: {k: p if cf.get(v, {}).get(k - 7, 0.0) > 0.5 else 0.0 for k, p in P[v].items()} for v in calib}
    crowd_only = {v: {k: cf.get(v, {}).get(k - 7, 0.0) for k in range(12, len(L[v]))} for v in calib}
    dP, dG, dC, dL = {}, {}, {}, {}
    for v in FT.DEMO_VIDS:
        ev = json.load(open(DEMO / f'{v:03d}_events.json', encoding='utf-8'))['frames']
        dL[v] = FT.labels(v)
        dP[v] = {k: e['fight_scene'] for k, e in enumerate(ev) if e['fight_scene'] is not None}
        dG[v] = {k: p if ev[k]['crowd'] and ev[k]['crowd']['Fight'] > 0.5 else 0.0 for k, p in dP[v].items()}
        dC[v] = {k: e['crowd']['Fight'] for k, e in enumerate(ev) if e['crowd']}
    rules = {'S0': (0.5, P, dP), 'S1': (thr['S1'], P, dP), 'S2': (0.5, gate, dG), 'S3': (0.5, crowd_only, dC)}
    res = {'calibration_videos': calib, 'auc_fight_vs_rest': auc, 'n_normal': int(len(normal)),
           'n_fight': int(len(fight)), 'thresholds': thr, 'rules': {}}
    hours = sum(int(((L[v] >= 0) & (L[v] != FIGHT)).sum()) for v in calib) / FPS / 3600
    for name, (t, PP, DP) in rules.items():
        cal = [FT.judge(alarms(PP[v], t), L[v]) for v in calib]
        nf = sum(c['false'] for c in cal)
        segs, hit = sum(c['fight_segments'] for c in cal), sum(c['fight_segments_hit'] for c in cal)
        demo = {f'{v:03d}': FT.judge(alarms(DP[v], t), dL[v]) for v in FT.DEMO_VIDS}
        res['rules'][name] = {'threshold': t, 'calib_false_alarms': nf, 'calib_false_per_hour': nf / hours,
                              'calib_fight_segments': segs, 'calib_fight_segments_hit': hit, 'demo': demo}
        print(f'{name} (práh {t:.3f}): kalibrace {nf} falešných ({nf / hours:.1f}/h), rvaček zachyceno {hit}/{segs}')
        for v, d in demo.items():
            print(f'   demo {v}: poplachy {d["alarms_s"]}, správně {d["correct"]}, rvačky {d["fight_segments_hit"]}/'
                  f'{d["fight_segments"]}')
    with open('med_fight_skeleton_transfer_results.json', 'w', encoding='utf-8') as fo:
        json.dump(res, fo, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
