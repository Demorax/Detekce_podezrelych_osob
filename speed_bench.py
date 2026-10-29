"""
Rychlost zpracování (metrika ze zadání práce): čas každého modulu pipeline a celkové FPS,
ve dvou režimech, na reálných záběrech MED (hustý dav 005, klidná scéna 021) a ShanghaiTech
(málo osob).

Režimy:
  přesný  = konfigurace použitá ve výsledcích práce: RT-DETR na 5 variantách snímku (better_detection),
            ViTPose-Huge fp32, YOLO11x objekty na 5 variantách, model zbraní fp32.
  rychlý  = RT-DETR jeden průchod, ViTPose-Huge v autocast fp16, YOLO11x i model zbraní jeden průchod fp16.
Pro oba režimy se počítají i příznaky davu (ResNet-18 + optický tok) a časové modely
(BiLSTM / Transformer davu, BiLSTM rvaček, GRU autoenkodér póz) a tracker + pravidla.

Dopad rychlého režimu na kvalitu: shoda detekcí osob s přesným režimem (recall boxů při IoU 0,5)
a průměrná odchylka kloubů spárovaných osob (px), měřeno na stejných snímcích.

Měřit na VOLNÉM GPU (jinak jsou časy zkreslené). Výstup speed_results.json + tabulka.
Použití:
    python speed_bench.py                 # 60 snímků z každé scény
    python speed_bench.py --frames 5      # rychlá kontrola, že vše běží
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch

import med_pose_anomaly as MP
import med_train as MT
from med_features import flow_features, FLOW_GAP

DEV = 'cuda'
SCENES = {
    'MED 005 (bitka, hustý dav)': ('video', 'data/Motion_Emotion/005.mp4', 840),
    'MED 021 (klidná scéna)': ('video', 'data/Motion_Emotion/021.mp4', 300),
    'ShanghaiTech 01_0014 (málo osob)': ('frames', 'data/ShanghaiTech/shanghaitech/testing/frames/01_0014', 0),
}


def sync():
    torch.cuda.synchronize()


def timed(fn, *a, **k):
    sync(); t = time.perf_counter(); r = fn(*a, **k); sync()
    return r, (time.perf_counter() - t) * 1000


def load_frames(kind, path, start, n, step=6):
    frames = []
    if kind == 'video':
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = 0
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            if i % step == 0:
                nxt = None
                for _ in range(FLOW_GAP):
                    ok2, nxt = cap.read(); i += 1
                frames.append((f, nxt if ok2 else f))
                i += 1
                continue
            i += 1
    else:
        files = sorted(Path(path).glob('*.jpg'))
        for j in range(0, len(files) - FLOW_GAP, 5):
            frames.append((cv2.imread(str(files[j])), cv2.imread(str(files[j + FLOW_GAP]))))
            if len(frames) >= n:
                break
    return frames


def iou_match(a, b, thr=0.5):
    """recall boxů a (reference) v b + spárování indexů."""
    pairs = []
    if len(a) == 0:
        return 1.0, pairs
    if len(b) == 0:
        return 0.0, pairs
    used = set()
    for i, x in enumerate(a):
        best, bj = 0, -1
        for j, y in enumerate(b):
            if j in used:
                continue
            ix = max(0, min(x[2], y[2]) - max(x[0], y[0])); iy = max(0, min(x[3], y[3]) - max(x[1], y[1]))
            inter = ix * iy
            u = (x[2] - x[0]) * (x[3] - x[1]) + (y[2] - y[0]) * (y[3] - y[1]) - inter
            if inter / (u + 1e-9) > best:
                best, bj = inter / (u + 1e-9), j
        if best >= thr:
            used.add(bj); pairs.append((i, bj))
    return len(pairs) / len(a), pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=60)
    a = ap.parse_args()
    import suspicious_detector as SD
    import abandoned_object_detector as AOD
    from ultralytics import YOLO
    import torchvision

    det = SD.SuspiciousDetector(device=DEV)
    rt, vp = det.rtdetr, det.vitpose
    resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = torch.nn.Identity(); resnet.eval().to(DEV)
    rmean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    rstd = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    crowd_lstm = MT.Net(48 + 512 + 24, True, arch='lstm').to(DEV).eval()
    crowd_tr = MT.Net(48 + 512 + 24, True, arch='transformer').to(DEV).eval()
    import run_train_lstm as RL
    fight = RL.BiLSTMAttention(input_dim=RL.N_KEYPOINTS * 4).to(DEV)
    fight.load_state_dict(torch.load('best_lstm_pytorch.pt', map_location=DEV)['model_state_dict']); fight.eval()
    ae = MP.GRUAutoencoder().to(DEV).eval()
    _orig_print = print

    results = {}
    for scene, (kind, path, start) in SCENES.items():
        frames = load_frames(kind, path, start, a.frames)
        # zahřátí
        for f, _ in frames[:2]:
            rt.detect(f); det.coco(f, verbose=False, device=DEV)
        rows = {'přesný': {}, 'rychlý': {}}
        quality = {'recall': [], 'kp_err': [], 'n_pers_precise': [], 'n_pers_fast': []}
        for f, f2 in frames:
            SD.print = lambda *x, **k: None    # better_detection tiskne každou variantu
            # --- osoby
            pb, t_pp = timed(rt.better_detection, f)
            pf, t_pf = timed(rt.detect, f, conf_threshold=0.3)
            SD.print = _orig_print
            # --- póza
            sk_p, t_kp = timed(vp.extract_keypoints, f, pb) if len(pb) else (np.array([]), 0.0)
            def pose_fp16(img, boxes):
                with torch.autocast('cuda', dtype=torch.float16):
                    return vp.extract_keypoints(img, boxes)
            sk_f, t_kf = timed(pose_fp16, f, pf) if len(pf) else (np.array([]), 0.0)
            # --- objekty a zbraně
            _, t_op = timed(det.better_object_detection, f)
            _, t_of = timed(det.coco, f, conf=0.15, classes=det.coco_filter, device=DEV, imgsz=1280, verbose=False, half=True)
            _, t_wp = timed(det.weapon, f, conf=0.2, device=DEV, imgsz=960, verbose=False)
            _, t_wf = timed(det.weapon, f, conf=0.2, device=DEV, imgsz=960, verbose=False, half=True)
            # --- příznaky davu
            def crowd_feats():
                g1 = cv2.cvtColor(cv2.resize(f, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(cv2.resize(f2, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
                fl = flow_features(g1, g2)
                x = torch.from_numpy(cv2.cvtColor(cv2.resize(f, (455, 256)), cv2.COLOR_BGR2RGB)).to(DEV)
                x = x.permute(2, 0, 1)[None].float() / 255
                with torch.no_grad():
                    return fl, resnet((x - rmean) / rstd)
            _, t_cf = timed(crowd_feats)
            # --- časové modely (na jedno vyhodnocení za snímek)
            with torch.no_grad():
                win = torch.randn(1, 16, 584, device=DEV)
                _, t_cl = timed(crowd_lstm, win)
                _, t_ct = timed(crowd_tr, win)
                n_p = max(len(pb), 1)
                _, t_fg = timed(fight, torch.randn(n_p, 15, 56, device=DEV))
                _, t_ae = timed(ae, torch.randn(n_p, 12, 30, device=DEV))
            for mode, vals in (('přesný', dict(osoby=t_pp, poza=t_kp, objekty=t_op, zbrane=t_wp)),
                               ('rychlý', dict(osoby=t_pf, poza=t_kf, objekty=t_of, zbrane=t_wf))):
                vals.update(priznaky_davu=t_cf, model_davu_lstm=t_cl, model_davu_transformer=t_ct,
                            model_rvacek=t_fg, autoenkoder_poz=t_ae)
                for k, v in vals.items():
                    rows[mode].setdefault(k, []).append(v)
            rec, pairs = iou_match(pb[:, :4] if len(pb) else [], pf[:, :4] if len(pf) else [])
            quality['recall'].append(rec)
            quality['n_pers_precise'].append(len(pb)); quality['n_pers_fast'].append(len(pf))
            if len(sk_p) and len(sk_f) and pairs:
                quality['kp_err'].append(float(np.mean([np.linalg.norm(sk_p[i][:, :2] - sk_f[j][:, :2], axis=1).mean()
                                                        for i, j in pairs])))
        res = {}
        for mode, r in rows.items():
            ms = {k: float(np.mean(v)) for k, v in r.items()}
            total = sum(v for k, v in ms.items() if k != 'model_davu_transformer')
            res[mode] = {'ms': ms, 'total_ms': total, 'fps': 1000.0 / total}
        res['kvalita_rychlého'] = {'recall_osob_vs_presny': float(np.mean(quality['recall'])),
                                   'odchylka_kloubu_px': float(np.mean(quality['kp_err'])) if quality['kp_err'] else None,
                                   'osob_na_snimek_presny': float(np.mean(quality['n_pers_precise'])),
                                   'osob_na_snimek_rychly': float(np.mean(quality['n_pers_fast']))}
        res['n_frames'] = len(frames)
        results[scene] = res
        print(f'\n== {scene} ({len(frames)} snímků, {res["kvalita_rychlého"]["osob_na_snimek_presny"]:.1f} osob/snímek)')
        print(f'{"modul":<24}{"přesný ms":>12}{"rychlý ms":>12}')
        for k in rows['přesný']:
            print(f'{k:<24}{res["přesný"]["ms"][k]:>12.1f}{res["rychlý"]["ms"][k]:>12.1f}')
        print(f'{"CELKEM (bez Transf.)":<24}{res["přesný"]["total_ms"]:>12.1f}{res["rychlý"]["total_ms"]:>12.1f}')
        print(f'{"FPS":<24}{res["přesný"]["fps"]:>12.2f}{res["rychlý"]["fps"]:>12.2f}   (realtime při vzorkování 5 snímků/s = 5 FPS)')
        print(f'kvalita rychlého: recall osob {res["kvalita_rychlého"]["recall_osob_vs_presny"]:.2f}, '
              f'odchylka kloubů {res["kvalita_rychlého"]["odchylka_kloubu_px"]} px', flush=True)
    results['gpu'] = torch.cuda.get_device_name(0)
    with open('speed_results.json', 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)
    print('-> speed_results.json')


if __name__ == '__main__':
    main()
