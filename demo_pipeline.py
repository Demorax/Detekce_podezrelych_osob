"""
Spojená pipeline / demo (bod 4 plánu praktické části): z jednoho videa všechny moduly práce najednou.

Na každém vzorkovaném snímku (5 snímků/s):
  1. osoby (RT-DETR) + skelety (ViTPose-Huge, CrowdPose 14 bodů); --fast = jeden průchod RT-DETR a fp16 pózy,
  2. objekty: YOLO11x COCO (zbraně, zavazadla) + vlastní detektor zbraní v2 (gun/knife/melee)
     + detektor nově odložených nehybných předmětů z modelu pozadí (static_object_detector.py),
  3. tracker osob a objektů, vlastnictví zavazadel, pravidlo opuštění, ARMED (zbraň u zápěstí),
  4. chování davu: ResNet-18 + optický tok + osoby (YOLO11x) → Transformer nad posledními 16 snímky (3,2 s),
     5 tříd MED (úloha 1 zadání),
  5. rvačka: BiLSTM s attention nad posledními 10 skelety každé osoby (2 s), skóre scény = průměr osob
     (úloha 3 zadání),
  6. anomální póza: GRU autoenkodér nad posledními 12 skelety osoby, anomálie = chyba nad 99. percentilem
     normálních trénovacích oken (úloha 2 zadání).
Modely chování: models/behavior/ (train_demo_models.py, demo videa 5, 10, 15 nebyla v tréninku).

Výstup: <out>/<video>_demo.mp4 (plná snímková frekvence, anotace z posledního zpracovaného snímku)
        <out>/<video>_events.json (stav po vzorkovaných snímcích + seznam poplachů + změřené časy).
Použití:
    python demo_pipeline.py --video data/Motion_Emotion/005.mp4
    python demo_pipeline.py --video data/Motion_Emotion/010.mp4 --fast
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

import abandoned_object_detector as AOD
import med_pose_anomaly as MP
import med_train as MT
import run_train_lstm as RL
import suspicious_detector as SD
from med_features import flow_features
from med_person_features import person_features
from static_object_detector import StaticObjectDetector

DEV = 'cuda'
SAMPLE_FPS = 5
FLOW_GAP = 3
BEH = Path('models/behavior')
CZ = {'Neutral': 'neutrální', 'Panic': 'panika', 'Fight': 'rvačka', 'Congestion': 'kongesce', 'Obstacle': 'překážka'}
FIGHT_THR = 0.5
# Časové potvrzení poplachů davu (stejný princip jako ARMED): třída musí vydržet CROWD_CONFIRM
# vzorků (1,6 s) a stejná třída se znovu nahlásí nejdřív po CROWD_COOLDOWN_SEC.
CROWD_CONFIRM = 8
CROWD_COOLDOWN_SEC = 10.0
FIGHT_CONFIRM = 3           # rvačka ze skeletů: skóre nad prahem 3 vzorky po sobě (0,6 s)


class Behavior:
    """Modely chování nad stavem pipeline."""

    def __init__(self):
        c = torch.load(BEH / 'crowd.pt', map_location=DEV, weights_only=False)
        self.crowd = MT.Net(c['d_in'], True, arch=c['arch']).to(DEV).eval()
        self.crowd.load_state_dict(c['state_dict'])
        self.c_mu, self.c_sd, self.c_T = c['mu'], c['sd'], c['window']
        p = torch.load(BEH / 'pose_ae.pt', map_location=DEV, weights_only=False)
        self.ae = MP.GRUAutoencoder().to(DEV).eval()
        self.ae.load_state_dict(p['state_dict'])
        self.p_mu, self.p_sd, self.p_T, self.p_thr = p['mu'].to(DEV), p['sd'].to(DEV), p['window'], p['threshold_p99']
        f = torch.load(BEH / 'fight.pt', map_location=DEV, weights_only=False)
        self.fight = RL.BiLSTMAttention(input_dim=RL.N_KEYPOINTS * 4).to(DEV).eval()
        self.fight.load_state_dict(f['state_dict'])
        self.f_T = f['seq_len']
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet = torchvision.models.resnet18(weights=w)
        self.resnet.fc = torch.nn.Identity()
        self.resnet.eval().to(DEV)
        self.rm = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
        self.rs = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
        self.crowd_buf = deque(maxlen=self.c_T)

    @torch.no_grad()
    def crowd_step(self, frame, prev_frame, yolo_person_boxes):
        g = lambda im: cv2.cvtColor(cv2.resize(im, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
        fl = flow_features(g(prev_frame), g(frame))
        x = torch.from_numpy(cv2.cvtColor(cv2.resize(frame, (455, 256)), cv2.COLOR_BGR2RGB)).to(DEV)
        cnn = self.resnet(((x.permute(2, 0, 1)[None].float() / 255) - self.rm) / self.rs)[0].cpu().numpy()
        pf = person_features(yolo_person_boxes, frame.shape[1], frame.shape[0])
        self.crowd_buf.append((np.concatenate([fl, cnn, pf]) - self.c_mu) / self.c_sd)
        if len(self.crowd_buf) < self.c_T // 2:
            return None
        w = np.stack(list(self.crowd_buf))
        w = np.concatenate([np.repeat(w[:1], self.c_T - len(w), 0), w]) if len(w) < self.c_T else w
        return torch.softmax(self.crowd(torch.from_numpy(w[None]).float().to(DEV)), 1)[0].cpu().numpy()

    @torch.no_grad()
    def fight_prob(self, skels):
        """skels: list (T, 14, 3) posledních skeletů jedné osoby."""
        sk = np.nan_to_num(np.stack(skels[-self.f_T:])[..., :2].astype(np.float32))
        sk = RL.add_velocity_features(RL.normalize_skeleton(sk)).reshape(self.f_T, -1)
        return float(torch.softmax(self.fight(torch.from_numpy(sk[None]).to(DEV)), 1)[0, 1])

    @torch.no_grad()
    def pose_error(self, skels):
        seq = torch.from_numpy(MP.person_sequence(np.stack(skels[-self.p_T:]))[None]).to(DEV)
        y, tgt = self.ae((seq - self.p_mu) / self.p_sd)
        return float(((y - tgt) ** 2).mean())


from PIL import Image, ImageDraw, ImageFont

_FONTS = {}


def _font(size):
    if size not in _FONTS:
        for f in ('C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
            try:
                _FONTS[size] = ImageFont.truetype(f, size); break
            except OSError:
                continue
        else:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def _texts(img, items):
    """items: (text, (x, y_baseline), BGR, size). OpenCV putText neumí diakritiku, proto PIL."""
    if not items:
        return img
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(im)
    for text, (x, y), col, size in items:
        dr.text((x, y - size), text, font=_font(size), fill=(col[2], col[1], col[0]))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def draw(frame, st, t_sec):
    v = frame.copy()
    tv, tp = [], []                              # texty do obrazu a do panelu
    for p in st['persons']:
        b = p['box']
        col = (0, 0, 255) if p['armed'] else (0, 140, 255) if p['pose_anom'] else (80, 200, 80)
        cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2 if (p['armed'] or p['pose_anom']) else 1)
        sk = p['skeleton']
        if sk is not None and len(sk):
            for a, c in SD.skeleton_connections(len(sk)):
                if sk[a, 2] > 0.3 and sk[c, 2] > 0.3:
                    cv2.line(v, tuple(map(int, sk[a, :2])), tuple(map(int, sk[c, :2])), col, 1)
        tag = ('ARMED ' if p['armed'] else '') + ('póza! ' if p['pose_anom'] else '') + \
              (f'rvačka {p["fight"]:.0%}' if p['fight'] is not None and p['fight'] > FIGHT_THR else '')
        if tag:
            tv.append((tag.strip(), (int(b[0]), int(b[1]) - 3), col, 13))
    for o in st['objects']:
        b = o['box']
        col = {'static_object': (255, 0, 255)}.get(o['label'], (0, 0, 255) if o['label'] in AOD.HIGH_THREAT_LABELS else (255, 160, 0))
        cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2 if o['abandoned'] else 1)
        tv.append((('OPUŠTĚNO ' if o['abandoned'] else '') + o['label'], (int(b[0]), int(b[3]) + 13), col, 12))
    # panel
    h, w = v.shape[:2]
    pw = 250
    panel = np.full((h, pw, 3), 30, np.uint8)
    y = 22
    tp.append((f't = {t_sec:5.1f} s', (10, y), (255, 255, 255), 16)); y += 26
    tp.append(('Chování davu (3,2 s):', (10, y), (200, 200, 200), 14)); y += 8
    if st['crowd'] is not None:
        top = int(np.argmax(st['crowd']))
        for k, cname in enumerate(MT.CLASSES):
            y += 20
            p = st['crowd'][k]
            cv2.rectangle(panel, (95, y - 12), (95 + int(140 * p), y - 2), (0, 200, 255) if k == top else (110, 110, 110), -1)
            tp.append((CZ[cname], (10, y - 1), (255, 255, 255), 13))
    y += 30
    fs = st['fight_scene']
    tp.append((f'Rvačka (skelety): {"-" if fs is None else f"{fs:.0%}"}', (10, y),
               (0, 0, 255) if fs is not None and fs > FIGHT_THR else (200, 200, 200), 14)); y += 22
    tp.append((f'Osob: {len(st["persons"])}   ARMED: {sum(p["armed"] for p in st["persons"])}', (10, y), (200, 200, 200), 14)); y += 22
    tp.append((f'Anomální póza: {sum(p["pose_anom"] for p in st["persons"])}', (10, y), (200, 200, 200), 14)); y += 30
    tp.append(('Poplachy:', (10, y), (200, 200, 200), 14))
    for a in st['alerts'][-8:]:
        y += 18
        tp.append((f'{a["t"]:5.1f} s  {a["type"]}', (10, y), (80, 80, 255), 13))
    return np.hstack([_texts(v, tv), _texts(panel, tp)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--fast', action='store_true', help='jeden průchod RT-DETR + fp16 ViTPose')
    ap.add_argument('--out', default='demo_outputs/full')
    ap.add_argument('--abandon-sec', type=float, default=AOD.DEFAULT_ABANDON_SEC)
    ap.add_argument('--max-sec', type=float, default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = Path(a.video).stem

    det = SD.SuspiciousDetector(device=DEV)
    SD.print = lambda *x, **k: None           # better_detection tiskne každou variantu
    if a.fast:
        det.rtdetr.better_detection = lambda img: det.rtdetr.detect(img, conf_threshold=0.3)
        _kp = det.vitpose.extract_keypoints
        def kp16(img, boxes, **k):
            with torch.autocast('cuda', dtype=torch.float16):
                return _kp(img, boxes, **k)
        det.vitpose.extract_keypoints = kp16
    beh = Behavior()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / SAMPLE_FPS)))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out / f'{name}_demo.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W + 250, H))

    pt = AOD.PersonTracker(max_dist=AOD.DEFAULT_MAX_PERSON_TRACK_DIST)
    ot = AOD.ObjectTracker(iou_threshold=AOD.DEFAULT_OBJ_IOU_THRESHOLD, max_missed=AOD.DEFAULT_MAX_OBJ_MISSED)
    sod = None
    hist = {}                                  # tid -> deque[(sampled_idx, skeleton)]
    recent = deque(maxlen=FLOW_GAP + 1)        # posledních pár snímků pro optický tok
    events, alerts, timings = [], [], []
    st = {'persons': [], 'objects': [], 'crowd': None, 'fight_scene': None, 'alerts': alerts}
    armed_prev, fight_run, fight_last_alert = set(), 0, -1e9
    crowd_run, crowd_last_alert = ('Neutral', 0), {}
    i = si = 0
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_sec and i / fps > a.max_sec):
            break
        recent.append(frame)
        if i % step == 0:
            t0 = time.perf_counter()
            t_sec = i / fps
            d = det.detect(frame)
            if sod is None:
                sod = StaticObjectDetector(frame.shape, fps=SAMPLE_FPS)
            obj = [(np.array(b), s, l) for tier in ('high', 'medium', 'carry') for b, s, l in d[tier]]
            obj += sod.update(frame, d['person_boxes'] if len(d['person_boxes']) else None)
            ptr = pt.update(si, d['person_boxes'] if len(d['person_boxes']) else np.zeros((0, 5)), d.get('skeletons'))
            otr = ot.update(si, obj)
            AOD.update_associations(si, ptr, otr, radius_factor=AOD.DEFAULT_OWNERSHIP_RADIUS_FACTOR)
            AOD.update_armed_status(ptr, otr)
            for al in AOD.check_abandonment(si, SAMPLE_FPS, otr, abandon_sec=a.abandon_sec,
                                            min_ownership_frames=AOD.DEFAULT_MIN_OWNERSHIP_FRAMES):
                alerts.append({'t': round(t_sec, 1), 'type': f'opuštěno: {al["label"]}', 'box': al['box']})
            # chování
            yp = det.coco(frame, classes=[0], conf=0.25, imgsz=1280, device=DEV, verbose=False)[0]
            yb = np.concatenate([yp.boxes.xyxy.cpu().numpy(), yp.boxes.conf.cpu().numpy()[:, None]], 1)
            crowd = beh.crowd_step(frame, recent[0], yb)
            persons, fights = [], []
            for tid, p in ptr.items():
                sk = p.get('skeleton')
                hq = hist.setdefault(tid, deque(maxlen=max(beh.f_T, beh.p_T)))
                if hq and hq[-1][0] != si - 1:
                    hq.clear()                 # jen souvislé úseky jako v tréninku
                if sk is not None and len(sk) == 14:
                    hq.append((si, sk))
                sks = [s for _, s in hq]
                fp = beh.fight_prob(sks) if len(sks) >= beh.f_T else None
                pe = beh.pose_error(sks) if len(sks) >= beh.p_T else None
                if fp is not None:
                    fights.append(fp)
                persons.append({'id': tid, 'box': p['box'].tolist(), 'skeleton': sk, 'armed': bool(p.get('armed')),
                                'fight': fp, 'pose_err': pe, 'pose_anom': pe is not None and pe > beh.p_thr})
            for tid in [t for t in hist if t not in pt.tracks]:
                del hist[tid]
            fight_scene = float(np.mean(fights)) if fights else None
            armed_now = {p['id'] for p in persons if p['armed']}
            for tid in armed_now - armed_prev:
                alerts.append({'t': round(t_sec, 1), 'type': f'ARMED osoba {tid}'})
            armed_prev = armed_now
            fight_run = fight_run + 1 if fight_scene is not None and fight_scene > FIGHT_THR else 0
            if fight_run == FIGHT_CONFIRM and t_sec - fight_last_alert >= CROWD_COOLDOWN_SEC:
                alerts.append({'t': round(t_sec, 1), 'type': 'rvačka (skelety)'})
                fight_last_alert = t_sec
            if crowd is not None:
                cname = MT.CLASSES[int(np.argmax(crowd))]
                crowd_run = (cname, crowd_run[1] + 1 if crowd_run[0] == cname else 1)
                if (cname != 'Neutral' and crowd_run[1] == CROWD_CONFIRM
                        and t_sec - crowd_last_alert.get(cname, -1e9) >= CROWD_COOLDOWN_SEC):
                    alerts.append({'t': round(t_sec, 1), 'type': f'dav: {CZ[cname]}'})
                    crowd_last_alert[cname] = t_sec
            objects = [{'box': np.asarray(o['box']).tolist(), 'label': o['label'], 'abandoned': bool(o.get('alerted'))}
                       for o in otr.values()]
            st = {'persons': persons, 'objects': objects, 'crowd': crowd, 'fight_scene': fight_scene, 'alerts': alerts}
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)
            events.append({'t': round(t_sec, 2), 'crowd': None if crowd is None else dict(zip(MT.CLASSES, map(float, crowd))),
                           'fight_scene': fight_scene, 'n_persons': len(persons), 'armed': sorted(armed_now),
                           'pose_anomalous': [p['id'] for p in persons if p['pose_anom']],
                           'objects': [(o['label'], o['abandoned']) for o in objects]})
            si += 1
            if si % 25 == 0:
                print(f'  {t_sec:6.1f} s  {np.mean(timings[-25:]) * 1000:.0f} ms/snímek', flush=True)
        writer.write(draw(frame, st, i / fps))
        i += 1
    writer.release()
    res = {'video': a.video, 'mode': 'fast' if a.fast else 'precise', 'sampled_frames': si,
           'mean_ms_per_sampled_frame': float(np.mean(timings) * 1000),
           'fps_processing': float(1 / np.mean(timings)), 'realtime_factor_at_5fps': float(5 * np.mean(timings)),
           'alerts': alerts, 'frames': events}
    with open(out / f'{name}_events.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f'-> {out / (name + "_demo.mp4")}  ({res["mean_ms_per_sampled_frame"]:.0f} ms/snímek, '
          f'{len(alerts)} poplachů)')
    for al in alerts:
        print(f'   {al["t"]:6.1f} s  {al["type"]}')


if __name__ == '__main__':
    main()
