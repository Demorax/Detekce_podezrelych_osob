"""
Rule-based abandoned object detector.

Pipeline:
    1. Per-frame: SuspiciousDetector → persons + objects (multi-pass YOLO11x)
    2. Person tracker (centroid-based, viz run_rwf_batch.track_persons)
    3. Object tracker (IoU-based — objekty se moc nehýbou)
    4. Person-object association (proximity)
    5. Track ownership over time
    6. Abandonment rule: objekt > N sekund bez nejbližší osoby v radiu R → ALERT

Použití:
    python abandoned_object_detector.py --video data/CAVIAR/LeftBag/LeftBag.mpg
    python abandoned_object_detector.py --video <path> --abandon-sec 10 --no-better-obj

Output:
    - Konzole log s alerty
    - Volitelně: <out>_alerts.json (timestamps + bbox)
    - Volitelně: <out>_annotated.mp4 (vizualizace)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suspicious_detector import SuspiciousDetector


# ===== Default Config =====
# Tuned 2026-05-10 přes grid search na CAVIAR (tune_caviar_params.py):
# abandon=2, radius=1.5, min_own=1 → 100% precision, 100% recall, 6.4s mean latency
DEFAULT_SAMPLE_FPS = 5
DEFAULT_ABANDON_SEC = 2                # was 5 — sníženo pro vyšší recall (LeftBag_PickedUp)
DEFAULT_OWNERSHIP_RADIUS_FACTOR = 1.5   # × person bbox height
DEFAULT_MAX_PERSON_TRACK_DIST = 100
DEFAULT_MAX_OBJ_MISSED = 5
DEFAULT_OBJ_IOU_THRESHOLD = 0.3
DEFAULT_MIN_OWNERSHIP_FRAMES = 1        # was 2 — sníženo, ownership claim snadnější
DEFAULT_MIN_OBJ_TRACK_LEN = 3

# Armed person detection — osoba držící high-threat objekt = podezřelá (ARMED)
# SKELETON-BASED: osoba je ARMED jen když má ZÁPĚSTÍ blízko bbox zbraně (ne jen v radiu těla).
# Řeší nejednoznačnost v davu — "drží" vs "stojí vedle".
HIGH_THREAT_LABELS = {'baseball bat', 'knife', 'gun'}  # scissors odebrán — generoval FP šum v davu
CROWDPOSE_WRIST_IDX = (4, 5)        # l_wrist, r_wrist v CrowdPose 14-keypoint formátu
WRIST_CONF_MIN = 0.3               # min confidence keypointu zápěstí, jinak se ignoruje
ARMED_WRIST_DIST_FACTOR = 0.45     # zbraň "v ruce" pokud zápěstí < faktor × výška od bbox zbraně
ARMED_FALLBACK_RADIUS_FACTOR = 0.7 # když chybí skelet → centroid-based (přísnější než dřív)
ARMED_MIN_FRAMES = 1               # flag hned při 1. detekci
ARMED_HOLD_FRAMES = 8              # HYSTEREZE: drž ARMED N sampled framů po výpadku detekce
                                   # (skeleton-based je přesnější → kratší hold než radius verze)

# ESPCN super-resolution (stejné jako extract_skeletons_two_stage.ipynb)
ESPCN_MODEL_PATH = 'models/super_resolution/ESPCN_x4.pb'
ESPCN_SCALE = 4

# Výchozí složka pro anotovaná demo videa + alerts
DEFAULT_OUTPUT_DIR = 'demo_outputs'

_SR_NET = None


def init_super_resolution():
    """Inicializuje ESPCN_x4 model. Identické s extract_skeletons_two_stage.ipynb."""
    global _SR_NET
    if _SR_NET is not None:
        return _SR_NET
    try:
        if not hasattr(cv2, 'dnn_superres'):
            print('⚠ cv2.dnn_superres nedostupné. Použije INTER_CUBIC fallback.')
            return None
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(ESPCN_MODEL_PATH)
        sr.setModel('espcn', ESPCN_SCALE)
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print('✓ ESPCN_x4 super-resolution loaded')
        _SR_NET = sr
        return sr
    except Exception as e:
        print(f'⚠ ESPCN init failed: {e}')
        return None


def apply_upscale(img, scale=ESPCN_SCALE):
    """Upscale 4× pomocí ESPCN, fallback INTER_CUBIC."""
    sr = init_super_resolution()
    h, w = img.shape[:2]
    if sr is None:
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    try:
        result = sr.upsample(img)
        target_h, target_w = h * scale, w * scale
        if result.shape[0] != target_h or result.shape[1] != target_w:
            result = cv2.resize(result, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return result
    except Exception:
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


# ===== Helpers =====
def bbox_iou(box1, box2):
    """IoU mezi dvěma xyxy boxy."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def bbox_centroid(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def bbox_height(box):
    return box[3] - box[1]


def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def point_to_bbox_dist(pt, box):
    """Vzdálenost bodu od obdélníku (0 pokud je bod uvnitř). box = [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = box
    dx = max(x1 - pt[0], 0, pt[0] - x2)
    dy = max(y1 - pt[1], 0, pt[1] - y2)
    return float((dx * dx + dy * dy) ** 0.5)


# ===== Trackers =====
class PersonTracker:
    """Greedy centroid tracker pro osoby."""

    def __init__(self, max_dist=DEFAULT_MAX_PERSON_TRACK_DIST, max_missed=3):
        self.tracks = {}      # tid -> {'centroid', 'box', 'missed', 'history': [(frame, box)]}
        self.next_id = 0
        self.max_dist = max_dist
        self.max_missed = max_missed

    def update(self, frame_idx, person_boxes, skeletons=None):
        """
        person_boxes: (N, 5) [x1,y1,x2,y2,conf]. skeletons: (N, K, 3) odpovídající po indexu
        (skeletons[i] patří person_boxes[i]). Returns active tracks dict.
        """
        def _skel(i):
            if skeletons is not None and len(skeletons) > i:
                return skeletons[i]
            return None

        # Increment missed for all
        for t in self.tracks.values():
            t['missed'] += 1

        if len(person_boxes) > 0:
            new_centroids = [bbox_centroid(b[:4]) for b in person_boxes]
            matched = set()

            # Greedy match: for each track, find nearest unmatched detection
            for tid, t in list(self.tracks.items()):
                if t['missed'] > 1:  # only match active (missed=1 means just lost this frame)
                    continue
                best_d, best_i = float('inf'), -1
                for i, c in enumerate(new_centroids):
                    if i in matched:
                        continue
                    d = distance(t['centroid'], c)
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i >= 0 and best_d < self.max_dist:
                    t['centroid'] = new_centroids[best_i]
                    t['box'] = person_boxes[best_i, :4]
                    t['conf'] = float(person_boxes[best_i, 4]) if person_boxes.shape[1] > 4 else 1.0
                    t['skeleton'] = _skel(best_i)
                    t['missed'] = 0
                    t['history'].append((frame_idx, person_boxes[best_i, :4]))
                    matched.add(best_i)

            # Start new tracks for unmatched
            for i, c in enumerate(new_centroids):
                if i not in matched:
                    self.tracks[self.next_id] = {
                        'centroid': c,
                        'box': person_boxes[i, :4],
                        'conf': float(person_boxes[i, 4]) if person_boxes.shape[1] > 4 else 1.0,
                        'skeleton': _skel(i),
                        'missed': 0,
                        'history': [(frame_idx, person_boxes[i, :4])],
                        'first_frame': frame_idx,
                    }
                    self.next_id += 1

        # Remove dead tracks
        dead = [tid for tid, t in self.tracks.items() if t['missed'] > self.max_missed]
        for tid in dead:
            del self.tracks[tid]

        return self.active()

    def active(self):
        return {tid: t for tid, t in self.tracks.items() if t['missed'] == 0}


class ObjectTracker:
    """Greedy IoU tracker pro objekty (s ownership history)."""

    def __init__(self, iou_threshold=DEFAULT_OBJ_IOU_THRESHOLD, max_missed=DEFAULT_MAX_OBJ_MISSED):
        self.tracks = {}     # oid -> {'box', 'label', 'missed', 'history', 'owner_history', 'first_frame'}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def update(self, frame_idx, object_dets):
        """object_dets: list of (bbox, score, label). Returns active tracks dict."""
        for t in self.tracks.values():
            t['missed'] += 1

        if object_dets:
            matched = set()
            for oid, t in list(self.tracks.items()):
                if t['missed'] > 1:
                    continue
                best_iou, best_i = 0.0, -1
                for i, (box, score, label) in enumerate(object_dets):
                    if i in matched:
                        continue
                    if label != t['label']:
                        continue
                    iou = bbox_iou(t['box'], box)
                    if iou > best_iou:
                        best_iou, best_i = iou, i
                if best_i >= 0 and best_iou > self.iou_threshold:
                    box, score, _ = object_dets[best_i]
                    t['box'] = box
                    t['score'] = score
                    t['missed'] = 0
                    t['history'].append((frame_idx, box, score))
                    matched.add(best_i)

            for i, (box, score, label) in enumerate(object_dets):
                if i not in matched:
                    self.tracks[self.next_id] = {
                        'box': np.array(box),
                        'label': label,
                        'score': score,
                        'missed': 0,
                        'history': [(frame_idx, box, score)],
                        'owner_history': [],   # list of (frame, owner_person_id or None)
                        'first_frame': frame_idx,
                        'last_owner_frame': None,  # frame when last had owner
                        'last_owner_id': None,
                    }
                    self.next_id += 1

        dead = [oid for oid, t in self.tracks.items() if t['missed'] > self.max_missed]
        for oid in dead:
            del self.tracks[oid]

        return self.active()

    def active(self):
        return {oid: t for oid, t in self.tracks.items() if t['missed'] == 0}


# ===== Person-Object Association =====
def update_associations(frame_idx, person_tracks, object_tracks, radius_factor=DEFAULT_OWNERSHIP_RADIUS_FACTOR):
    """
    Pro každý aktivní objekt: najít nejbližší osobu.
    Pokud je osoba v radiu (factor × její výška) → owner.
    Aktualizuje object['owner_history'] + 'last_owner_frame'.
    """
    for oid, obj in object_tracks.items():
        obj_centroid = bbox_centroid(obj['box'])
        best_d, best_pid = float('inf'), None
        for pid, p in person_tracks.items():
            person_centroid = bbox_centroid(p['box'])
            d = distance(obj_centroid, person_centroid)
            radius = bbox_height(p['box']) * radius_factor
            if d < radius and d < best_d:
                best_d, best_pid = d, pid

        obj['owner_history'].append((frame_idx, best_pid))
        if best_pid is not None:
            obj['last_owner_frame'] = frame_idx
            obj['last_owner_id'] = best_pid


# ===== Abandonment Rule =====
def check_abandonment(frame_idx, fps, object_tracks, abandon_sec, min_ownership_frames):
    """
    Pro každý objekt s předchozím vlastníkem: pokud je teď bez vlastníka > abandon_sec → alert.
    Vrátí list nových alertů (bez duplicit — alert se vyšle jen jednou).
    """
    alerts = []
    for oid, obj in object_tracks.items():
        if obj.get('alerted', False):
            continue
        # Měl objekt někdy ownera?
        owners = [pid for _, pid in obj['owner_history'] if pid is not None]
        if len(owners) < min_ownership_frames:
            continue
        # Současný stav: nemá ownera?
        if obj['owner_history'] and obj['owner_history'][-1][1] is not None:
            continue
        # Jak dlouho nemá ownera?
        last_owner_frame = obj['last_owner_frame']
        if last_owner_frame is None:
            continue
        sec_alone = (frame_idx - last_owner_frame) / fps
        if sec_alone >= abandon_sec:
            alerts.append({
                'frame': frame_idx,
                'object_id': oid,
                'label': obj['label'],
                'box': obj['box'].tolist() if isinstance(obj['box'], np.ndarray) else list(obj['box']),
                'sec_alone': sec_alone,
                'previous_owner': obj['last_owner_id'],
            })
            obj['alerted'] = True
    return alerts


# ===== Armed person detection =====
def _wrist_points(skeleton):
    """Vrátí list (x,y) zápěstí s dostatečnou confidence z CrowdPose skeletonu."""
    pts = []
    if skeleton is None or len(skeleton) == 0:
        return pts
    for wi in CROWDPOSE_WRIST_IDX:
        if wi < len(skeleton) and skeleton[wi, 2] > WRIST_CONF_MIN:
            pts.append(skeleton[wi, :2])
    return pts


def update_armed_status(person_tracks, object_tracks,
                        wrist_dist_factor=ARMED_WRIST_DIST_FACTOR,
                        fallback_radius_factor=ARMED_FALLBACK_RADIUS_FACTOR,
                        min_frames=ARMED_MIN_FRAMES, hold_frames=ARMED_HOLD_FRAMES):
    """
    Flag osoby DRŽÍCÍ high-threat objekt (pálka/nůž/zbraň) jako ARMED.

    SKELETON-BASED: osoba je ARMED jen když má ZÁPĚSTÍ blízko bbox zbraně (drží ji),
    ne jen když je zbraň v radiu těla. Řeší nejednoznačnost v davu (stojí vedle ≠ drží).
    Fallback na centroid-radius pokud chybí keypointy zápěstí.

    HYSTEREZE: jakmile je osoba jednou ARMED, drží se flag `hold_frames` sampled framů
    i bez další detekce (YOLO detekuje zbraně nekonzistentně, osoba ji nezahodí).
    """
    for pid, p in person_tracks.items():
        p_h = bbox_height(p['box'])
        wrists = _wrist_points(p.get('skeleton'))
        weapon, best_d = None, float('inf')

        for oid, obj in object_tracks.items():
            if obj.get('label') not in HIGH_THREAT_LABELS:
                continue
            if wrists:
                # vzdálenost nejbližšího zápěstí od bbox zbraně
                d = min(point_to_bbox_dist(w, obj['box']) for w in wrists)
                thresh = p_h * wrist_dist_factor
            else:
                # fallback: centroid těla → centroid zbraně
                d = distance(bbox_centroid(p['box']), bbox_centroid(obj['box']))
                thresh = p_h * fallback_radius_factor
            if d < thresh and d < best_d:
                best_d, weapon = d, obj['label']

        if weapon:
            p['armed_count'] = p.get('armed_count', 0) + 1
            if p['armed_count'] >= min_frames:
                p['armed'] = True
                p['weapon_label'] = weapon
                p['armed_hold'] = hold_frames
        else:
            hold = p.get('armed_hold', 0)
            if hold > 0:
                p['armed_hold'] = hold - 1
            else:
                p['armed'] = False
                p['weapon_label'] = None
                p['armed_count'] = 0
    return person_tracks


# ===== Visualization =====
def _object_color_status(obj, sampled_idx, out_fps, oid):
    """Vrátí (color_BGR, status_text) pro objekt podle jeho stavu v daném snapshotu."""
    if obj.get('alerted', False):
        return (0, 0, 255), 'ABANDONED'
    if obj['last_owner_frame'] is not None:
        sec_alone = (sampled_idx - obj['last_owner_frame']) / out_fps
        if obj['owner_history'] and obj['owner_history'][-1][1] is not None:
            return (0, 200, 200), f'O{oid}: {obj["label"]} (P{obj["last_owner_id"]})'
        return (0, 165, 255), f'O{oid}: {obj["label"]} alone {sec_alone:.1f}s'
    return (255, 255, 0), f'O{oid}: {obj["label"]}'


def build_snapshot(video_frame, sampled_idx, out_fps, person_tracks, object_tracks, scale):
    """
    Zachytí stav trackerů jako snapshot pro pozdější interpolaci.
    Boxy se ukládají v ORIGINÁLNÍCH souřadnicích videa (vydělené scale faktorem,
    protože detekce běžela na upscaled framu).

    Returns dict: {video_frame, persons:{tid:{box,conf}}, objects:{oid:{box,color,status}}}
    """
    persons = {}
    for tid, p in person_tracks.items():
        persons[tid] = {
            'box': np.asarray(p['box'], dtype=np.float32) * scale,
            'conf': float(p.get('conf', 1.0)),
            'armed': bool(p.get('armed', False)),
            'weapon_label': p.get('weapon_label'),
        }
    objects = {}
    for oid, obj in object_tracks.items():
        color, status = _object_color_status(obj, sampled_idx, out_fps, oid)
        objects[oid] = {
            'box': np.asarray(obj['box'], dtype=np.float32) * scale,
            'color': color,
            'status': status,
        }
    return {'video_frame': video_frame, 'persons': persons, 'objects': objects}


def _lerp_box(b0, b1, a):
    return b0 * (1.0 - a) + b1 * a


def interpolate_state(snapshots, video_frame):
    """
    Pro libovolný frame videa vrátí interpolovaný stav (persons, objects)
    mezi dvěma nejbližšími snapshoty. Track přítomný v obou → lineární interpolace
    pozice; track jen v jednom → drží se beze změny (krátké objevení/zmizení).
    """
    if not snapshots:
        return {}, {}
    if video_frame <= snapshots[0]['video_frame']:
        s = snapshots[0]
        return s['persons'], s['objects']
    if video_frame >= snapshots[-1]['video_frame']:
        s = snapshots[-1]
        return s['persons'], s['objects']

    # najdi bracketing snapshots
    before, after = snapshots[0], snapshots[-1]
    for i in range(len(snapshots) - 1):
        if snapshots[i]['video_frame'] <= video_frame <= snapshots[i + 1]['video_frame']:
            before, after = snapshots[i], snapshots[i + 1]
            break

    span = after['video_frame'] - before['video_frame']
    alpha = (video_frame - before['video_frame']) / span if span > 0 else 0.0

    # Persons
    persons = {}
    for tid, pb in before['persons'].items():
        if tid in after['persons']:
            pa = after['persons'][tid]
            persons[tid] = {
                'box': _lerp_box(pb['box'], pa['box'], alpha),
                'conf': pb['conf'] * (1 - alpha) + pa['conf'] * alpha,
                'armed': pb.get('armed', False),          # diskrétní — z 'before'
                'weapon_label': pb.get('weapon_label'),
            }
        elif alpha < 0.5:
            persons[tid] = pb  # mizí — drž v první půlce intervalu
    for tid, pa in after['persons'].items():
        if tid not in before['persons'] and alpha >= 0.5:
            persons[tid] = pa  # objevuje se — ukaž v druhé půlce

    # Objects (status/barva diskrétní — z "before"; pozice interpolovaná)
    objects = {}
    for oid, ob in before['objects'].items():
        if oid in after['objects']:
            oa = after['objects'][oid]
            objects[oid] = {
                'box': _lerp_box(ob['box'], oa['box'], alpha),
                'color': ob['color'],
                'status': ob['status'],
            }
        elif alpha < 0.5:
            objects[oid] = ob
    for oid, oa in after['objects'].items():
        if oid not in before['objects'] and alpha >= 0.5:
            objects[oid] = oa

    return persons, objects


def draw_state(img, persons, objects, alert_active, video_frame, video_fps):
    """Vykreslí interpolovaný stav na originální frame (souřadnice už v orig prostoru)."""
    out = img.copy()

    for pid, p in persons.items():
        x1, y1, x2, y2 = [int(v) for v in p['box']]
        if p.get('armed'):
            box_color = (0, 0, 255)        # červená = ozbrojená/podezřelá osoba
            label = f'P{pid} ARMED: {p.get("weapon_label", "?")}'
            thickness = 3
        else:
            box_color = (0, 255, 0)        # zelená = normální osoba
            label = f'P{pid}: {p["conf"]:.2f}'
            thickness = 2
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), box_color, -1)
        txt_color = (255, 255, 255) if p.get('armed') else (0, 0, 0)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)

    for oid, obj in objects.items():
        x1, y1, x2, y2 = [int(v) for v in obj['box']]
        cv2.rectangle(out, (x1, y1), (x2, y2), obj['color'], 3)
        cv2.putText(out, obj['status'], (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj['color'], 1)

    cv2.putText(out, f'Frame {video_frame} ({video_frame/video_fps:.1f}s)', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if alert_active:
        cv2.putText(out, f'! ALERT: {alert_active} abandoned', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


# ===== Main pipeline =====
def process_video(video_path, detector, args):
    """
    Dvouprůchodový processing pro plynulé video:
      PASS 1: detekce na sample-FPS → stavové snapshoty (orig souřadnice)
      PASS 2: re-read videa, každý frame renderován s interpolovanými boxy → plné FPS

    Vrací list alertů. Anotované video (pokud --save-video) jde do args.output_dir.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'⚠ Could not open {video_path}')
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_step = max(1, int(round(video_fps / args.sample_fps)))
    out_fps = video_fps / sample_step

    person_tracker = PersonTracker(max_dist=args.max_person_dist)
    object_tracker = ObjectTracker(iou_threshold=args.obj_iou, max_missed=args.max_obj_missed)

    print(f'\n=== Processing {video_path} ===')
    print(f'  Video FPS: {video_fps:.1f}, sample at: {args.sample_fps} (step {sample_step})')
    print(f'  Abandonment threshold: {args.abandon_sec}s, ownership radius: {args.radius_factor}× person height')
    if args.upscale:
        init_super_resolution()
        print(f'  Upscaling: ESPCN_x{ESPCN_SCALE} aktivní (low-res input → HD pro detection)')

    # ---------- PASS 1: detekce + tracking → snapshots ----------
    alerts = []
    snapshots = []
    scale = 1.0  # orig/detect — zjistí se z prvního framu
    frame_idx = 0
    sampled_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_step == 0:
            frame_for_detect = apply_upscale(frame) if args.upscale else frame
            if scale == 1.0 and frame_for_detect.shape[1] != frame.shape[1]:
                scale = frame.shape[1] / frame_for_detect.shape[1]

            det = detector.detect(frame_for_detect, conf_obj=args.conf_obj, conf_weapon=args.conf_weapon,
                                   imgsz=args.imgsz, use_better_obj=args.use_better_obj)

            obj_dets = []
            for tier in ['high', 'medium', 'carry']:
                for box, score, label in det[tier]:
                    obj_dets.append((np.array(box), score, label))

            person_tracks = person_tracker.update(sampled_idx, det['person_boxes'], det.get('skeletons'))
            object_tracks = object_tracker.update(sampled_idx, obj_dets)
            update_associations(sampled_idx, person_tracks, object_tracks, radius_factor=args.radius_factor)
            if args.armed:
                # ARMED person-flagging — experimentální, křehké na COCO detekci pálek (viz weapon_detectors.md).
                # Default VYPNUTO; čeká na fine-tuned detektor tyčí/pálek.
                update_armed_status(person_tracks, object_tracks)

            new_alerts = check_abandonment(sampled_idx, out_fps, object_tracks,
                                            abandon_sec=args.abandon_sec,
                                            min_ownership_frames=args.min_ownership_frames)
            for a in new_alerts:
                a['video_frame'] = frame_idx
                a['video_time_sec'] = frame_idx / video_fps
                alerts.append(a)
                print(f'  ⚠ FRAME {frame_idx} (t={frame_idx/video_fps:.1f}s) — '
                      f'ABANDONED {a["label"]} (object {a["object_id"]}, '
                      f'previous owner P{a["previous_owner"]}, alone {a["sec_alone"]:.1f}s)')

            if args.save_video:
                snapshots.append(build_snapshot(frame_idx, sampled_idx, out_fps,
                                                 person_tracks, object_tracks, scale))
            sampled_idx += 1
        frame_idx += 1

    total_frames = frame_idx
    cap.release()
    pass1_time = time.time() - t0
    print(f'  PASS 1 (detekce): {pass1_time:.1f}s — {sampled_idx} sampled framů, {len(alerts)} alertů')

    # ---------- PASS 2: render every frame s interpolací → plynulé video ----------
    if args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)
        out_video_path = os.path.join(args.output_dir, Path(video_path).stem + '_annotated.mp4')

        cap = cv2.VideoCapture(video_path)
        ret, sample = cap.read()
        h, w = sample.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video_path, fourcc, video_fps, (w, h))  # PLNÉ video FPS

        # alert overlay: počet aktivních alertů od jejich video_frame
        alert_frames = sorted(a['video_frame'] for a in alerts)

        t1 = time.time()
        vf = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            persons, objects = interpolate_state(snapshots, vf)
            alert_active = sum(1 for af in alert_frames if af <= vf)
            annotated = draw_state(frame, persons, objects, alert_active, vf, video_fps)
            writer.write(annotated)
            vf += 1

        cap.release()
        writer.release()
        print(f'  PASS 2 (render): {time.time()-t1:.1f}s — {vf} framů @ {video_fps:.0f}fps (plynulé)')
        print(f'  ✓ Video: {out_video_path}')

    print(f'  ✓ Hotovo za {time.time()-t0:.1f}s celkem | {total_frames} framů | {len(alerts)} alertů')
    return alerts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True, help='Path to video file')
    p.add_argument('--sample-fps', type=int, default=DEFAULT_SAMPLE_FPS,
                   help='Frames per second to process (default 5)')
    p.add_argument('--abandon-sec', type=float, default=DEFAULT_ABANDON_SEC,
                   help='Abandonment threshold in seconds (default 5)')
    p.add_argument('--radius-factor', type=float, default=DEFAULT_OWNERSHIP_RADIUS_FACTOR,
                   help='Ownership radius = factor × person bbox height (default 1.5)')
    p.add_argument('--max-person-dist', type=float, default=DEFAULT_MAX_PERSON_TRACK_DIST)
    p.add_argument('--max-obj-missed', type=int, default=DEFAULT_MAX_OBJ_MISSED)
    p.add_argument('--obj-iou', type=float, default=DEFAULT_OBJ_IOU_THRESHOLD)
    p.add_argument('--min-ownership-frames', type=int, default=DEFAULT_MIN_OWNERSHIP_FRAMES)
    p.add_argument('--conf-obj', type=float, default=0.15)
    p.add_argument('--conf-weapon', type=float, default=0.55)
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--use-better-obj', action='store_true',
                   help='Multi-pass YOLO11x (5× pomalejší, lepší recall)')
    p.add_argument('--upscale', action='store_true',
                   help='ESPCN_x4 super-resolution před detekcí (pro low-res CCTV jako CAVIAR)')
    p.add_argument('--save-video', action='store_true', help='Save annotated video')
    p.add_argument('--save-alerts', default=None,
                   help='Cesta k JSON s alerty (default: <output-dir>/<video>_alerts.json)')
    p.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                   help=f'Složka pro výstupy (video + alerts). Default: {DEFAULT_OUTPUT_DIR}')
    p.add_argument('--skip-weapon', action='store_true', help='Vynechat weapon model (rychleji)')
    p.add_argument('--armed', action='store_true',
                   help='Zapnout ARMED person-flagging (skeleton-based). Default VYPNUTO — '
                        'křehké na COCO detekci pálek, čeká na fine-tuned detektor (viz weapon_detectors.md).')
    args = p.parse_args()

    print('Loading detector...')
    detector = SuspiciousDetector(skip_weapon_model=args.skip_weapon)

    alerts = process_video(args.video, detector, args)

    # Alerts JSON — default do output složky, pojmenované podle videa
    os.makedirs(args.output_dir, exist_ok=True)
    alerts_path = args.save_alerts or os.path.join(
        args.output_dir, Path(args.video).stem + '_alerts.json')
    with open(alerts_path, 'w') as f:
        json.dump(alerts if alerts else [], f, indent=2, default=str)
    print(f'  ✓ Alerts ({len(alerts) if alerts else 0}) → {alerts_path}')


if __name__ == '__main__':
    main()
