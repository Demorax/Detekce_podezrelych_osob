"""
CAVIAR rule-based abandoned-object detector eval.

Místo YOLO detekce (která nefunguje na CAVIAR low-res 384×288 fisheye) používá
GT bboxy přímo z CAVIAR XML. Tím izolujeme test logiky (tracking + association +
abandonment rule) od kvality detektoru.

Postup:
    1. Parse XML → per-frame (person_boxes, object_boxes)
    2. Krmení do PersonTracker + ObjectTracker + update_associations
    3. check_abandonment → alerts
    4. Compare alerts proti GT drop event

Použití:
    python eval_caviar_rules.py --xml data/CAVIAR/LeftBag/lb1gt.xml
    python eval_caviar_rules.py --all   # eval na všech LeftBag* + Fight* + baselines
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abandoned_object_detector import (
    PersonTracker, ObjectTracker,
    update_associations, check_abandonment,
)


# ===== CAVIAR roles → kategorie =====
PERSON_ROLES = {
    'walker', 'walkers', 'browser', 'browsers', 'meeters',
    'fighter', 'fighters', 'leaving victim',
}
OBJECT_ROLES = {'leaving object'}


# ===== XML Parser =====
def cvml_box_to_xyxy(box_elem):
    """CAVIAR box: xc, yc, w, h (center+size). Vrátí (x1,y1,x2,y2)."""
    xc = float(box_elem.get('xc'))
    yc = float(box_elem.get('yc'))
    w = float(box_elem.get('w'))
    h = float(box_elem.get('h'))
    return np.array([xc - w/2, yc - h/2, xc + w/2, yc + h/2], dtype=np.float32)


def parse_caviar_xml(xml_path):
    """
    Returns:
        list of dict per frame:
            {'frame': int,
             'persons': [(id, box_xyxy, role), ...],
             'objects': [(id, box_xyxy, role), ...]}
        + GT abandonment events: [(start_frame, end_frame, object_id), ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frames_data = []
    object_role_history = {}  # gt_id -> list of (frame, role)

    for fr in root.findall('frame'):
        fnum = int(fr.get('number'))
        persons, objects = [], []
        for obj in fr.findall('.//object'):
            gt_id = obj.get('id')
            box_elem = obj.find('box')
            if box_elem is None:
                continue
            box = cvml_box_to_xyxy(box_elem)
            # role
            h = obj.find('.//hypothesis')
            role = h.findtext('role', 'unknown') if h is not None else 'unknown'

            object_role_history.setdefault(gt_id, []).append((fnum, role))

            if role in PERSON_ROLES:
                persons.append((gt_id, box, role))
            elif role in OBJECT_ROLES:
                objects.append((gt_id, box, role))
            # else: ignore (unknown/empty)
        frames_data.append({'frame': fnum, 'persons': persons, 'objects': objects})

    # Build GT abandonment events: contiguous frames with role=='leaving object'
    gt_events = []
    for gt_id, history in object_role_history.items():
        leaving_frames = [f for f, r in history if r == 'leaving object']
        if leaving_frames:
            gt_events.append({
                'gt_id': gt_id,
                'start_frame': min(leaving_frames),
                'end_frame': max(leaving_frames),
                'duration_frames': len(leaving_frames),
            })

    return frames_data, gt_events


# ===== Format conversion: GT (id, box, role) → tracker input format =====
def to_person_boxes(persons):
    """Tracker očekává numpy (N, 5) [x1,y1,x2,y2,conf]. Conf = 1.0 pro GT."""
    if not persons:
        return np.array([])
    return np.array([[*box, 1.0] for _, box, _ in persons], dtype=np.float32)


def to_object_dets(objects, label='leaving_object'):
    """Tracker očekává list of (bbox, score, label). Score = 1.0 pro GT."""
    return [(box, 1.0, label) for _, box, _ in objects]


# ===== Eval utilities =====
def evaluate_video(xml_path, abandon_sec=5, video_fps=25.0,
                    radius_factor=2.0, min_ownership_frames=2,
                    max_obj_missed=15, verbose=False):
    """
    Spustí rule pipeline na GT z CAVIAR XML, vrátí dict s metriky.

    Args:
        abandon_sec: kdy spustit alert (sec po ztrátě ownera)
        video_fps: original video FPS (CAVIAR = 25)
        radius_factor: ownership radius = factor × person_height
        min_ownership_frames: min frames blízko osoby pro ownership
        max_obj_missed: kolik framů bez detekce před concealing track
    """
    frames_data, gt_events = parse_caviar_xml(xml_path)
    name = Path(xml_path).parent.name

    person_tracker = PersonTracker(max_dist=80, max_missed=10)
    object_tracker = ObjectTracker(iou_threshold=0.2, max_missed=max_obj_missed)

    alerts = []
    for fd in frames_data:
        fnum = fd['frame']
        person_boxes = to_person_boxes(fd['persons'])
        object_dets = to_object_dets(fd['objects'])

        person_tracks = person_tracker.update(fnum, person_boxes)
        object_tracks = object_tracker.update(fnum, object_dets)

        update_associations(fnum, person_tracks, object_tracks, radius_factor=radius_factor)

        new_alerts = check_abandonment(fnum, video_fps, object_tracks,
                                        abandon_sec=abandon_sec,
                                        min_ownership_frames=min_ownership_frames)
        for a in new_alerts:
            alerts.append(a)
            if verbose:
                print(f'  ALERT @ frame {fnum} ({fnum/video_fps:.1f}s): '
                      f'object {a["object_id"]} alone {a["sec_alone"]:.1f}s, owner P{a["previous_owner"]}')

    # Eval against GT
    result = {
        'video': name,
        'total_frames': len(frames_data),
        'gt_events': gt_events,
        'alerts': alerts,
    }

    # Match each alert to a GT event (by temporal overlap with object presence)
    if gt_events:
        gt = gt_events[0]  # CAVIAR LeftBag* má typicky 1 abandonment event
        result['gt_drop_frame'] = gt['start_frame']
        result['gt_drop_sec'] = gt['start_frame'] / video_fps
        result['gt_event_duration_sec'] = gt['duration_frames'] / video_fps

        if alerts:
            first_alert = alerts[0]
            alert_frame = first_alert['frame']
            result['first_alert_frame'] = alert_frame
            result['first_alert_sec'] = alert_frame / video_fps
            # Latency: kolik sec po GT drop_frame přišel alert (positive = po, negative = před = false alarm)
            result['latency_sec'] = (alert_frame - gt['start_frame']) / video_fps
            # Alert is TP if it falls within or shortly after GT abandonment window
            # We tolerate alert up to abandon_sec + 5 after start of GT event
            tp = (alert_frame >= gt['start_frame']
                  and alert_frame <= gt['end_frame'] + abandon_sec * video_fps)
            result['outcome'] = 'TP' if tp else 'FP'
            result['n_alerts'] = len(alerts)
            result['n_extra_alerts'] = len(alerts) - 1  # extra after first
        else:
            result['first_alert_frame'] = None
            result['outcome'] = 'FN'
            result['n_alerts'] = 0
    else:
        # No GT abandonment event (e.g., baseline non-LeftBag video)
        result['gt_drop_frame'] = None
        if alerts:
            result['outcome'] = 'FP_BASELINE'  # nějaký alert kde žádný neměl být
            result['first_alert_frame'] = alerts[0]['frame']
            result['n_alerts'] = len(alerts)
        else:
            result['outcome'] = 'TN'
            result['n_alerts'] = 0

    return result


def print_summary(results):
    """Print compact table + per-class stats."""
    print(f"\n{'=' * 90}")
    print(f"{'video':<22} {'GT':<7} {'alert':<7} {'lat(s)':<8} {'n_al':<5} {'outcome':<10}")
    print('-' * 90)

    counts = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'FP_BASELINE': 0}
    latencies = []

    for r in results:
        gt = f"{r['gt_drop_frame']}" if r.get('gt_drop_frame') is not None else '-'
        al = f"{r['first_alert_frame']}" if r.get('first_alert_frame') is not None else '-'
        lat = f"{r['latency_sec']:+.1f}" if r.get('latency_sec') is not None else '-'
        n = r['n_alerts']
        out = r['outcome']
        counts[out] = counts.get(out, 0) + 1
        if 'latency_sec' in r and out == 'TP':
            latencies.append(r['latency_sec'])
        print(f"{r['video']:<22} {gt:<7} {al:<7} {lat:<8} {n:<5} {out:<10}")

    print('-' * 90)
    print(f"\nCounts: TP={counts['TP']} FP={counts['FP']} FN={counts['FN']} "
          f"TN={counts['TN']} FP_baseline={counts['FP_BASELINE']}")
    n_with_gt = counts['TP'] + counts['FN'] + counts['FP']
    if n_with_gt > 0:
        precision = counts['TP'] / max(counts['TP'] + counts['FP'] + counts['FP_BASELINE'], 1)
        recall = counts['TP'] / max(counts['TP'] + counts['FN'], 1)
        print(f"Precision = {precision:.2%}, Recall = {recall:.2%}")
    if latencies:
        print(f"Mean TP latency: {np.mean(latencies):.1f}s (min {min(latencies):.1f}, max {max(latencies):.1f})")


# ===== Main =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', help='Single XML file')
    parser.add_argument('--all', action='store_true', help='Eval all CAVIAR scenarios')
    parser.add_argument('--abandon-sec', type=float, default=5.0)
    parser.add_argument('--video-fps', type=float, default=25.0)
    parser.add_argument('--radius-factor', type=float, default=2.0)
    parser.add_argument('--min-ownership-frames', type=int, default=2)
    parser.add_argument('--max-obj-missed', type=int, default=15,
                        help='Pro CAVIAR vyšší než default — bag může chvíli zmizet')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    xml_paths = []
    if args.xml:
        xml_paths = [args.xml]
    elif args.all:
        # Find all .xml in data/CAVIAR/*
        for d in sorted(Path('data/CAVIAR').iterdir()):
            if d.is_dir():
                xmls = list(d.glob('*.xml'))
                if xmls:
                    xml_paths.append(str(xmls[0]))  # take first XML per scenario
    else:
        # Default: eval just LeftBag* scenarios
        for d in sorted(Path('data/CAVIAR').iterdir()):
            if d.is_dir() and d.name.startswith('Left'):
                xmls = list(d.glob('*.xml'))
                if xmls:
                    xml_paths.append(str(xmls[0]))

    if not xml_paths:
        print('⚠ No XML files found.')
        return

    print(f'Evaluating {len(xml_paths)} scenarios...')
    print(f'Settings: abandon_sec={args.abandon_sec}, radius={args.radius_factor}× person height, '
          f'min_ownership={args.min_ownership_frames} frames')

    results = []
    for xp in xml_paths:
        print(f'\n--- {xp} ---')
        r = evaluate_video(
            xp,
            abandon_sec=args.abandon_sec,
            video_fps=args.video_fps,
            radius_factor=args.radius_factor,
            min_ownership_frames=args.min_ownership_frames,
            max_obj_missed=args.max_obj_missed,
            verbose=args.verbose,
        )
        results.append(r)

    print_summary(results)


if __name__ == '__main__':
    main()
