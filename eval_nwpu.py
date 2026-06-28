"""
NWPU Campus eval — abandoned-object detector vs. frame-level anomaly GT.

NWPU má binary GT (0=normal, 1=anomaly) — neumí rozlišit abandonment od running/fighting.
Náš detector hledá specificky abandonment. Eval tedy odpovídá na otázku:
"Když náš detector vyhlásí abandonment alert, je to v okně skutečně anomálního chování?"

Computed metrics:
- Frame-level TP/FP/FN/TN, precision, recall, F1
- Per-video alert presence vs GT presence
- Confusion matrix

Použití:
    python eval_nwpu.py --max-videos 10            # rychlá ukázka
    python eval_nwpu.py --max-videos 50 --skip-weapon  # solidní subset
    python eval_nwpu.py                             # full eval (~12h compute)
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suspicious_detector import SuspiciousDetector
from abandoned_object_detector import (
    PersonTracker, ObjectTracker,
    update_associations, check_abandonment,
)


NWPU_TEST_DIR = 'data/NWPU_Campus/NWPUCampusDataset_extracted/NWPUCampusDataset/videos/Test'
NWPU_GT_PATH = 'data/NWPU_Campus/NWPUCampusDataset_extracted/NWPUCampusDataset/groundtruth/NWPU_Campus_gt.npz'
LOG_DIR = 'logs'

DEFAULT_SAMPLE_FPS = 2  # 1080p HD video je velký, 2 FPS pro rychlost
DEFAULT_ABANDON_SEC = 5
DEFAULT_RADIUS_FACTOR = 1.5
DEFAULT_OBJ_IOU = 0.3
DEFAULT_MAX_OBJ_MISSED = 5
DEFAULT_MIN_OWNERSHIP_FRAMES = 2


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(LOG_DIR, f'eval_nwpu_{ts}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_file


def process_video_for_eval(video_path, detector, args):
    """
    Process video, return per-frame alert mask (1 if any abandonment alert active in that frame).
    Alert "active" = od chvíle vyhlášení do konce videa (jednou alert = stále abandoned).

    Returns:
        alert_mask: np.array shape (N_frames,) of {0, 1}
        n_alerts: int
        first_alert_frame_video: int or None
        processing_time_sec: float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, None, 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_step = max(1, int(round(video_fps / args.sample_fps)))
    out_fps = video_fps / sample_step

    person_tracker = PersonTracker(max_dist=100, max_missed=3)
    object_tracker = ObjectTracker(iou_threshold=args.obj_iou, max_missed=args.max_obj_missed)

    alerts = []  # list of {'frame', 'video_frame', 'object_id', 'label', ...}
    alert_mask = np.zeros(n_frames, dtype=np.uint8)

    t0 = time.time()
    frame_idx = 0
    sampled_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_step == 0:
            # Detect (NWPU is 1080p HD - no upscale needed)
            det = detector.detect(
                frame,
                conf_obj=args.conf_obj,
                conf_weapon=args.conf_weapon,
                imgsz=args.imgsz,
                use_better_obj=args.use_better_obj,
            )

            obj_dets = []
            for tier in ['high', 'medium', 'carry']:
                for box, score, label in det[tier]:
                    obj_dets.append((np.array(box), score, label))

            person_tracks = person_tracker.update(sampled_idx, det['person_boxes'])
            object_tracks = object_tracker.update(sampled_idx, obj_dets)

            update_associations(sampled_idx, person_tracks, object_tracks,
                                radius_factor=args.radius_factor)

            new_alerts = check_abandonment(
                sampled_idx, out_fps, object_tracks,
                abandon_sec=args.abandon_sec,
                min_ownership_frames=args.min_ownership_frames,
            )
            for a in new_alerts:
                a['video_frame'] = frame_idx
                a['video_time_sec'] = frame_idx / video_fps
                alerts.append(a)
                # Mark all subsequent frames as alert-active
                alert_mask[frame_idx:] = 1

            sampled_idx += 1
        frame_idx += 1

    cap.release()
    elapsed = time.time() - t0

    first_alert = alerts[0]['video_frame'] if alerts else None
    return alert_mask, len(alerts), first_alert, elapsed


def evaluate_video(gt_mask, pred_mask, video_name):
    """Frame-level metrics (handle length mismatch by truncating to min)."""
    n = min(len(gt_mask), len(pred_mask))
    gt = gt_mask[:n].astype(bool)
    pred = pred_mask[:n].astype(bool)

    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    tn = int((~gt & ~pred).sum())

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)

    return {
        'video': video_name,
        'n_frames': n,
        'gt_anomaly_frames': int(gt.sum()),
        'pred_alert_frames': int(pred.sum()),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': prec, 'recall': rec, 'f1': f1,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-videos', type=int, default=None,
                   help='Max videí (None = vše 242)')
    p.add_argument('--sample-fps', type=int, default=DEFAULT_SAMPLE_FPS)
    p.add_argument('--abandon-sec', type=float, default=DEFAULT_ABANDON_SEC)
    p.add_argument('--radius-factor', type=float, default=DEFAULT_RADIUS_FACTOR)
    p.add_argument('--obj-iou', type=float, default=DEFAULT_OBJ_IOU)
    p.add_argument('--max-obj-missed', type=int, default=DEFAULT_MAX_OBJ_MISSED)
    p.add_argument('--min-ownership-frames', type=int, default=DEFAULT_MIN_OWNERSHIP_FRAMES)
    p.add_argument('--conf-obj', type=float, default=0.15)
    p.add_argument('--conf-weapon', type=float, default=0.55)
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--use-better-obj', action='store_true', help='Multi-pass YOLO11x (5× pomalejší)')
    p.add_argument('--skip-weapon', action='store_true')
    p.add_argument('--results-json', default='nwpu_eval_results.json')
    args = p.parse_args()

    log_file = setup_logging()
    logging.info('=' * 70)
    logging.info('NWPU Campus eval started')
    logging.info(f'  Test dir: {NWPU_TEST_DIR}')
    logging.info(f'  GT: {NWPU_GT_PATH}')
    logging.info(f'  Sample FPS: {args.sample_fps}')
    logging.info(f'  Multi-pass: {args.use_better_obj}')
    logging.info(f'  Log: {log_file}')
    logging.info('=' * 70)

    # Load GT
    gt_data = np.load(NWPU_GT_PATH)
    gt_keys = sorted(gt_data.keys())

    # Process videos
    if args.max_videos:
        gt_keys = gt_keys[:args.max_videos]
    logging.info(f'Processing {len(gt_keys)} videos\n')

    detector = SuspiciousDetector(skip_weapon_model=args.skip_weapon)
    logging.info('')

    all_results = []
    total_t0 = time.time()

    for i, vid_name in enumerate(gt_keys, 1):
        video_path = os.path.join(NWPU_TEST_DIR, f'{vid_name}.avi')
        if not os.path.exists(video_path):
            logging.warning(f'[{i}/{len(gt_keys)}] {vid_name}: video file not found')
            continue

        gt_mask = gt_data[vid_name]
        gt_anomaly = int(gt_mask.sum() > 0)

        try:
            alert_mask, n_alerts, first_alert, proc_time = process_video_for_eval(
                video_path, detector, args
            )
            if alert_mask is None:
                logging.warning(f'[{i}/{len(gt_keys)}] {vid_name}: cannot read video')
                continue

            metrics = evaluate_video(gt_mask, alert_mask, vid_name)
            metrics['n_alerts'] = n_alerts
            metrics['first_alert_frame'] = first_alert
            metrics['has_gt_anomaly'] = gt_anomaly
            metrics['proc_time'] = proc_time
            all_results.append(metrics)

            logging.info(
                f'[{i}/{len(gt_keys)}] {vid_name}: '
                f'GT_anom={"Y" if gt_anomaly else "N"} ({metrics["gt_anomaly_frames"]}f) | '
                f'alerts={n_alerts} ({metrics["pred_alert_frames"]}f) | '
                f'P={metrics["precision"]:.2f} R={metrics["recall"]:.2f} F1={metrics["f1"]:.2f} | '
                f'{proc_time:.1f}s'
            )

        except Exception as e:
            logging.error(f'[{i}/{len(gt_keys)}] {vid_name}: ERROR {type(e).__name__}: {e}')

    # Aggregate
    elapsed_total = time.time() - total_t0
    logging.info('\n' + '=' * 70)
    logging.info('AGGREGATE RESULTS')
    logging.info('=' * 70)

    total_tp = sum(r['tp'] for r in all_results)
    total_fp = sum(r['fp'] for r in all_results)
    total_fn = sum(r['fn'] for r in all_results)
    total_tn = sum(r['tn'] for r in all_results)
    total_frames = total_tp + total_fp + total_fn + total_tn

    micro_prec = total_tp / max(total_tp + total_fp, 1)
    micro_rec = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 1e-6)

    logging.info(f'Total frames: {total_frames}')
    logging.info(f'TP: {total_tp}  FP: {total_fp}  FN: {total_fn}  TN: {total_tn}')
    logging.info(f'Frame-level micro: precision={micro_prec:.3f} recall={micro_rec:.3f} F1={micro_f1:.3f}')

    # Video-level: alerts vs GT presence
    n_gt_anom = sum(1 for r in all_results if r['has_gt_anomaly'])
    n_normal = len(all_results) - n_gt_anom
    n_pred_anom = sum(1 for r in all_results if r['n_alerts'] > 0)
    video_tp = sum(1 for r in all_results if r['has_gt_anomaly'] and r['n_alerts'] > 0)
    video_fp = sum(1 for r in all_results if not r['has_gt_anomaly'] and r['n_alerts'] > 0)
    video_fn = n_gt_anom - video_tp

    logging.info(f'\nVideo-level (alert presence vs GT presence):')
    logging.info(f'  GT anomaly: {n_gt_anom}, GT normal: {n_normal}')
    logging.info(f'  Detector alert: {n_pred_anom}')
    logging.info(f'  TP={video_tp} FP={video_fp} FN={video_fn}')
    if n_gt_anom > 0:
        logging.info(f'  Video recall: {video_tp/n_gt_anom:.2%}')
    if n_pred_anom > 0:
        logging.info(f'  Video precision: {video_tp/n_pred_anom:.2%}')

    logging.info(f'\nTotal eval time: {elapsed_total/60:.1f} min')

    # Save results
    with open(args.results_json, 'w') as f:
        json.dump({
            'aggregate': {
                'total_frames': total_frames,
                'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'tn': total_tn,
                'frame_micro_precision': micro_prec,
                'frame_micro_recall': micro_rec,
                'frame_micro_f1': micro_f1,
                'video_tp': video_tp, 'video_fp': video_fp, 'video_fn': video_fn,
                'n_gt_anomaly': n_gt_anom, 'n_gt_normal': n_normal,
            },
            'per_video': all_results,
        }, f, indent=2)
    logging.info(f'Results saved: {args.results_json}')


if __name__ == '__main__':
    main()
