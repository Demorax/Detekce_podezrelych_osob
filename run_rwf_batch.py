"""
RWF-2000 batch processing — standalone Python skript pro běh v pozadí.

Identický s process_rwf.ipynb Cell 8 (process_split), ale s:
- Logováním do souboru (logs/rwf_batch_YYYY-MM-DD-HH-MM.log)
- CUDA OOM handlingem (clear cache + retry)
- Resume přes skip_existing (default zapnuto)
- Per-video try/except (jeden corrupt soubor nezabije batch)
- Progress logging každých 50 video

Použití:
    python run_rwf_batch.py                   # full batch (train+val), resume on
    python run_rwf_batch.py --split train     # jen train
    python run_rwf_batch.py --no-resume       # re-process všechno
    python run_rwf_batch.py --log-file <path> # custom log file path
"""
import argparse
import logging
import os
import pickle
import sys
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch

# Project root for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from suspicious_detector import (
    RTDETRDetector, VitPoseExtractor,
    _RTDETR_CONFIG, _RTDETR_MODEL,
    _VITPOSE_CONFIG, _VITPOSE_MODEL,
)


# ===== Configuration =====
RWF_ROOT = 'data/RWF-2000/RWF-2000'
OUTPUT_ROOT = 'data/RWF-2000/skeletons'
LOG_DIR = 'logs'

SAMPLE_FPS = 5
RTDETR_CONF = 0.5

MAX_TRACK_DISTANCE = 100
MAX_MISSED_FRAMES = 3
MIN_TRACK_LENGTH = 3


def setup_logging(log_file=None):
    if log_file is None:
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_file = os.path.join(LOG_DIR, f'rwf_batch_{ts}.log')
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


def extract_video_frames(video_path, sample_fps=SAMPLE_FPS):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(video_fps / sample_fps)))
    frames, indices = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(frame)
            indices.append(idx)
        idx += 1
    cap.release()
    return frames, indices


def get_centroid(skeleton_xy):
    if skeleton_xy is None or len(skeleton_xy) == 0:
        return None
    valid = ~np.all(skeleton_xy == 0, axis=-1) & ~np.isnan(skeleton_xy).any(axis=-1)
    if valid.sum() == 0:
        return None
    return skeleton_xy[valid].mean(axis=0)


def _split_tracks(tracks, max_missed):
    active, complete = [], []
    for t in tracks:
        (complete if t['missed'] >= max_missed else active).append(t)
    return active, complete


def track_persons(per_frame_skeletons, per_frame_bboxes,
                  max_dist=MAX_TRACK_DISTANCE, max_missed=MAX_MISSED_FRAMES,
                  min_length=MIN_TRACK_LENGTH):
    tracks, completed = [], []
    for fi, (skeletons, bboxes) in enumerate(zip(per_frame_skeletons, per_frame_bboxes)):
        if skeletons is None or len(skeletons) == 0:
            for t in tracks:
                t['missed'] += 1
            tracks, complete = _split_tracks(tracks, max_missed)
            completed.extend(complete)
            continue
        centroids = [get_centroid(s[:, :2]) for s in skeletons]
        matched = set()
        for t in tracks:
            best_d, best_i = float('inf'), -1
            for i, c in enumerate(centroids):
                if i in matched or c is None:
                    continue
                d = float(np.linalg.norm(t['last'] - c))
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0 and best_d < max_dist:
                t['skeletons'].append(skeletons[best_i])
                t['bboxes'].append(bboxes[best_i])
                t['frames'].append(fi)
                t['last'] = centroids[best_i]
                t['missed'] = 0
                matched.add(best_i)
            else:
                t['missed'] += 1
        tracks, complete = _split_tracks(tracks, max_missed)
        completed.extend(complete)
        for i, c in enumerate(centroids):
            if i not in matched and c is not None:
                tracks.append({
                    'skeletons': [skeletons[i]],
                    'bboxes': [bboxes[i]],
                    'frames': [fi],
                    'last': c,
                    'missed': 0,
                })
    completed.extend(tracks)
    result = []
    for i, t in enumerate(completed):
        if len(t['frames']) < min_length:
            continue
        result.append({
            'person_id': i,
            'frame_indices': t['frames'],
            'skeletons': np.array(t['skeletons']),
            'bboxes': np.array(t['bboxes']),
        })
    return result


def process_video(rtdetr, vitpose, video_path, label):
    """Process single video. Vrací dict nebo None pokud video nelze otevřít."""
    frames, frame_indices = extract_video_frames(video_path)
    if not frames:
        return None

    per_frame_skeletons = []
    per_frame_bboxes = []

    for frame in frames:
        try:
            boxes = rtdetr.detect(frame, conf_threshold=RTDETR_CONF)
            if len(boxes) == 0:
                per_frame_skeletons.append(np.array([]))
                per_frame_bboxes.append(np.array([]))
                continue
            skeletons = vitpose.extract_keypoints(frame, boxes)
            per_frame_skeletons.append(skeletons)
            per_frame_bboxes.append(boxes)
        except torch.cuda.OutOfMemoryError as e:
            logging.warning(f'CUDA OOM on frame {len(per_frame_skeletons)}: {e}. Clearing cache, skipping.')
            torch.cuda.empty_cache()
            per_frame_skeletons.append(np.array([]))
            per_frame_bboxes.append(np.array([]))

    tracks = track_persons(per_frame_skeletons, per_frame_bboxes)

    return {
        'video_path': video_path,
        'label': label,
        'sampled_frame_indices': frame_indices,
        'n_persons_per_frame': [
            len(s) if s is not None and hasattr(s, '__len__') else 0
            for s in per_frame_skeletons
        ],
        'person_tracks': tracks,
    }


def process_split(rtdetr, vitpose, split_name, resume=True):
    n_total = n_ok = n_skip = n_fail = 0
    t_split_start = time.time()

    for class_dir, label in [('Fight', 1), ('NonFight', 0)]:
        in_dir = os.path.join(RWF_ROOT, split_name, class_dir)
        out_dir = os.path.join(OUTPUT_ROOT, split_name, class_dir)
        os.makedirs(out_dir, exist_ok=True)

        videos = sorted([f for f in os.listdir(in_dir) if f.endswith('.avi')])
        logging.info(f'>>> {split_name}/{class_dir}: {len(videos)} videos')

        last_log = time.time()
        for vname in videos:
            n_total += 1
            out_path = os.path.join(out_dir, vname.replace('.avi', '.pkl'))

            if resume and os.path.exists(out_path):
                n_skip += 1
                continue

            in_path = os.path.join(in_dir, vname)
            try:
                t0 = time.time()
                result = process_video(rtdetr, vitpose, in_path, label)
                if result is None:
                    n_fail += 1
                    logging.error(f'Failed (could not open): {vname}')
                    continue
                with open(out_path, 'wb') as f:
                    pickle.dump(result, f)
                n_ok += 1

                # Log every 50 successful processes (or first one)
                if n_ok == 1 or n_ok % 50 == 0:
                    elapsed_split = time.time() - t_split_start
                    logging.info(
                        f'  {split_name}/{class_dir}: ok={n_ok} skip={n_skip} fail={n_fail} | '
                        f'last_video={time.time()-t0:.1f}s | split_elapsed={elapsed_split/60:.1f}min | '
                        f'last={vname}'
                    )
            except torch.cuda.OutOfMemoryError as e:
                n_fail += 1
                logging.error(f'CUDA OOM on {vname}: {e}')
                torch.cuda.empty_cache()
                time.sleep(2)
            except Exception as e:
                n_fail += 1
                logging.error(f'Error on {vname}: {type(e).__name__}: {e}')
                logging.debug(traceback.format_exc())

    logging.info(
        f'=== {split_name} DONE: total={n_total} ok={n_ok} skip={n_skip} fail={n_fail} '
        f'in {(time.time()-t_split_start)/60:.1f} min ==='
    )
    return n_total, n_ok, n_skip, n_fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val', 'all'], default='all')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--log-file', default=None)
    args = parser.parse_args()

    log_file = setup_logging(args.log_file)
    logging.info('=' * 70)
    logging.info('RWF-2000 batch processing started')
    logging.info(f'  Split: {args.split}')
    logging.info(f'  Resume: {not args.no_resume}')
    logging.info(f'  Log file: {log_file}')
    logging.info(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        logging.info(
            f'  GPU: {torch.cuda.get_device_name(0)}, '
            f'mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB'
        )
    logging.info('=' * 70)

    logging.info('Loading RT-DETR + ViTPose...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rtdetr = RTDETRDetector(_RTDETR_CONFIG, _RTDETR_MODEL, device=device)
    vitpose = VitPoseExtractor(_VITPOSE_CONFIG, _VITPOSE_MODEL, device=device)
    logging.info('Models loaded.')

    splits = ['train', 'val'] if args.split == 'all' else [args.split]

    t_start = time.time()
    totals = {'total': 0, 'ok': 0, 'skip': 0, 'fail': 0}
    for sp in splits:
        n_total, n_ok, n_skip, n_fail = process_split(rtdetr, vitpose, sp, resume=not args.no_resume)
        totals['total'] += n_total
        totals['ok'] += n_ok
        totals['skip'] += n_skip
        totals['fail'] += n_fail

    elapsed = time.time() - t_start
    logging.info('=' * 70)
    logging.info(f'BATCH DONE in {elapsed/60:.1f} min ({elapsed/3600:.2f} h)')
    logging.info(f'  Total processed (this run): {totals["total"]}')
    logging.info(f'  Successful: {totals["ok"]}')
    logging.info(f'  Skipped (resumed): {totals["skip"]}')
    logging.info(f'  Failed: {totals["fail"]}')
    logging.info('=' * 70)


if __name__ == '__main__':
    main()
