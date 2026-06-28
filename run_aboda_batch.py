"""ABODA batch — projede video2..video11 přes abandoned_object_detector."""
import subprocess
import sys
import os
from pathlib import Path

VIDEOS = [f'video{i}.avi' for i in range(2, 12)]  # video2 .. video11
ABODA_DIR = 'data/ABODA'
OUT_DIR = 'reference_img/aboda_alerts'
os.makedirs(OUT_DIR, exist_ok=True)

print(f'ABODA batch: {len(VIDEOS)} videí')
for v in VIDEOS:
    in_path = f'{ABODA_DIR}/{v}'
    if not os.path.exists(in_path):
        print(f'⚠ Skip (not found): {in_path}')
        continue
    out_alerts = f'{OUT_DIR}/{Path(v).stem}_alerts.json'
    if os.path.exists(out_alerts):
        print(f'⊘ Skip (already done): {v}')
        continue

    print(f'\n>>> {v}', flush=True)
    cmd = [
        sys.executable, 'abandoned_object_detector.py',
        '--video', in_path,
        '--upscale', '--use-better-obj',
        '--abandon-sec', '5',
        '--save-alerts', out_alerts,
        '--skip-weapon',
        '--conf-obj', '0.10',
    ]
    try:
        subprocess.run(cmd, check=False, timeout=2400)
    except subprocess.TimeoutExpired:
        print(f'  ⚠ Timeout on {v}')

print('\n=== ABODA batch DONE ===')
print(f'Alerts saved in: {OUT_DIR}/')
