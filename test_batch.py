"""Testovací batch — anotuje reprezentativní vzorek videí napříč zdroji do demo_outputs/test/."""
import subprocess
import sys
import os

OUT = 'demo_outputs/test'
os.makedirs(OUT, exist_ok=True)

# (popis, video, extra flags)
VIDEOS = [
    # CAVIAR abandonment (low-res + upscale) — testuje abandonment alert (GT)
    ('CAVIAR LeftBag (abandonment)', 'data/CAVIAR/LeftBag/LeftBag.mpg',
     ['--upscale', '--conf-obj', '0.10']),
    ('CAVIAR LeftBox (abandonment)', 'data/CAVIAR/LeftBox/LeftBox.mpg',
     ['--upscale', '--conf-obj', '0.10']),
    # ABODA real CCTV abandonment
    ('ABODA video1 (real CCTV abandonment)', 'data/ABODA/video1.avi',
     ['--upscale', '--conf-obj', '0.10']),
    # MED netrénovaný dav + objekty
    ('MED 015 (dav netrénovaný)', 'data/Motion_Emotion/015.mp4',
     ['--upscale', '--conf-obj', '0.12']),
    ('MED 020 (dav netrénovaný)', 'data/Motion_Emotion/020.mp4',
     ['--upscale', '--conf-obj', '0.12']),
    # NWPU HD (no upscale, nejrychlejší)
    ('NWPU D002_01 (HD CCTV)',
     'data/NWPU_Campus/NWPUCampusDataset_extracted/NWPUCampusDataset/videos/Test/D002_01.avi',
     ['--conf-obj', '0.15']),
    ('NWPU D014_01 (HD CCTV)',
     'data/NWPU_Campus/NWPUCampusDataset_extracted/NWPUCampusDataset/videos/Test/D014_01.avi',
     ['--conf-obj', '0.15']),
]

for i, (desc, video, extra) in enumerate(VIDEOS, 1):
    print(f'\n{"="*70}\n[{i}/{len(VIDEOS)}] {desc}\n  {video}\n{"="*70}', flush=True)
    if not os.path.exists(video):
        print(f'  ⚠ SKIP — soubor neexistuje')
        continue
    cmd = [sys.executable, 'abandoned_object_detector.py', '--video', video,
           '--sample-fps', '5', '--skip-weapon', '--save-video', '--output-dir', OUT] + extra
    subprocess.run(cmd, check=False)

print(f'\n=== TEST BATCH DONE — výstupy v {OUT}/ ===')
