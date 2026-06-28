"""Regeneruje demo videa plynule do demo_outputs/. Spustí 3 reprezentativní scény."""
import subprocess
import sys

PY = sys.executable
DEMOS = [
    # (popis, video, extra args)
    ('NWPU HD (1080p, osoby u brány)',
     'data/NWPU_Campus/NWPUCampusDataset_extracted/NWPUCampusDataset/videos/Test/D001_03.avi',
     ['--sample-fps', '5', '--skip-weapon', '--conf-obj', '0.15']),
    ('CAVIAR Fight_Chase (agresoři, upscale, weapon ON)',
     'data/CAVIAR/Fight_Chase/Fight_Chase.mpg',
     ['--sample-fps', '5', '--upscale', '--conf-obj', '0.10', '--conf-weapon', '0.40']),
    ('MED 005 netrénované (dav + batoh, upscale, weapon ON)',
     'data/Motion_Emotion/005.mp4',
     ['--sample-fps', '5', '--upscale', '--conf-obj', '0.12', '--conf-weapon', '0.40']),
]

for i, (desc, video, extra) in enumerate(DEMOS, 1):
    print(f'\n{"="*70}\n[{i}/{len(DEMOS)}] {desc}\n{"="*70}', flush=True)
    cmd = [PY, 'abandoned_object_detector.py', '--video', video, '--save-video'] + extra
    subprocess.run(cmd, check=False)

print('\n=== REGEN DONE — výstupy v demo_outputs/ ===')
