import os
import cv2

# Nastavení
INPUT_DIR  = 'data/Motion_Emotion'  # cesta k tvé složce s MP4
OUTPUT_DIR = 'frames_0.5'                  # kam to uložíme
FPS        = 2                         # rámečky za sekundu
MAX_FRAMES = None                      # oddělej limit nebo dej např. 100

# Vytvoříme výstupní složku
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith('.mp4'):
        continue

    video_path = os.path.join(INPUT_DIR, fname)
    base = os.path.splitext(fname)[0]
    out_dir = os.path.join(OUTPUT_DIR, base)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = int(total_fps / FPS) if total_fps >= FPS else 1

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Uložíme jen každý n-tý frame podle FPS
        if count % frame_interval == 0:
            out_path = os.path.join(out_dir, f"{base}_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
            if MAX_FRAMES and saved >= MAX_FRAMES:
                break
        count += 1

    cap.release()
    print(f"✔ Uloženo {saved} snímků z videa {fname}")
