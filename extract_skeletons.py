import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import time
import torch
import mmcv
import mmpose.models
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import PoseDataSample

# --- Konfigurace ---
FRAMES_DIR = 'frames_0.5'
FRAMES_UPSCALED_DIR = 'frames_0.5_upscaled'
SKELETON_DIR = 'skeletons_yolo_11_new'
SKELETON_UPSCALED_DIR = 'skeletons_yolo_11_upscaled_2'
MODEL_PATH = 'yolo11x-pose.pt'
YOLO_MODEL_DETECTION = 'models/yolo/yolo11x.pt'
SUPER_RESOLUTION_MODEL_PATH = 'models/super_resolution'
VITPOSE_MODEL_PATH = 'models/vitpose'

# Zajištění existence výstupních složek
os.makedirs(FRAMES_UPSCALED_DIR, exist_ok=True)
os.makedirs(SKELETON_UPSCALED_DIR, exist_ok=True)

# Načtení YOLOv11 Pose modelu jednou
tmp_model = YOLO(MODEL_PATH)

# Modely
_DETECTION_MODEL = None
_VITPOSE_MODEL = None
_DATASET_INFO = None

# --- Nastavení super-resolution ---
_SR_NET = None
_SR_SCALE = 4
_SR_PB = os.path.join(SUPER_RESOLUTION_MODEL_PATH, "ESPCN_x4.pb")


def init_yolo_vitpose_models():
    """
    Inicializace YOLO detekčního modelu a ViTPose modelu pro MMPose 1.3.2
    """
    global _DETECTION_MODEL, _VITPOSE_MODEL, _DATASET_INFO

    if _DETECTION_MODEL is None:
        _DETECTION_MODEL = YOLO(YOLO_MODEL_DETECTION)
        print("YOLO detekční model načten")

    if _VITPOSE_MODEL is None:
        try:
            config_file = os.path.join(VITPOSE_MODEL_PATH, 'ViTPose_huge_256x192.py')
            checkpoint_file = os.path.join(VITPOSE_MODEL_PATH, 'vitpose_huge.pth')

            if not os.path.exists(checkpoint_file):
                print(f"ViTPose checkpoint nenalezen: {checkpoint_file}")
                return

            if not os.path.exists(config_file):
                print(f"Konfigurační soubor nenalezen: {config_file}")
                return

            # Pro MMPose 1.3.2 - použití nového MMPoseInferencer API
            from mmpose.apis import MMPoseInferencer

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Inicializace s konfigurací a checkpointem
            _VITPOSE_MODEL = MMPoseInferencer(
                pose2d=config_file,
                pose2d_weights=checkpoint_file,
                device=device,
                scope='mmpose'
            )

            print("ViTPose model úspěšně načten s MMPose 1.3.2 API")
            _DATASET_INFO = None  # Není potřeba s novým API

        except Exception as e:
            print(f"Nepodařilo se načíst ViTPose: {e}")
            print("Zkouším alternativní přístup...")

            # Alternativa: Použití vestavěného názvu ViTPose modelu
            try:
                from mmpose.apis import MMPoseInferencer

                # Zkusíme použít předefinovaný název modelu
                _VITPOSE_MODEL = MMPoseInferencer(
                    pose2d='human',  # Použití výchozího modelu pro lidskou pózu
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                print("ViTPose načten s výchozím modelem pro lidskou pózu")

            except Exception as e2:
                print(f"Alternativní přístup také selhal: {e2}")
                _VITPOSE_MODEL = None

    return _DETECTION_MODEL, _VITPOSE_MODEL


def init_super_resolution():
    """
    Inicializace super-resolution modelu s ošetřením chyb
    """
    global _SR_NET

    if _SR_NET is not None:
        return _SR_NET

    try:
        # Kontrola dostupnosti dnn_superres
        if hasattr(cv2, 'dnn_superres'):
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
        else:
            print("dnn_superres není dostupné ve vaší verzi OpenCV")
            return None

        # Načtení modelu
        sr.readModel(_SR_PB)
        sr.setModel('espcn', _SR_SCALE)

        # Nastavení backendu (CPU pro kompatibilitu)
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        print("Super-resolution model úspěšně inicializován (CPU backend)")
        _SR_NET = sr
        return sr

    except Exception as e:
        print(f"Nepodařilo se inicializovat super-resolution: {e}")
        return None


def apply_super_resolution(img, max_size=1500):
    """
    Aplikace super-resolution na obrázek s omezením velikosti kvůli výkonu
    """
    sr = init_super_resolution()
    if sr is None:
        return img

    try:
        h, w = img.shape[:2]

        # Nezvětšovat velmi velké obrázky (příliš pomalé)
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img, (new_w, new_h))

            # Aplikace super-resolution na menší obrázek
            sr_result = sr.upsample(img_resized)

            # Změna zpět na původní velikost
            final_result = cv2.resize(sr_result, (w, h))
            return final_result
        else:
            # Přímá aplikace super-resolution
            return sr.upsample(img)

    except Exception as e:
        print(f"Super-resolution selhalo: {e}")
        return img


def create_upscaled_frames(input_folder, output_folder):
    """
    Vytvoření zvětšených verzí všech snímků ve složce
    """
    os.makedirs(output_folder, exist_ok=True)

    files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])

    print(f"Vytváření zvětšených snímků pro {os.path.basename(input_folder)}...")

    # Inicializace super-resolution jednou
    init_super_resolution()

    for fname in tqdm(files, desc=f"Zvětšování {os.path.basename(input_folder)}"):
        input_path = os.path.join(input_folder, fname)
        output_path = os.path.join(output_folder, fname)

        # Přeskočení, pokud již existuje
        if os.path.exists(output_path):
            continue

        img = cv2.imread(input_path)
        if img is None:
            continue

        # Aplikace super-resolution nebo vysoce kvalitního zvětšení
        upscaled_img = apply_super_resolution(img)

        # Uložení zvětšeného obrázku
        cv2.imwrite(output_path, upscaled_img)

    print(f"Zvětšování dokončeno pro {os.path.basename(input_folder)}")


def enhanced_detection_pipeline(img, model):
    """
    Vylepšená detekční pipeline optimalizovaná pro zvětšené obrázky
    Používá TOP-DOWN přístup: nejdřív detekce osob, pak extrakce skeletonu
    """
    all_kps = []

    # Pro zvětšené obrázky můžeme použít agresivnější nastavení
    enhanced_imgs = []

    # 1. Původní zvětšený obrázek
    enhanced_imgs.append(img)

    # 2. CLAHE vylepšení na zvětšeném obrázku
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced1 = cv2.merge([l, a, b])
    enhanced1 = cv2.cvtColor(enhanced1, cv2.COLOR_LAB2BGR)
    enhanced_imgs.append(enhanced1)

    # 3. Histogramová ekvalizace
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    enhanced2 = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    enhanced_imgs.append(enhanced2)

    # 4. Gamma korekce
    gamma = 0.8
    enhanced3 = np.power(img / 255.0, gamma) * 255.0
    enhanced3 = enhanced3.astype(np.uint8)
    enhanced_imgs.append(enhanced3)

    # Vyšší konfidence pro zvětšené obrázky (lepší kvalita detekcí)
    configs = [
        {'conf': 0.25, 'imgsz': 1280, 'iou': 0.45},  # Vysoká konfidence, dobrá vyváženost
        {'conf': 0.20, 'imgsz': 1024, 'iou': 0.4},   # Středně vysoká konfidence
        {'conf': 0.30, 'imgsz': 1536, 'iou': 0.5},   # Vysoká konfidence, vysoké rozlišení
        {'conf': 0.35, 'imgsz': 1792, 'iou': 0.55},  # Velmi vysoká konfidence, velmi vysoké rozlišení
        {'conf': 0.15, 'imgsz': 2048, 'iou': 0.35},  # Střední konfidence, max rozlišení (zachycení vzdálených osob)
    ]

    # Zpracování každého vylepšeného obrázku
    for img_idx, enhanced_img in enumerate(enhanced_imgs):
        if img_idx == 0:  # Původní zvětšený
            config_indices = [0, 2]  # Vysoká konfidence, dobré pokrytí
        elif img_idx == 1:  # CLAHE
            config_indices = [1, 3]  # Středně vysoká a velmi vysoká konfidence
        elif img_idx == 2:  # Histogramově ekvalizovaný
            config_indices = [0, 4]  # Vysoká konfidence + pokrytí vzdálených osob
        else:  # Gamma korigovaný
            config_indices = [1]

        for config_idx in config_indices:
            try:
                res = model(enhanced_img, verbose=False, **configs[config_idx])
                data = res[0].keypoints.data
                if data.numel() > 0:
                    kps = data.cpu().numpy()[:, :, :2]
                    all_kps.append(kps)
            except Exception as e:
                print(f"Detekce selhala pro vylepšení {img_idx}, config {config_idx}: {e}")
                continue

    if not all_kps:
        return np.full((1, 17, 2), np.nan, dtype=np.float32)

    combined = np.vstack(all_kps)

    # Vylepšené odstranění duplikátů pro davy
    return remove_duplicate_skeletons_advanced(combined, threshold=12)  # Ještě přísnější pro davy


def remove_duplicate_skeletons_advanced(keypoints, threshold=15):
    """
    Vylepšené odstranění duplikátů s kontrolou podobnosti skeletonu - optimalizováno pro davy
    """
    if len(keypoints) <= 1:
        return keypoints

    # Výpočet centroidů a příznaků
    centroids = []
    valid_idx = []
    skeleton_features = []

    for i, sk in enumerate(keypoints):
        if not np.isnan(sk).all():
            c = np.nanmean(sk, axis=0)
            if not np.isnan(c).any():
                centroids.append(c)
                valid_idx.append(i)

                # Výpočet příznaků skeletu pro kontrolu podobnosti
                features = []

                # Šířka ramen
                if not np.isnan(sk[5:7]).any():
                    shoulder_width = np.linalg.norm(sk[5] - sk[6])
                else:
                    shoulder_width = 0
                features.append(shoulder_width)

                # Šířka boků
                if not np.isnan(sk[11:13]).any():
                    hip_width = np.linalg.norm(sk[11] - sk[12])
                else:
                    hip_width = 0
                features.append(hip_width)

                # Výška trupu (ramena k bokům)
                if not (np.isnan(sk[5:7]).any() or np.isnan(sk[11:13]).any()):
                    shoulder_center = np.mean(sk[5:7], axis=0)
                    hip_center = np.mean(sk[11:13], axis=0)
                    torso_height = abs(shoulder_center[1] - hip_center[1])
                else:
                    torso_height = 0
                features.append(torso_height)

                skeleton_features.append(features)

    if not centroids:
        return keypoints

    centroids = np.array(centroids)
    skeleton_features = np.array(skeleton_features)
    unique_mask = np.ones(len(centroids), dtype=bool)

    for i in range(len(centroids)):
        if not unique_mask[i]:
            continue
        for j in range(i + 1, len(centroids)):
            if unique_mask[j]:
                centroid_dist = np.linalg.norm(centroids[i] - centroids[j])

                # Kontrola podobnosti skeletu
                if skeleton_features[i].sum() > 0 and skeleton_features[j].sum() > 0:
                    feature_dist = np.linalg.norm(skeleton_features[i] - skeleton_features[j])
                    feature_similarity = feature_dist / max(np.linalg.norm(skeleton_features[i]), 1)
                else:
                    feature_similarity = 0

                # Odstranění duplikátu pokud blízký centroid A podobný skeleton
                if centroid_dist < threshold and (centroid_dist < threshold * 0.7 or feature_similarity < 0.25):
                    unique_mask[j] = False

    unique_inds = [valid_idx[i] for i in range(len(valid_idx)) if unique_mask[i]]
    return keypoints[unique_inds]


def process_upscaled_folder(frames_dir, skeleton_dir, model):
    """
    Zpracování zvětšených snímků pro detekci skeletonu
    """
    os.makedirs(skeleton_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))

    total_frames = len(files)
    detected_frames = 0
    total_people = 0

    for fname in tqdm(files, desc=f"Detekce skeletů v {os.path.basename(frames_dir)}"):
        path = os.path.join(frames_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        # Použití vylepšené detekční pipeline pro zvětšené obrázky
        kps = enhanced_detection_pipeline(img, model)

        if not np.isnan(kps).all():
            detected_frames += 1
            total_people += len(kps)

        out_path = os.path.join(skeleton_dir, fname.replace('.jpg', '.npy'))
        np.save(out_path, kps)

    rate = 100 * detected_frames / total_frames if total_frames else 0
    avg_people = total_people / total_frames if total_frames else 0
    print(
        f"{os.path.basename(frames_dir)}: {detected_frames}/{total_frames} detekováno ({rate:.1f}%), průměr osob/snímek {avg_people:.1f}")
    return total_frames, detected_frames


def process_video_with_upscaling(video_name):
    """
    Kompletní pipeline: vytvoření zvětšených snímků -> detekce skeletů
    """
    print(f"\n=== Zpracování {video_name} se zvětšením ===")

    # Cesty
    input_frames = os.path.join(FRAMES_DIR, video_name)
    upscaled_frames = os.path.join(FRAMES_UPSCALED_DIR, video_name)
    upscaled_skeletons = os.path.join(SKELETON_UPSCALED_DIR, video_name)

    if not os.path.exists(input_frames):
        print(f"Vstupní složka nenalezena: {input_frames}")
        return

    # Krok 1: Vytvoření zvětšených snímků
    print("Krok 1: Vytváření zvětšených snímků...")
    start_time = time.time()
    create_upscaled_frames(input_frames, upscaled_frames)
    upscale_time = time.time() - start_time
    print(f"Zvětšování dokončeno za {upscale_time:.2f} sekund")

    # Krok 2: Detekce skeletů ve zvětšených snímcích
    print("Krok 2: Detekce skeletů ve zvětšených snímcích...")
    start_time = time.time()
    total, detected = process_upscaled_folder(upscaled_frames, upscaled_skeletons, tmp_model)
    detection_time = time.time() - start_time
    print(f"Detekce skeletů dokončena za {detection_time:.2f} sekund")

    print(f"Celkový čas zpracování: {upscale_time + detection_time:.2f} sekund")
    print(f"Výsledky uloženy do: {upscaled_skeletons}")


def process_single_image_skeleton_detection(img_path, model, save_skeleton=True, save_visualization=True):
    """
    Zpracování jednoho obrázku pro detekci skeletonu

    Args:
        img_path: Cesta k souboru obrázku
        model: YOLO pose model
        save_skeleton: Zda uložit data skeletu jako .npy
        save_visualization: Zda uložit vizualizaci obrázku

    Returns:
        keypoints: Pole detekovaných klíčových bodů
    """
    print(f"\n=== Zpracování jednoho obrázku: {img_path} ===")

    if not os.path.exists(img_path):
        print(f"Chyba: Obrázek nenalezen na {img_path}")
        return None

    # Načtení obrázku
    img = cv2.imread(img_path)
    if img is None:
        print(f"Chyba: Nepodařilo se načíst obrázek z {img_path}")
        return None

    print(f"Obrázek načten: {img.shape}")

    # Krok 1: Detekce skeletů
    start_time = time.time()
    keypoints = enhanced_detection_pipeline(img, model)
    detection_time = time.time() - start_time

    num_people = len(keypoints) if not np.isnan(keypoints).all() else 0
    print(f"Detekce dokončena za {detection_time:.2f} sekund")
    print(f"Nalezeno {num_people} osob")

    # Krok 2: Uložení výsledků
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    if save_skeleton:
        # Uložení dat skeletu
        skeleton_output_dir = os.path.join(SKELETON_UPSCALED_DIR, 'single_tests')
        os.makedirs(skeleton_output_dir, exist_ok=True)
        skeleton_file = os.path.join(skeleton_output_dir, f"{base_name}_skeletons.npy")
        np.save(skeleton_file, keypoints)
        print(f"Data skeletu uložena do: {skeleton_file}")

    if save_visualization:
        # Vytvoření a uložení vizualizace
        vis_img = visualize_skeletons_on_image(img, keypoints)
        vis_output_dir = os.path.join(SKELETON_UPSCALED_DIR, 'visualizations')
        os.makedirs(vis_output_dir, exist_ok=True)
        vis_file = os.path.join(vis_output_dir, f"{base_name}_visualization.jpg")
        cv2.imwrite(vis_file, vis_img)
        print(f"Vizualizace uložena do: {vis_file}")

    return keypoints


def visualize_skeletons_on_image(img, keypoints):
    """
    Vykreslení skeletů na obrázek pro vizualizaci
    """
    if np.isnan(keypoints).all():
        return img

    img_vis = img.copy()

    # COCO skeleton spojení
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Hlava
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Paže
        (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Nohy
        (11, 12)  # Boky
    ]

    # Barvy pro různé osoby
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (128, 0, 255), (255, 128, 128), (128, 255, 128), (128, 128, 255),
    ]

    for person_idx, skeleton in enumerate(keypoints):
        if skeleton.shape != (17, 2) or np.isnan(skeleton).all():
            continue

        color = colors[person_idx % len(colors)]

        # Vykreslení spojení
        for connection in connections:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(skeleton) and pt2_idx < len(skeleton):
                pt1 = skeleton[pt1_idx]
                pt2 = skeleton[pt2_idx]

                if not (np.isnan(pt1).any() or np.isnan(pt2).any()):
                    cv2.line(img_vis,
                             (int(pt1[0]), int(pt1[1])),
                             (int(pt2[0]), int(pt2[1])),
                             color, 3)

        # Vykreslení kloubů
        for joint_idx, joint in enumerate(skeleton):
            if not np.isnan(joint).any():
                cv2.circle(img_vis, (int(joint[0]), int(joint[1])), 6, color, -1)
                cv2.circle(img_vis, (int(joint[0]), int(joint[1])), 7, (255, 255, 255), 2)

        # Přidání popisku osoby
        centroid = np.nanmean(skeleton, axis=0)
        if not np.isnan(centroid).any():
            label = f'Osoba {person_idx}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # Získání velikosti textu pro pozadí
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Vykreslení obdélníku pozadí
            cv2.rectangle(img_vis,
                          (int(centroid[0]) - text_width // 2 - 5,
                           int(centroid[1]) - 60),
                          (int(centroid[0]) + text_width // 2 + 5,
                           int(centroid[1]) - 35),
                          color, -1)

            # Vykreslení textu
            cv2.putText(img_vis, label,
                        (int(centroid[0]) - text_width // 2,
                         int(centroid[1]) - 40),
                        font, font_scale, (255, 255, 255), thickness)

    # Přidání nadpisu
    title = f"Detekce skeletů - detekováno {len(keypoints)} osob"
    cv2.putText(img_vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img_vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

    return img_vis


def test_single_upscaled_image(img_name):
    """
    Test detekce skeletu na jednom zvětšeném obrázku

    Args:
        img_name: Relativní cesta od FRAMES_UPSCALED_DIR (např. '001/001_0076.jpg')
    """
    print(f"\n=== Testování jednoho zvětšeného obrázku: {img_name} ===")

    # Plná cesta ke zvětšenému obrázku
    img_path = os.path.join(FRAMES_UPSCALED_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"Chyba: Zvětšený obrázek nenalezen na {img_path}")
        print(f"Ujistěte se, že jste nejdřív spustili krok zvětšování nebo zkontrolujte cestu")
        return None

    # Zpracování obrázku
    keypoints = process_single_image_skeleton_detection(
        img_path,
        tmp_model,
        save_skeleton=True,
        save_visualization=True
    )

    if keypoints is not None:
        print(f"\nShrnutí výsledků:")
        print(f"- Počet detekovaných osob: {len(keypoints)}")
        print(f"- Tvar klíčových bodů: {keypoints.shape}")
        print(f"- Validní detekce: {np.sum(~np.isnan(keypoints).all(axis=(1, 2)))}")

    return keypoints


# Hlavní sekce
if __name__ == '__main__':

    # Zpracování videa 001 s pipeline pro zvětšení
    process_video_with_upscaling('001')

    # Test jednoho zvětšeného obrázku
    img_to_process = '001/001_0076.jpg'

    # Možnost 1: Test pouze zvětšeného obrázku
    #test_single_upscaled_image(img_to_process)
    #init_yolo_vitpose_models()