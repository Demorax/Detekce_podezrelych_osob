"""
Test RT-DETR a YOLO detekce osob na testovacich obrazcich
Porovnani vykonnosti obou modelu
"""

import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time

# Pridani RT-DETR do cesty
sys.path.insert(0, os.path.join(os.getcwd(), 'models', 'rtdetr'))
from src.core import YAMLConfig


class RTDETRDetector:
    """RT-DETR detektor osob - wrapper"""

    def __init__(self, config_path, checkpoint_path, device='cuda'):
        """
        Inicializace RT-DETR modelu

        Args:
            config_path: Cesta ke konfiguracnimu souboru (.yml)
            checkpoint_path: Cesta k modelu (checkpoint .pth soubor)
            device: 'cuda' nebo 'cpu'

        Returns:
            None
        """
        self.device = device

        # Nacteni konfigurace
        cfg = YAMLConfig(config_path, resume=checkpoint_path)

        # Nacteni checkpointu
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # Nacteni vah modelu
        cfg.model.load_state_dict(state)

        # Vytvoreni modelu pro inferenci
        class Model(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        self.model = Model(cfg).to(device)
        self.model.eval()

        # Transformace obrazku
        self.transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

        print(f"RT-DETR model nacten uspesne na {device}")

    def detect(self, image_path, conf_threshold=0.6, person_class_id=0):
        """
        Detekce osob na obrazku

        Args:
            image_path: Cesta k obrazku (nebo numpy array/PIL Image)
            conf_threshold: Prah spolehlivosti detekce
            person_class_id: COCO ID pro tridu 'osoba' (0)

        Returns:
            boxes: numpy array tvaru (N, 5) s [x1, y1, x2, y2, conf]
        """
        # Nacteni obrazku
        if isinstance(image_path, str):
            im_pil = Image.open(image_path).convert('RGB')
        else:
            if isinstance(image_path, np.ndarray):
                im_pil = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
            else:
                im_pil = image_path

        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(self.device)

        # Transformace a inference
        im_data = self.transforms(im_pil)[None].to(self.device)

        with torch.no_grad():
            labels, boxes, scores = self.model(im_data, orig_size)

        # Filtrovani pouze trid 'osoba'
        labels = labels[0].cpu().numpy()
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()

        # Filtr podle tridy a spolehlivosti
        person_mask = (labels == person_class_id) & (scores > conf_threshold)
        person_boxes = boxes[person_mask]
        person_scores = scores[person_mask]

        # Spojeni boxu a skore
        if len(person_boxes) > 0:
            result = np.column_stack([person_boxes, person_scores])
        else:
            result = np.array([])

        return result


class YOLODetector:
    """YOLO detektor osob - wrapper"""

    def __init__(self, model_path, device='cuda'):
        """
        Inicializace YOLO modelu

        Args:
            model_path: Cesta k YOLO modelu (.pt soubor)
            device: 'cuda' nebo 'cpu'

        Returns:
            None
        """
        from ultralytics import YOLO

        self.device = device
        self.model = YOLO(model_path)
        print(f"YOLO model nacten uspesne: {model_path}")

    def detect(self, image_path, conf_threshold=0.6, person_class_id=0, imgsz=640):
        """
        Detekce osob na obrazku

        Args:
            image_path: Cesta k obrazku (nebo numpy array)
            conf_threshold: Prah spolehlivosti detekce
            person_class_id: COCO ID pro tridu 'osoba' (0)
            imgsz: Velikost obrazku pro inferenci (default: 640)

        Returns:
            boxes: numpy array tvaru (N, 5) s [x1, y1, x2, y2, conf]
        """
        # Nacteni obrazku
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path

        # Spusteni detekce s explicitni velikosti obrazku
        results = self.model(img, conf=conf_threshold, imgsz=imgsz, verbose=False)

        # Filtrovani pouze trid 'osoba'
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            person_mask = classes == person_class_id
            person_boxes = boxes[person_mask]
            person_confs = confs[person_mask]

            if len(person_boxes) > 0:
                result = np.column_stack([person_boxes, person_confs])
            else:
                result = np.array([])
        else:
            result = np.array([])

        return result


def calculate_iou(box1, box2):
    """
    Vypocet IoU (Intersection over Union) mezi dvema boxy

    Args:
        box1, box2: [x1, y1, x2, y2, ...]

    Returns:
        iou: IoU skore (0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_statistics(boxes, image_shape=None):
    """
    Vypocet statistik detekci

    Args:
        boxes: numpy array tvaru (N, 5) s [x1, y1, x2, y2, conf]
        image_shape: (height, width) obrazku

    Returns:
        stats: slovnik se statistikami
    """
    if len(boxes) == 0:
        return {
            'count': 0,
            'avg_confidence': 0,
            'min_confidence': 0,
            'max_confidence': 0,
            'avg_box_area': 0,
            'avg_box_width': 0,
            'avg_box_height': 0
        }

    confidences = boxes[:, 4]
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights

    stats = {
        'count': len(boxes),
        'avg_confidence': np.mean(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'std_confidence': np.std(confidences),
        'avg_box_area': np.mean(areas),
        'avg_box_width': np.mean(widths),
        'avg_box_height': np.mean(heights),
    }

    # Pokud je k dispozici velikost obrazku, vypocitej relativni velikosti
    if image_shape is not None:
        img_area = image_shape[0] * image_shape[1]
        stats['avg_relative_area'] = np.mean(areas) / img_area * 100  # v procentech

    return stats


def match_detections(boxes1, boxes2, iou_threshold=0.5):
    """
    Parovani detekci mezi dvema sadami boxu pomoci IoU

    Args:
        boxes1: numpy array tvaru (N, 5) - prvni sada boxu [x1, y1, x2, y2, conf]
        boxes2: numpy array tvaru (M, 5) - druha sada boxu [x1, y1, x2, y2, conf]
        iou_threshold: minimalni IoU pro sparovani (0-1)

    Returns:
        tuple: (matched, only_in_1, only_in_2, avg_iou)
            - matched: pocet sparovanych detekci
            - only_in_1: pocet detekci pouze v prvni sade
            - only_in_2: pocet detekci pouze v druhe sade
            - avg_iou: prumerne IoU sparovanych detekci
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return 0, len(boxes1), len(boxes2), 0

    matched = 0
    matched_ious = []
    matched_indices_2 = set()

    for box1 in boxes1:
        best_iou = 0
        best_idx = -1

        for idx, box2 in enumerate(boxes2):
            if idx in matched_indices_2:
                continue

            iou = calculate_iou(box1, box2)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold:
            matched += 1
            matched_ious.append(best_iou)
            matched_indices_2.add(best_idx)

    only_in_1 = len(boxes1) - matched
    only_in_2 = len(boxes2) - matched
    avg_iou = np.mean(matched_ious) if matched_ious else 0

    return matched, only_in_1, only_in_2, avg_iou


def visualize_detections(image_path, boxes, title="Detection", save_path=None):
    """
    Vizualizace vysledku detekce

    Args:
        image_path: Cesta k obrazku
        boxes: numpy array tvaru (N, 5) s [x1, y1, x2, y2, conf]
        title: Nadpis grafu
        save_path: Cesta pro ulozeni obrazku (None = neulozi se)

    Returns:
        None
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=16, weight='bold')
    ax.axis('off')

    if len(boxes) > 0:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(boxes)))

        for idx, (box, color) in enumerate(zip(boxes, colors)):
            x1, y1, x2, y2, conf = box

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                fill=False, edgecolor=color[:3], linewidth=3, alpha=0.8)
            ax.add_patch(rect)

            label = f'Person {idx}: {conf:.2f}'
            ax.text(x1, y1 - 10, label, fontsize=10, color='white', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color[:3], alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Vizualizace ulozena do: {save_path}")

    plt.show()


def print_statistics(name, stats):
    """
    Vytiskne statistiky detekci

    Args:
        name: Nazev modelu (pro tisk)
        stats: Slovnik se statistikami z calculate_statistics()

    Returns:
        None
    """
    print(f"\n{name} Statistiky:")
    print(f"  Pocet detekci: {stats['count']}")
    if stats['count'] > 0:
        print(f"  Prumerna spolehlivost: {stats['avg_confidence']:.3f}")
        print(f"  Min/Max spolehlivost: {stats['min_confidence']:.3f} / {stats['max_confidence']:.3f}")
        print(f"  Std spolehlivost: {stats['std_confidence']:.3f}")
        print(f"  Prumerna plocha boxu: {stats['avg_box_area']:.0f} px²")
        print(f"  Prumerna velikost boxu: {stats['avg_box_width']:.0f} x {stats['avg_box_height']:.0f} px")
        if 'avg_relative_area' in stats:
            print(f"  Prumerna relativni plocha: {stats['avg_relative_area']:.2f}%")


def test_rtdetr():
    """
    Test RT-DETR na testovacim obrazku

    Args:
        None

    Returns:
        boxes: numpy array s detekcemi tvaru (N, 5)
    """

    # Cesty k souborum
    config_path = 'models/rtdetr/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml'
    checkpoint_path = 'models/rtdetr/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth'
    test_image = 'frames_0.5_upscaled/002/002_0000.jpg'

    if not os.path.exists(test_image):
        print(f"Testovaci obrazek nenalezen: {test_image}")
        print("Nejprve extrahujte a zvyste rozliseni snimku.")
        return

    # Kontrola dostupnosti CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Pouzite zarizeni: {device}")

    # Inicializace detektoru
    print("Inicializace RT-DETR detektoru...")
    detector = RTDETRDetector(config_path, checkpoint_path, device=device)

    # Spusteni detekce
    print(f"\nSpousteni detekce na {test_image}...")
    start_time = time.time()
    boxes = detector.detect(test_image, conf_threshold=0.3)
    inference_time = time.time() - start_time

    print(f"\nVysledky:")
    print(f"  Cas inference: {inference_time:.3f}s")
    print(f"  Detekovano {len(boxes)} osob")

    # Vypocet statistik
    img = cv2.imread(test_image)
    stats = calculate_statistics(boxes, img.shape[:2])
    print_statistics("RT-DETR", stats)

    # Vizualizace
    print("\nVizualizace vysledku...")
    visualize_detections(
        test_image,
        boxes,
        title=f"RT-DETR Detekce ({len(boxes)} osob, {inference_time:.2f}s)",
        save_path="rtdetr_test_result.jpg"
    )

    return boxes


def plot_comparison_graphs(rtdetr_boxes, yolo_boxes, rtdetr_time, yolo_time,
                          rtdetr_stats, yolo_stats, matched, only_rtdetr, only_yolo):
    """
    Vytvori grafy porovnavajici RT-DETR a YOLO

    Args:
        rtdetr_boxes: RT-DETR detekce - numpy array tvaru (N, 5)
        yolo_boxes: YOLO detekce - numpy array tvaru (M, 5)
        rtdetr_time: Cas RT-DETR inference (sekundy)
        yolo_time: Cas YOLO inference (sekundy)
        rtdetr_stats: Statistiky RT-DETR (slovnik)
        yolo_stats: Statistiky YOLO (slovnik)
        matched: Pocet sparovanych detekci
        only_rtdetr: Pocet detekci pouze v RT-DETR
        only_yolo: Pocet detekci pouze v YOLO

    Returns:
        None (ulozi grafy do souboru)
    """
    # Vytvoreni figure s vice subploty
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Pocet detekci - sloupcovy graf
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['RT-DETR', 'YOLO11x']
    counts = [len(rtdetr_boxes), len(yolo_boxes)]
    colors = ['#ff6b6b', '#4ecdc4']
    bars = ax1.bar(models, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Počet detekcí', fontsize=12, fontweight='bold')
    ax1.set_title('Počet detekovaných osob', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Pridani hodnot nad sloupce
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 2. Cas inference - sloupcovy graf
    ax2 = fig.add_subplot(gs[0, 1])
    times = [rtdetr_time * 1000, yolo_time * 1000]  # v milisekundach
    bars = ax2.bar(models, times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Čas (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Čas inference', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.0f}ms',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 3. FPS - sloupcovy graf
    ax3 = fig.add_subplot(gs[0, 2])
    fps_values = [1/rtdetr_time, 1/yolo_time]
    bars = ax3.bar(models, fps_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_ylabel('FPS', fontsize=12, fontweight='bold')
    ax3.set_title('Snímky za sekundu (FPS)', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, fps in zip(bars, fps_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{fps:.1f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 4. Rozdeleni spolehlivosti - histogramy
    ax4 = fig.add_subplot(gs[1, :2])
    if len(rtdetr_boxes) > 0:
        ax4.hist(rtdetr_boxes[:, 4], bins=20, alpha=0.6, color='#ff6b6b',
                label='RT-DETR', edgecolor='black', linewidth=1.5)
    if len(yolo_boxes) > 0:
        ax4.hist(yolo_boxes[:, 4], bins=20, alpha=0.6, color='#4ecdc4',
                label='YOLO11x', edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Spolehlivost (confidence)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Počet detekcí', fontsize=12, fontweight='bold')
    ax4.set_title('Rozdělení spolehlivosti detekcí', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=12, loc='upper left')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    # 5. Prumerna spolehlivost - sloupcovy graf
    ax5 = fig.add_subplot(gs[1, 2])
    avg_confs = [rtdetr_stats['avg_confidence'], yolo_stats['avg_confidence']]
    bars = ax5.bar(models, avg_confs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Průměrná spolehlivost', fontsize=12, fontweight='bold')
    ax5.set_title('Průměrná spolehlivost', fontsize=14, fontweight='bold')
    ax5.set_ylim([0, 1])
    ax5.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, conf in zip(bars, avg_confs):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{conf:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 6. Rozdeleni ploch bboxu
    ax6 = fig.add_subplot(gs[2, :2])
    if len(rtdetr_boxes) > 0:
        rtdetr_areas = (rtdetr_boxes[:, 2] - rtdetr_boxes[:, 0]) * (rtdetr_boxes[:, 3] - rtdetr_boxes[:, 1])
        ax6.hist(rtdetr_areas / 1000, bins=20, alpha=0.6, color='#ff6b6b',
                label='RT-DETR', edgecolor='black', linewidth=1.5)
    if len(yolo_boxes) > 0:
        yolo_areas = (yolo_boxes[:, 2] - yolo_boxes[:, 0]) * (yolo_boxes[:, 3] - yolo_boxes[:, 1])
        ax6.hist(yolo_areas / 1000, bins=20, alpha=0.6, color='#4ecdc4',
                label='YOLO11x', edgecolor='black', linewidth=1.5)
    ax6.set_xlabel('Plocha bbox (× 1000 px²)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Počet detekcí', fontsize=12, fontweight='bold')
    ax6.set_title('Rozdělení ploch bounding boxů', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=12, loc='upper right')
    ax6.grid(axis='y', alpha=0.3, linestyle='--')

    # 7. Shoda detekci - koláčový graf
    ax7 = fig.add_subplot(gs[2, 2])
    detection_data = [matched, only_rtdetr, only_yolo]
    labels = [f'Shoda\n{matched}', f'Pouze RT-DETR\n{only_rtdetr}', f'Pouze YOLO\n{only_yolo}']
    colors_pie = ['#95e1d3', '#ff6b6b', '#4ecdc4']
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax7.pie(detection_data, labels=labels, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90, explode=explode,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'},
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    ax7.set_title('Shoda detekcí (IoU ≥ 0.5)', fontsize=14, fontweight='bold')

    # Hlavni nadpis
    fig.suptitle('RT-DETR vs YOLO11x - Porovnání výkonnosti',
                 fontsize=18, fontweight='bold', y=0.98)

    # Ulozeni
    plt.savefig('rtdetr_vs_yolo_graphs.jpg', dpi=150, bbox_inches='tight')
    print("\nGrafy porovnani ulozeny do: rtdetr_vs_yolo_graphs.jpg")
    plt.show()


def compare_with_yolo():
    """
    Porovnani RT-DETR s YOLO na stejnem obrazku

    Args:
        None

    Returns:
        None (vytvori a ulozi vizualizace + grafy)
    """

    confidence_threshold = 0.3
    test_image = 'frames_0.5_upscaled/002/001_0076.jpg'

    if not os.path.exists(test_image):
        print(f"Testovaci obrazek nenalezen: {test_image}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = cv2.imread(test_image)
    img_shape = img.shape[:2]

    # RT-DETR detekce
    print("=" * 70)
    print("RT-DETR Detekce")
    print("=" * 70)
    rtdetr_detector = RTDETRDetector(
        'models/rtdetr/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
        'models/rtdetr/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth',
        device=device
    )

    start = time.time()
    rtdetr_boxes = rtdetr_detector.detect(test_image, conf_threshold=confidence_threshold)
    rtdetr_time = time.time() - start

    print(f"RT-DETR: {len(rtdetr_boxes)} osob za {rtdetr_time:.3f}s")
    rtdetr_stats = calculate_statistics(rtdetr_boxes, img_shape)

    # YOLO detekce
    print("\n" + "=" * 70)
    print("YOLO11x Detekce")
    print("=" * 70)
    yolo_detector = YOLODetector('models/yolo/yolo11x.pt')

    start = time.time()
    yolo_boxes = yolo_detector.detect(test_image, conf_threshold=confidence_threshold)
    yolo_time = time.time() - start

    print(f"YOLO: {len(yolo_boxes)} osob za {yolo_time:.3f}s")
    yolo_stats = calculate_statistics(yolo_boxes, img_shape)

    # Statistiky
    print("\n" + "=" * 70)
    print("STATISTIKY")
    print("=" * 70)
    print_statistics("RT-DETR", rtdetr_stats)
    print_statistics("YOLO11x", yolo_stats)

    # Shoda detekci
    print("\n" + "=" * 70)
    print("SHODA DETEKCI (IoU >= 0.5)")
    print("=" * 70)
    matched, only_rtdetr, only_yolo, avg_iou = match_detections(rtdetr_boxes, yolo_boxes)
    print(f"  Sparovano: {matched}")
    print(f"  Pouze RT-DETR: {only_rtdetr}")
    print(f"  Pouze YOLO: {only_yolo}")
    print(f"  Prumerne IoU sparovanych: {avg_iou:.3f}")

    # Souhrn vykonnosti
    print("\n" + "=" * 70)
    print("SOUHRN VYKONNOSTI")
    print("=" * 70)
    print(f"RT-DETR:  {len(rtdetr_boxes):2d} osob | {rtdetr_time:.3f}s | {1/rtdetr_time:5.1f} FPS")
    print(f"YOLO11x:  {len(yolo_boxes):2d} osob | {yolo_time:.3f}s | {1/yolo_time:5.1f} FPS")
    print(f"Rychlejsi: {'RT-DETR' if rtdetr_time < yolo_time else 'YOLO11x'} o {abs(rtdetr_time - yolo_time):.3f}s")
    print("=" * 70)

    # Vytvoreni grafu porovnani
    print("\nGenerovani grafu porovnani...")
    plot_comparison_graphs(rtdetr_boxes, yolo_boxes, rtdetr_time, yolo_time,
                          rtdetr_stats, yolo_stats, matched, only_rtdetr, only_yolo)

    # Vizualizace porovnani
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # RT-DETR vizualizace
    ax1.imshow(img_rgb)
    ax1.set_title(f'RT-DETR: {len(rtdetr_boxes)} osob ({rtdetr_time:.2f}s, {1/rtdetr_time:.1f} FPS)',
                  fontsize=14, weight='bold')
    ax1.axis('off')

    if len(rtdetr_boxes) > 0:
        for idx, box in enumerate(rtdetr_boxes):
            x1, y1, x2, y2, conf = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                fill=False, edgecolor='red', linewidth=2)
            ax1.add_patch(rect)
            ax1.text(x1, y1 - 10, f'{conf:.2f}', color='red', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    # YOLO vizualizace
    ax2.imshow(img_rgb)
    ax2.set_title(f'YOLO11x: {len(yolo_boxes)} osob ({yolo_time:.2f}s, {1/yolo_time:.1f} FPS)',
                  fontsize=14, weight='bold')
    ax2.axis('off')

    if len(yolo_boxes) > 0:
        for idx, box in enumerate(yolo_boxes):
            x1, y1, x2, y2, conf = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                fill=False, edgecolor='blue', linewidth=2)
            ax2.add_patch(rect)
            ax2.text(x1, y1 - 10, f'{conf:.2f}', color='blue', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    plt.tight_layout()
    plt.savefig('rtdetr_vs_yolo_comparison.jpg', dpi=150, bbox_inches='tight')
    print("\nPorovnani vizualizace ulozeno do: rtdetr_vs_yolo_comparison.jpg")
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test RT-DETR detekce osob')
    parser.add_argument('--compare', action='store_true',
                       help='Porovnat RT-DETR s YOLO')
    args = parser.parse_args()

    if args.compare:
        compare_with_yolo()
    else:
        test_rtdetr()
