"""
Test YOLOv8 Weapon Detection modelu
Detekuje zbrane (pistole, noze, dalsi nebezpecne predmety)
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time


class WeaponDetector:
    """YOLOv8 detektor zbrani - wrapper"""

    def __init__(self, model_path, device='cuda'):
        """
        Inicializace YOLOv8 weapon detection modelu

        Args:
            model_path: Cesta k YOLOv8 modelu (.pt soubor)
            device: 'cuda' nebo 'cpu'

        Returns:
            None
        """
        self.device = device
        self.model = YOLO(model_path)

        # Ziskej nazvy trid z modelu
        self.class_names = self.model.names
        print(f"Weapon Detection model nacten uspesne: {model_path}")
        print(f"Detekovane tridy: {self.class_names}")

    def detect(self, image_path, conf_threshold=0.5):
        """
        Detekce zbrani na obrazku

        Args:
            image_path: Cesta k obrazku (nebo numpy array)
            conf_threshold: Prah spolehlivosti detekce (0-1)

        Returns:
            boxes: numpy array tvaru (N, 6) s [x1, y1, x2, y2, conf, class_id]
        """
        # Nacteni obrazku
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path

        # Spusteni detekce
        results = self.model(img, conf=conf_threshold, verbose=False)

        # Extrakce vysledku
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            # Spojeni vsech informaci
            result = np.column_stack([boxes, confs, classes])
        else:
            result = np.empty((0, 6))

        return result

    def get_class_name(self, class_id):
        """
        Ziskani nazvu tridy podle ID

        Args:
            class_id: ID tridy (int)

        Returns:
            name: Nazev tridy (str)
        """
        return self.class_names[int(class_id)]


def visualize_weapons(image_path, detections, class_names, title="Weapon Detection", save_path=None):
    """
    Vizualizace detekovanych zbrani

    Args:
        image_path: Cesta k obrazku
        detections: numpy array tvaru (N, 6) s [x1, y1, x2, y2, conf, class_id]
        class_names: Slovnik s nazvy trid
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

    if len(detections) > 0:
        # Barvy podle typu zbrane
        color_map = {
            'gun': 'red',
            'pistol': 'red',
            'rifle': 'darkred',
            'knife': 'orange',
            'weapon': 'red',
        }

        for idx, det in enumerate(detections):
            x1, y1, x2, y2, conf, class_id = det
            class_name = class_names[int(class_id)]

            # Vyber barvu podle typu
            color = 'red'  # default
            for key, value in color_map.items():
                if key in class_name.lower():
                    color = value
                    break

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                fill=False, edgecolor=color, linewidth=3, alpha=0.8)
            ax.add_patch(rect)

            label = f'{class_name}: {conf:.2f}'
            ax.text(x1, y1 - 10, label, fontsize=12, color='white', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Vizualizace ulozena do: {save_path}")

    plt.show()


def calculate_statistics(detections, class_names):
    """
    Vypocet statistik detekovanych zbrani

    Args:
        detections: numpy array tvaru (N, 6) s [x1, y1, x2, y2, conf, class_id]
        class_names: Slovnik s nazvy trid

    Returns:
        stats: slovnik se statistikami
    """
    if len(detections) == 0:
        return {
            'count': 0,
            'by_class': {},
            'avg_confidence': 0,
            'min_confidence': 0,
            'max_confidence': 0,
        }

    confidences = detections[:, 4]
    class_ids = detections[:, 5].astype(int)

    # Pocet detekcí podle tridy
    by_class = {}
    for class_id in np.unique(class_ids):
        class_name = class_names[class_id]
        count = np.sum(class_ids == class_id)
        by_class[class_name] = count

    stats = {
        'count': len(detections),
        'by_class': by_class,
        'avg_confidence': np.mean(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
    }

    return stats


def print_statistics(stats):
    """
    Vytiskne statistiky detekovanych zbrani

    Args:
        stats: Slovnik se statistikami z calculate_statistics()

    Returns:
        None
    """
    print(f"\n{'='*70}")
    print("STATISTIKY DETEKCE ZBRANI")
    print(f"{'='*70}")
    print(f"  Celkem detekovano: {stats['count']}")

    if stats['count'] > 0:
        print(f"\n  Detekce podle typu:")
        for class_name, count in stats['by_class'].items():
            print(f"    - {class_name}: {count}")

        print(f"\n  Spolehlivost:")
        print(f"    - Prumerna: {stats['avg_confidence']:.3f}")
        print(f"    - Min/Max: {stats['min_confidence']:.3f} / {stats['max_confidence']:.3f}")
    print(f"{'='*70}\n")


def test_weapon_detection():
    """
    Test YOLOv8 weapon detection na testovacim obrazku

    Args:
        None

    Returns:
        detections: numpy array s detekcemi
    """
    # Cesty k souborum
    model_path = 'models/weapon_detection/best (3).pt'
    test_image = 'frames_0.5_upscaled/004/004_0065.jpg'

    # Kontrola existence souboru
    if not os.path.exists(model_path):
        print(f"Model nenalezen: {model_path}")
        print("Prosim zkontrolujte cestu k modelu.")
        return None

    if not os.path.exists(test_image):
        print(f"Testovaci obrazek nenalezen: {test_image}")
        print("Pouziji jiny testovaci obrazek...")
        # Zkus najit jiny obrazek
        possible_images = [
            'frames_0.5_upscaled/002/001_0076.jpg',
            'frames_0.5/002/002_0000.jpg',
        ]
        for img in possible_images:
            if os.path.exists(img):
                test_image = img
                break
        else:
            print("Zadny testovaci obrazek nenalezen.")
            return None

    print(f"Pouzity testovaci obrazek: {test_image}")

    # Inicializace detektoru
    print("\nInicializace Weapon Detection modelu...")
    detector = WeaponDetector(model_path, device='cuda')

    # Spusteni detekce
    print(f"\nSpousteni detekce zbrani na {test_image}...")
    start_time = time.time()
    detections = detector.detect(test_image, conf_threshold=0.3)
    inference_time = time.time() - start_time

    print(f"\nVysledky:")
    print(f"  Cas inference: {inference_time:.3f}s")
    print(f"  Detekovano: {len(detections)} objektu")

    # Vypocet statistik
    stats = calculate_statistics(detections, detector.class_names)
    print_statistics(stats)

    # Vizualizace
    if len(detections) > 0:
        print("Vizualizace vysledku...")
        visualize_weapons(
            test_image,
            detections,
            detector.class_names,
            title=f"Weapon Detection ({len(detections)} objektu, {inference_time:.2f}s)",
            save_path="weapon_detection_result.jpg"
        )
    else:
        print("Zadne zbrane detekovany.")

    return detections


if __name__ == '__main__':
    test_weapon_detection()
