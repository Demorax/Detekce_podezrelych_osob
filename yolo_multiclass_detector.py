"""
YOLO Multi-Class Detector - detekce osob + podezřelých objektů
Používá YOLO11x s COCO třídami pro detekci zbraní a podezřelých předmětů
"""
import cv2
import numpy as np
from ultralytics import YOLO


# YOLO11x třídy relevantní pro bezpečnost (⚠️ JINÉ ID než COCO!)
SUSPICIOUS_CLASSES = {
    0: 'person',           # Pro skeleton extraction
    24: 'backpack',        # Podezřelý objekt
    26: 'handbag',         # Podezřelý objekt
    28: 'suitcase',        # Podezřelý objekt
    32: 'sports ball',     # Možná podezřelý
    34: 'baseball bat',    # ZBRAŇ! (pálka/hůl) - ⚠️ ID 34, ne 39!
    35: 'baseball glove',  # Možná podezřelý
    43: 'knife',           # ZBRAŇ! - ⚠️ ID 43, ne 77!
    76: 'scissors',        # Možná zbraň
}

# Rozdělení na kategorie
PERSON_CLASS = 0
WEAPON_CLASSES = [34, 43, 76]  # baseball bat, knife, scissors
SUSPICIOUS_OBJECT_CLASSES = [24, 26, 28, 32, 35]  # backpack, handbag, suitcase, sports ball, baseball glove


class YOLOMultiClassDetector:
    """
    YOLO detektor pro osoby + podezřelé objekty

    Args:
        model_path: Cesta k YOLO modelu (např. yolo11x.pt)
        device: 'cuda' nebo 'cpu'
    """

    def __init__(self, model_path='models/yolo/yolo11x.pt', device='cuda'):
        self.model = YOLO(model_path)
        self.device = device
        print(f"✓ YOLO Multi-Class Detector loaded: {model_path}")
        print(f"✓ Monitoring {len(SUSPICIOUS_CLASSES)} classes:")
        for class_id, name in SUSPICIOUS_CLASSES.items():
            print(f"    [{class_id:2d}] {name}")

    def detect(self, image, conf_threshold=0.25, class_ids=None):
        """
        Detekce objektů s filtrováním podle tříd

        Args:
            image: numpy array nebo cesta k obrázku
            conf_threshold: práh spolehlivosti (0-1)
            class_ids: list class ID k detekci (None = všechny SUSPICIOUS_CLASSES)

        Returns:
            detections: dict {
                'boxes': numpy array (N, 4) [x1, y1, x2, y2],
                'scores': numpy array (N,),
                'class_ids': numpy array (N,),
                'class_names': list of str (N,)
            }
        """
        # Default: detekuj všechny suspicious classes
        if class_ids is None:
            class_ids = list(SUSPICIOUS_CLASSES.keys())

        # YOLO detection
        results = self.model(image, conf=conf_threshold, device=self.device, verbose=False)

        # Parse results
        boxes = []
        scores = []
        cls_ids = []
        cls_names = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])

                # Filter by class
                if class_id not in class_ids:
                    continue

                # Get box coordinates and score
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0])

                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                cls_ids.append(class_id)
                cls_names.append(SUSPICIOUS_CLASSES.get(class_id, f'class_{class_id}'))

        return {
            'boxes': np.array(boxes) if boxes else np.array([]).reshape(0, 4),
            'scores': np.array(scores),
            'class_ids': np.array(cls_ids),
            'class_names': cls_names
        }

    def detect_persons_only(self, image, conf_threshold=0.25):
        """
        Detekce pouze osob (pro skeleton extraction)

        Returns:
            boxes: numpy array (N, 5) [x1, y1, x2, y2, conf] - kompatibilní s ViTPose
        """
        detections = self.detect(image, conf_threshold, class_ids=[PERSON_CLASS])

        if len(detections['boxes']) == 0:
            return np.array([])

        # Spojit boxes + scores do formátu (N, 5)
        boxes_with_conf = np.column_stack([
            detections['boxes'],
            detections['scores']
        ])

        return boxes_with_conf

    def detect_all_suspicious(self, image, conf_threshold=0.25):
        """
        Detekce všech podezřelých objektů (osoby + zbraně + podezřelé předměty)

        Returns:
            detections: dict s boxes, scores, class_ids, class_names
        """
        return self.detect(image, conf_threshold, class_ids=list(SUSPICIOUS_CLASSES.keys()))

    def detect_with_categories(self, image, conf_threshold=0.25):
        """
        Detekce s rozdělením do kategorií

        Returns:
            dict {
                'persons': detections pro osoby,
                'weapons': detections pro zbraně,
                'suspicious': detections pro podezřelé objekty
            }
        """
        # Detekuj všechno
        all_dets = self.detect_all_suspicious(image, conf_threshold)

        # Rozdělení do kategorií
        persons_mask = all_dets['class_ids'] == PERSON_CLASS
        weapons_mask = np.isin(all_dets['class_ids'], WEAPON_CLASSES)
        suspicious_mask = np.isin(all_dets['class_ids'], SUSPICIOUS_OBJECT_CLASSES)

        def filter_detections(mask):
            if mask.sum() == 0:
                return {
                    'boxes': np.array([]).reshape(0, 4),
                    'scores': np.array([]),
                    'class_ids': np.array([]),
                    'class_names': []
                }
            return {
                'boxes': all_dets['boxes'][mask],
                'scores': all_dets['scores'][mask],
                'class_ids': all_dets['class_ids'][mask],
                'class_names': [all_dets['class_names'][i] for i, m in enumerate(mask) if m]
            }

        return {
            'persons': filter_detections(persons_mask),
            'weapons': filter_detections(weapons_mask),
            'suspicious': filter_detections(suspicious_mask)
        }

    def visualize(self, image, detections, save_path=None, show=True):
        """
        Vizualizace detekcí s barevným kódováním

        Args:
            image: numpy array (BGR)
            detections: výstup z detect() nebo detect_with_categories()
            save_path: cesta pro uložení (None = neukládá)
            show: zobrazit obrázek
        """
        img_vis = image.copy()

        # Barvy pro kategorie
        colors = {
            0: (0, 255, 0),      # person - zelená
            34: (0, 0, 255),     # baseball bat - červená
            43: (0, 0, 255),     # knife - červená
            76: (0, 165, 255),   # scissors - oranžová
            24: (255, 255, 0),   # backpack - cyan
            26: (255, 255, 0),   # handbag - cyan
            28: (255, 255, 0),   # suitcase - cyan
            32: (255, 165, 0),   # sports ball - modrá
            35: (255, 200, 0),   # baseball glove - světle modrá
        }

        # Pokud je to dict s kategoriemi, rozbal to
        if isinstance(detections, dict) and 'persons' in detections:
            # Spojit všechny kategorie
            all_boxes = []
            all_scores = []
            all_class_ids = []
            all_class_names = []

            for category in ['persons', 'weapons', 'suspicious']:
                det = detections[category]
                if len(det['boxes']) > 0:
                    all_boxes.append(det['boxes'])
                    all_scores.append(det['scores'])
                    all_class_ids.append(det['class_ids'])
                    all_class_names.extend(det['class_names'])

            if all_boxes:
                detections = {
                    'boxes': np.vstack(all_boxes),
                    'scores': np.concatenate(all_scores),
                    'class_ids': np.concatenate(all_class_ids),
                    'class_names': all_class_names
                }
            else:
                detections = {
                    'boxes': np.array([]).reshape(0, 4),
                    'scores': np.array([]),
                    'class_ids': np.array([]),
                    'class_names': []
                }

        # Kresli detekce
        for box, score, class_id, class_name in zip(
            detections['boxes'],
            detections['scores'],
            detections['class_ids'],
            detections['class_names']
        ):
            x1, y1, x2, y2 = map(int, box)
            color = colors.get(class_id, (128, 128, 128))

            # Box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f'{class_name}: {score:.2f}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_vis, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img_vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if save_path:
            cv2.imwrite(save_path, img_vis)
            print(f"✓ Saved visualization: {save_path}")

        if show:
            cv2.imshow('YOLO Multi-Class Detection', img_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img_vis


# Quick test
if __name__ == "__main__":
    # Test na tvém obrázku s pálkami
    detector = YOLOMultiClassDetector('models/yolo/yolo11x.pt')

    test_image = 'frames_0.5_upscaled/004/004_0065.jpg'

    print(f"\n{'='*70}")
    print(f"TEST: {test_image}")
    print('='*70)

    # Načti obrázek
    img = cv2.imread(test_image)

    # Detekce s kategoriemi (NIŽŠÍ confidence pro detekci zbraní!)
    results = detector.detect_with_categories(img, conf_threshold=0.15)

    print(f"\n✓ VÝSLEDKY:")
    print(f"  Persons: {len(results['persons']['boxes'])}")
    print(f"  Weapons: {len(results['weapons']['boxes'])}")
    if len(results['weapons']['class_names']) > 0:
        print(f"    → {results['weapons']['class_names']}")
    print(f"  Suspicious objects: {len(results['suspicious']['boxes'])}")
    if len(results['suspicious']['class_names']) > 0:
        print(f"    → {results['suspicious']['class_names']}")

    # Vizualizace
    detector.visualize(img, results, save_path='test_multiclass_detection.jpg', show=False)

    print(f"\n✓ Vizualizace uložena: test_multiclass_detection.jpg")
