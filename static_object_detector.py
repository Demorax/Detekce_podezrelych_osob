"""
Detekce nově odložených nehybných předmětů z modelu pozadí (bod 3 plánu praktické části).

ABODA ukázala, že COCO detektor (YOLO11x) zavazadlo pozná v ruce majitele, ale odložené ležící
samostatně často ne. Klasický přístup k opuštěným předmětům (Porikli et al. 2008, „dual background“)
na třídě objektu nezávisí: udržují se dvě pozadí s různou rychlostí učení.
  - dlouhodobé pozadí B_L se učí pomalu (α_L), krátkodobé B_S rychle (α_S),
  - pixel, který se liší od B_L, ale B_S ho už „pohltilo“ (od B_S se neliší), patří předmětu,
    který se v posledních sekundách objevil a od té doby se nehýbe,
  - důkaz se akumuluje po snímcích; oblast se souvislým důkazem delším než min_static_sec
    je kandidát na statický objekt,
  - oblasti překryté osobou se nezapočítávají (stojící člověk by jinak byl „statický předmět“)
    a dlouhodobé pozadí se pod osobami neučí.
Výstup má stejný formát jako objektové detekce pipeline: list (box, score, 'static_object'),
takže se zapojí do ObjectTrackeru a pravidel opuštění beze změny.

Použití jako modul:
    sod = StaticObjectDetector(frame_shape)
    dets = sod.update(frame_bgr, person_boxes)   # volat se stejnou frekvencí jako zbytek pipeline
"""
import cv2
import numpy as np


class StaticObjectDetector:
    def __init__(self, frame_shape, fps=5.0, alpha_long=None, alpha_short=None,
                 diff_thr=30, min_static_sec=2.0, min_area_frac=0.0005, max_area_frac=0.05,
                 person_pad=0.3, scale=1.0, use_gradient=True, sat_thr=230, warmup_sec=8.0):
        h, w = frame_shape[:2]
        self.scale = scale                     # zmenšení pro rychlost (souřadnice výstupu v originále)
        self.h, self.w = int(h * scale), int(w * scale)
        self.fps = fps
        # časové konstanty v sekundách: krátkodobé pozadí pohltí nehybný předmět za ~2 s,
        # dlouhodobé ho drží ~60 s
        self.a_l = alpha_long if alpha_long is not None else 1.0 / (60 * fps)
        self.a_s = alpha_short if alpha_short is not None else 1.0 / (1.5 * fps)
        self.diff_thr = diff_thr
        self.need = int(min_static_sec * fps)
        self.min_area = min_area_frac * self.h * self.w
        self.max_area = max_area_frac * self.h * self.w
        self.person_pad = person_pad
        # Gradientní reprezentace: odložený předmět přidá do obrazu hrany, pomalá změna osvětlení
        # (slunce, automatická expozice) mění hlavně jas. Na CAVIAR jasová varianta hlásila jako
        # „předmět“ světlou skvrnu od okna ve všech scénářích.
        self.use_gradient = use_gradient
        self.sat_thr = sat_thr
        self.warmup = int(warmup_sec * fps)
        self.bl = self.bs = None
        self.evidence = np.zeros((self.h, self.w), np.int32)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.n = 0

    def _person_mask(self, person_boxes):
        m = np.zeros((self.h, self.w), np.uint8)
        for b in (person_boxes if person_boxes is not None else []):
            x1, y1, x2, y2 = [v * self.scale for v in b[:4]]
            pw, ph = (x2 - x1) * self.person_pad, (y2 - y1) * self.person_pad
            cv2.rectangle(m, (int(x1 - pw), int(y1 - ph)), (int(x2 + pw), int(y2 + ph)), 1, -1)
        return m.astype(bool)

    def update(self, frame, person_boxes=None):
        img = frame if self.scale == 1.0 else cv2.resize(frame, (self.w, self.h))
        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        if self.use_gradient:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            g = cv2.GaussianBlur(cv2.magnitude(gx, gy), (5, 5), 0)
        else:
            g = gray
        self.n += 1
        if self.bl is None:
            self.bl, self.bs = g.copy(), g.copy()
            return []
        pmask = self._person_mask(person_boxes) | (gray > self.sat_thr)   # přepálené oblasti ignorovat
        if self.use_gradient:
            # jen NÁRŮST hran proti dlouhodobému pozadí: odložený předmět hrany přidá, kdežto místo,
            # odkud odešla osoba zachycená v dlouhodobém pozadí („duch“), hrany ztrácí
            fg_l = (g - self.bl) > self.diff_thr
        else:
            fg_l = np.abs(g - self.bl) > self.diff_thr
        fg_s = np.abs(g - self.bs) > self.diff_thr
        static = fg_l & ~fg_s & ~pmask
        self.evidence = np.where(static, self.evidence + 1, np.maximum(self.evidence - 2, 0))
        # učení: krátkodobé všude, dlouhodobé ne pod osobami a ne v kandidátech (aby předmět „nezapomnělo“);
        # prvních warmup_sec se dlouhodobé pozadí učí rychle, aby nezůstaly osoby z úvodu záznamu
        self.bs += self.a_s * (g - self.bs)
        a_l = self.a_s * 0.5 if self.n < self.warmup else self.a_l
        learn_l = ~pmask & (self.evidence == 0)
        self.bl[learn_l] += a_l * (g[learn_l] - self.bl[learn_l])
        if self.n < self.warmup:                 # zahřátí pozadí
            self.evidence[:] = 0
            return []
        mask = (self.evidence >= self.need).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if not (self.min_area <= area <= self.max_area):
                continue
            if bw > 6 * bh or bh > 6 * bw:       # tenké pruhy = hrany stínů, ne předmět
                continue
            score = float(min(1.0, self.evidence[lab == i].mean() / (3 * self.need)))
            box = np.array([x, y, x + bw, y + bh], np.float32) / self.scale
            out.append((box, score, 'static_object'))
        return out
