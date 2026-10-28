"""
Anotace po snímcích pro UMN crowd dataset (Crowd-Activity-All.avi, 7 739 snímků, 320×240, 30 FPS).

Dataset nemá oficiální soubor s GT; abnormální úseky jsou ale vypálené přímo do obrazu červeným
nápisem „Abnormal Crowd Activity“ vlevo nahoře. Label se tedy odvodí z podílu sytě červených
pixelů v oblasti nápisu. Hranice klipů: každý z 11 klipů končí abnormálním úsekem, scény (3) ze změny obrazu.

Výstup: data/UMN/umn_labels.npz (label (N,) 0/1, clip (N,) index klipu, scene (N,) index scény)
a tisk tabulky klipů pro kontrolu s literaturou (scény 1 453 / 4 144 / 2 144 snímků).

Použití:
    python umn_labels.py
"""
import cv2
import numpy as np

VIDEO = 'data/UMN/Crowd-Activity-All.avi'
OUT = 'data/UMN/umn_labels.npz'
ROI = (slice(5, 30), slice(5, 165))        # oblast nápisu (y, x)


def main():
    cap = cv2.VideoCapture(VIDEO)
    labels, diffs, prev = [], [], None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        roi = f[ROI]
        b, g, r = roi[..., 0].astype(int), roi[..., 1].astype(int), roi[..., 2].astype(int)
        red = ((r > 150) & (g < 90) & (b < 90)).mean()
        labels.append(red > 0.02)
        small = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (40, 30)).astype(np.float32)
        diffs.append(0.0 if prev is None else float(np.abs(small - prev).mean()))
        prev = small
    labels, diffs = np.array(labels, dtype=np.int64), np.array(diffs)
    # hranice klipů: každý z 11 klipů končí abnormálním úsekem (nápis zhasne = nový klip);
    # střihy v obraze samotné najdou jen 3 změny prostředí (scény), ne klipy uvnitř scény
    ends = np.where(np.diff(np.r_[labels, 0]) == -1)[0] + 1
    # střih scény (změna prostředí) je přesnější hranice než konec nápisu, který zhasne
    # o pár snímků dřív; střihy scén proto mají přednost a blízké konce nápisu se k nim přitáhnou
    scene_cuts = np.where(diffs > 10 * np.median(diffs[1:]) + 15)[0]
    cuts = []
    for e in ends:
        near = scene_cuts[np.abs(scene_cuts - e) < 30]
        c = int(near[0]) if len(near) else int(e)
        if c < len(labels) - 30 and c not in cuts:      # bez prázdného ocásku na konci videa
            cuts.append(c)
    cuts = np.array(sorted(cuts))
    clip = np.zeros(len(labels), dtype=np.int64)
    for c in cuts:
        clip[c:] += 1
    scene = np.zeros(len(labels), dtype=np.int64)
    for c in scene_cuts:
        scene[c:] += 1
    starts = [0] + cuts.tolist()
    scene_of_clip = [int(scene[s]) for s in starts]
    centers = range(len(scene_cuts) + 1)
    np.savez(OUT, label=labels, clip=clip, scene=scene)
    print(f'{len(labels)} snímků, {len(starts)} klipů, {len(centers)} scén, abnormálních {labels.sum()} '
          f'({labels.mean():.1%})')
    print('klip  scéna  start   délka  abnorm. od  abnorm. snímků')
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(labels)
        ab = np.where(labels[s:e])[0]
        print(f'{k:>4} {scene_of_clip[k]:>6} {s:>6} {e - s:>7} {s + ab[0] if len(ab) else -1:>11} {len(ab):>15}')
    for sc in range(len(centers)):
        print(f'scéna {sc}: {(scene == sc).sum()} snímků')


if __name__ == '__main__':
    main()
