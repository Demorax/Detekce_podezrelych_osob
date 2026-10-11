#!/usr/bin/env python3
"""Hustší extrakce framů z okna, kde se ve videu vyskytují zbraně.

Proč: při 2 FPS má video 004 jen 82 framů a zbraně jsou v ~10 z nich, což dalo
16 anotovaných boxů — na fine-tune málo. Bitka ale trvá jen několik sekund,
takže hustší vzorkování právě tohoto okna zvýší počet pozitivních příkladů,
aniž by bylo nutné anotovat další klidné scény.

DŮLEŽITÉ — shoda s existující pipeline:
  * Zdrojový frame N*14 odpovídá existujícímu framu 2 FPS číslo N
    (ověřeno porovnáním pixelů: MAE ~1.1, tj. jen rozdíl JPEG komprese).
  * Existující `frames_0.5_upscaled` NEJSOU ESPCN super-resolution, jak uvádí
    README, ale prosté BIKUBICKÉ zvětšení 4× (ověřeno: MAE proti bikubice
    0,25–0,55, což je úroveň šumu JPEG). Nové framy se proto zvětšují stejně,
    jinak by model dostal dva různé typy obrazu.

Framy, které už existují v 2 FPS sadě (násobky 14), se přeskakují.

Použití:
    python extract_dense.py --video ../data/Motion_Emotion/004.mp4 \\
        --from-frame 868 --to-frame 1092 --step 5 --out ../frames_004_dense
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

EXISTING_STEP = 14   # krok původní extrakce 2 FPS při 29.97 fps
UPSCALE = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--from-frame", type=int, required=True)
    ap.add_argument("--to-frame", type=int, required=True)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="004d")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="nepřeskakovat násobky 14 — pro videa, ze kterých upscalovaná "
                         "2 FPS sada neexistuje (vše kromě 001, 002, 004)")
    ap.set_defaults(skip_existing=True)
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video nenalezeno: {video}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg není v PATH")

    wanted = list(range(args.from_frame, args.to_frame + 1, args.step))
    skipped = []
    if args.skip_existing:
        keep = []
        for n in wanted:
            (skipped if n % EXISTING_STEP == 0 else keep).append(n)
        wanted = keep

    if not wanted:
        raise SystemExit("Po odfiltrování nezbyl žádný frame")

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    expr = "+".join(f"eq(n\\,{n})" for n in wanted)
    with tempfile.TemporaryDirectory() as td:
        cmd = ["ffmpeg", "-loglevel", "error", "-i", str(video),
               "-vf", f"select='{expr}'", "-vsync", "0", "-q:v", "2",
               str(Path(td) / "f%05d.jpg"), "-y"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[:800], file=sys.stderr)
            raise SystemExit("ffmpeg selhal")

        got = sorted(Path(td).glob("f*.jpg"))
        if len(got) != len(wanted):
            print(f"[!] ffmpeg vrátil {len(got)} framů, čekal jsem {len(wanted)} — "
                  f"mapování na zdrojové indexy nemusí sedět", file=sys.stderr)
            if len(got) < len(wanted):
                wanted = wanted[: len(got)]

        n_written = 0
        for src_idx, fp in zip(wanted, got):
            im = Image.open(fp).convert("RGB")
            big = im.resize((im.width * UPSCALE, im.height * UPSCALE), Image.BICUBIC)
            dst = out / f"{args.prefix}_{src_idx:05d}.jpg"
            big.save(dst, quality=95)
            n_written += 1

    print(f"=== hotovo ===")
    print(f"okno zdrojových framů: {args.from_frame}–{args.to_frame}, krok {args.step}")
    print(f"přeskočeno (už v 2 FPS sadě): {len(skipped)}  {skipped}")
    print(f"zapsáno: {n_written} framů @ {UPSCALE}× bikubicky -> {out}")
    ex = next(out.glob("*.jpg"), None)
    if ex:
        with Image.open(ex) as im:
            print(f"rozlišení: {im.width}x{im.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
