"""
Stažení videí RWF-2000 (Hugging Face mirror DanJoshua/RWF-2000, viz data/RWF-2000/README.md) s rozbalením
za běhu: archiv (12,3 GB) se neukládá na disk, ze streamu se rovnou zapisují .avi do data/RWF-2000/videos/.

Názvy s ne-ASCII znaky nebo delší než 100 znaků (Windows MAX_PATH) se nahradí
sanitized_<md5(název)[:10]>_<pořadí v archivu>.avi; mapování na původní názvy se uloží do
data/RWF-2000/videos/sanitized_names.json.

Použití:
    python -I download_rwf.py
"""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

URL = 'https://huggingface.co/datasets/DanJoshua/RWF-2000/resolve/main/RWF-2000.tar.gz'
OUT = Path('data/RWF-2000/videos')


def safe_name(name, idx):
    stem = Path(name).name
    if stem.isascii() and len(stem) <= 100:
        return stem
    return f'sanitized_{hashlib.md5(stem.encode("utf-8")).hexdigest()[:10]}_{idx}{Path(stem).suffix}'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mapping, n = {}, 0
    with urllib.request.urlopen(URL) as resp, tarfile.open(fileobj=resp, mode='r|gz') as tar:
        for idx, m in enumerate(tar):
            parts = Path(m.name).parts
            if not m.isfile() or not m.name.lower().endswith('.avi') or len(parts) < 3:
                continue
            split, cls = parts[-3], parts[-2]
            if split not in ('train', 'val') or cls not in ('Fight', 'NonFight'):
                continue
            name = safe_name(m.name, idx)
            if name != Path(m.name).name:
                mapping[f'{split}/{cls}/{name}'] = {'original': Path(m.name).name, 'tar_index': idx}
            dst = OUT / split / cls / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, open(dst, 'wb') as f:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
            n += 1
            if n % 100 == 0:
                print(f'{n} videí', flush=True)
    with open(OUT / 'sanitized_names.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=1, ensure_ascii=False)
    print(f'hotovo: {n} videí, {len(mapping)} přejmenovaných')


if __name__ == '__main__':
    main()
