#!/bin/bash
# Domain adaptation: výchozí vyhodnocení -> fine-tune -> vyhodnocení.
# Spouští se odpojeně (setsid nohup), výsledky jdou do domain_adapt.log.
set -u
cd ~/weapon_training
PY=~/yolo-train-venv/bin/python
BASE=runs/detect/runs/weapons/weapons_sgd_lr005/weights/best.pt
MED=datasets/med_finetune/data.yaml
ORIG=datasets/weapons_merged/data.yaml

evaluate () {   # $1 = váhy, $2 = popisek
  $PY - "$1" "$2" <<'PYEOF'
import sys
from ultralytics import YOLO
w, tag = sys.argv[1], sys.argv[2]
m = YOLO(w)
for name, data in (("MED val", "datasets/med_finetune/data.yaml"),
                   ("puvodni val", "datasets/weapons_merged/data.yaml")):
    r = m.val(data=data, split="val", imgsz=960, batch=16, device="0",
              verbose=False, plots=False, workers=4)
    cls = {r.names[int(c)]: (r.box.p[i], r.box.r[i], r.box.ap50[i])
           for i, c in enumerate(r.box.ap_class_index)}
    per = "  ".join(f"{k}: P={v[0]:.3f} R={v[1]:.3f} AP50={v[2]:.3f}" for k, v in cls.items())
    print(f"RESULT [{tag}] {name}: mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f} | {per}", flush=True)
PYEOF
}

if [ -z "${SKIP_BASE:-}" ]; then
  echo "[$(date +%H:%M:%S)] === vychozi model ==="
  evaluate "$BASE" "vychozi"
fi

echo "[$(date +%H:%M:%S)] === fine-tune ==="
$PY train_weapon_yolo.py --data "$MED" --model "$BASE" \
  --imgsz 960 --epochs 30 --batch 16 --workers 4 \
  --optimizer SGD --lr0 0.0005 --lrf 0.1 --warmup-epochs 1 \
  --patience 0 --freeze ${FREEZE:-0} --name ${RUN:-med_domain_adapt} > ${RUN:-med_domain_adapt}_train.log 2>&1
echo "[$(date +%H:%M:%S)] fine-tune exit=$?"

FT=runs/detect/runs/weapons/${RUN:-med_domain_adapt}/weights/last.pt
echo "[$(date +%H:%M:%S)] === po fine-tune ($FT) ==="
evaluate "$FT" "fine-tune"
echo "[$(date +%H:%M:%S)] HOTOVO"
