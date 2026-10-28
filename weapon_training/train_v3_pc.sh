#!/bin/bash
# Windows PC (RTX 3090, Git Bash): zápis sporných případů -> sestavení v3 -> fine-tune v3 -> evaluace.
# Hyperparametry jsou IDENTICKÉ se spark_setup_and_train_v3.sh (a tedy s během v2), mění se jen data.
# Vynecháno jen to, co bylo kvůli sdílenému Sparku: nice, --workers 2, --cache ram.
#
# Předpoklad: vyplněný review_ambiguous/rozhodnuti.csv (T / N / ?).
# Spuštění z weapon_training/:   bash train_v3_pc.sh            (celý řetězec)
#                                bash train_v3_pc.sh eval       (jen evaluace hotového v3)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1          # JSONy z Macu jsou UTF-8, Windows by bez toho četl cp1250
NAME=med_domain_adapt_frozen_neg_v3
V3=runs/detect/runs/weapons/$NAME/weights/best.pt
log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ "${1:-}" != "eval" ]; then
  log "zápis rozhodnutí sporných případů"
  python review_ambiguous.py apply
  log "sestavení datasetu v3"
  python build_finetune_dataset.py --out datasets/med_finetune_v3 \
      --neg-dirs annotate_neg_train annotate_neg_hard \
      --exclude-file annotation_results/exclusions_ambiguous.txt
  [ -f base_best.pt ] || cp ../models/weapon_detection_yolo11l/best.pt base_best.pt
  log "trénink v3"
  python train_weapon_yolo.py --data datasets/med_finetune_v3/data.yaml \
    --model base_best.pt --imgsz 960 --epochs 30 --batch 16 --workers 4 \
    --optimizer SGD --lr0 0.0005 --lrf 0.1 --warmup-epochs 1 \
    --patience 0 --freeze 11 --name $NAME
fi

log "evaluace v3 (v2 se přeměřuje na stejném stroji, AP50 se mezi ultralytics 8.3 a 8.4 liší o ~0,02)"
python eval_test005.py --add v3=$V3 | tee v3_eval_test005.txt
python calibrate_melee_threshold.py --weights $V3 | tee v3_calibrate_melee.txt
python eval_temporal_filter.py --weights $V3 --cache temporal_eval_v3 | tee v3_eval_temporal.txt
python eval_armed_pipeline.py --weights $V3 --cache armed_eval_v3 | tee v3_eval_armed.txt
log "HOTOVO. Nasadit jen když je v3 lepší na testu 005 i na negativech (POKRACOVANI.md §2.4)."
