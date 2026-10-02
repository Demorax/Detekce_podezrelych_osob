#!/bin/bash
cd ~/weapon_training
log=chain.log
echo "[$(date +%H:%M:%S)] cekam na dokonceni oversamplingu" >> $log
while pgrep -f oversample_knife >/dev/null; do sleep 10; done
D=datasets/weapons_knife_boost
ok=1
for s in train val test; do
  i=$(ls $D/$s/images 2>/dev/null | wc -l); l=$(ls $D/$s/labels 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] $s: $i img / $l lbl" >> $log
  [ "$i" != "$l" ] && ok=0
  [ "$i" -lt 100 ] && ok=0
done
if [ "$ok" != "1" ]; then
  echo "[$(date +%H:%M:%S)] KONTROLA SELHALA - fine-tune se nespusti" >> $log
  exit 1
fi
echo "[$(date +%H:%M:%S)] kontrola OK, spoustim fine-tune z best.pt" >> $log
BEST=$(find runs -path "*sgd_lr005*" -name best.pt | head -1)
echo "[$(date +%H:%M:%S)] vahy: $BEST" >> $log
~/yolo-train-venv/bin/python train_weapon_yolo.py \
  --data $D/data.yaml --model "$BEST" \
  --imgsz 960 --epochs 12 --batch 32 --workers 6 \
  --optimizer SGD --lr0 0.001 --lrf 0.05 --warmup-epochs 1.0 \
  --patience 6 --name knife_boost_ft >> finetune.log 2>&1
echo "[$(date +%H:%M:%S)] fine-tune ukoncen, exit=$?" >> $log
