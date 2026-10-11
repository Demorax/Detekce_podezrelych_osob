#!/bin/bash
# Spark: obnova prostředí + fine-tune v3 (v2 + těžká negativa z videí bez zbraní).
# Prostředí bylo 2026-10-02 na žádost smazáno, proto se instaluje znovu.
# Hyperparametry jsou IDENTICKÉ s během med_domain_adapt_frozen(_neg), mění se jen data
# (+41 těžkých negativ: framy, kde v2 falešně hlásí tyč; + rozhodnuté sporné případy, pokud jsou).
set -u
cd ~/weapon_training
log=v3.log; exec >>"$log" 2>&1
echo "[$(date +%H:%M:%S)] start"
if [ ! -x ~/yolo-train-venv/bin/python ]; then
  python3 -m venv ~/yolo-train-venv
  ~/yolo-train-venv/bin/pip install -q --upgrade pip
  # GB10 = sm_121 -> nutný build pro CUDA 13 (viz TRAINING_REPORT §2.1)
  ~/yolo-train-venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu130
  ~/yolo-train-venv/bin/pip install -q ultralytics
fi
~/yolo-train-venv/bin/python -c "import torch;assert torch.cuda.is_available();print('torch',torch.__version__,torch.cuda.get_device_name(0))" || { echo "CUDA NEFUNGUJE"; exit 1; }
# data.yaml má absolutní cestu z Macu -> přepsat na Spark
sed -i "s|^path:.*|path: $HOME/weapon_training/datasets/med_finetune_v3|" datasets/med_finetune_v3/data.yaml
grep "^path:" datasets/med_finetune_v3/data.yaml
echo "[$(date +%H:%M:%S)] trenink v3"
# nice 19 = nejnižší priorita CPU, produkční služby na stroji mají přednost
nice -n 19 ~/yolo-train-venv/bin/python train_weapon_yolo.py --data datasets/med_finetune_v3/data.yaml \
  --model base_best.pt --imgsz 960 --epochs 30 --batch 16 --workers 2 --cache ram \
  --optimizer SGD --lr0 0.0005 --lrf 0.1 --warmup-epochs 1 \
  --patience 0 --freeze 11 --name med_domain_adapt_frozen_neg_v3
echo "[$(date +%H:%M:%S)] exit=$? HOTOVO"
