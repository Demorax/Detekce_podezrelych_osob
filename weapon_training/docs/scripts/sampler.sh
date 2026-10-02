#!/bin/bash
# Vzorkuje skutecne tempo treninku (iterace/cas) + pametove metriky do CSV.
out=~/weapon_training/rate_samples.csv
[ -f "$out" ] || echo "ts,iter,epoch,rate_s_it,gpu_mem_mib,avail_gb,swap_gb,si,so,load1" > "$out"
prev_it=""; prev_t=""
while pgrep -f train_weapon_yolo >/dev/null; do
  line=$(tr '\r' '\n' < ~/weapon_training/train.log | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E '[0-9]+/100' | tail -1)
  it=$(echo "$line" | grep -oE '[0-9]+/623' | tail -1 | cut -d/ -f1)
  ep=$(echo "$line" | grep -oE '^ *[0-9]+/100' | tr -d ' ' | cut -d/ -f1)
  now=$(date +%s)
  gm=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | head -1)
  read -r av sw <<<"$(free -g | awk '/^Mem:/{a=$7} /^Swap:/{s=$3} END{print a, s}')"
  read -r si so <<<"$(vmstat 1 2 | tail -1 | awk '{print $7, $8}')"
  l1=$(cut -d' ' -f1 /proc/loadavg)
  rate=""
  if [ -n "$prev_it" ] && [ -n "$it" ] && [ "$it" -gt "$prev_it" ]; then
    rate=$(awk -v a="$now" -v b="$prev_t" -v i="$it" -v j="$prev_it" 'BEGIN{printf "%.2f",(a-b)/(i-j)}')
  fi
  echo "$(date +%H:%M:%S),$it,$ep,$rate,$gm,$av,$sw,$si,$so,$l1" >> "$out"
  [ -n "$it" ] && { prev_it=$it; prev_t=$now; }
  sleep 60
done
echo "$(date +%H:%M:%S),TRENINK_SKONCIL,,,,,,,," >> "$out"
