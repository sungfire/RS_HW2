#!/bin/bash
# Wait until one GPU has had no compute processes for two consecutive checks (60s apart),
# then run the NeuMF validation search and the full-data refit on that GPU.
cd "$(dirname "$0")"
LOG=results/logs/q2_neumf.log
mkdir -p results/logs

free_gpu() {
  for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$i")
    if ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"; then
      echo "$i"; return 0
    fi
  done
  return 1
}

echo "$(date '+%F %T') waiting for a free GPU" | tee -a "$LOG"
prev=""
while true; do
  g=$(free_gpu)
  if [ -n "$g" ] && [ "$g" = "$prev" ]; then break; fi
  prev="$g"
  sleep 60
done
echo "$(date '+%F %T') GPU $g free, starting" | tee -a "$LOG"

export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8
.venv/bin/python -u src/q2_neumf.py search --device cuda:0 >> "$LOG" 2>&1 &&
.venv/bin/python -u src/q2_neumf.py refit --device cuda:0 >> "$LOG" 2>&1
status=$?
echo "$(date '+%F %T') finished with status $status" | tee -a "$LOG"
exit $status
