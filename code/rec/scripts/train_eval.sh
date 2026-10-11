#!/bin/bash
# train + (T, beta) on validation + test once for configs/<name>.yaml.   scripts/train_eval.sh <name> [seed]
# score_norm: mean configs: sum vs mean (beta = 0) is chosen on validation by scripts/score_norm_select.py before the test run.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
n=$1; s=${2:-2023}; r=${n}_r$s; PY=${REC_PY:-.venv/bin/python}
while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt 14000 ]; do sleep 60; done
if [ ! -f runs/$r/ckpt/best.pth ] || [ "${RETRAIN:-1}" = 1 ]; then
$PY scripts/run_adarq.py --config=configs/$n.yaml --rand_seed=$s --ckpt_dir=runs/$r/ckpt/ --log_dir=runs/$r/logs/ --tensorboard_log_dir=runs/$r/tb/ --results_path=results/$r.json > logs/$r.log 2>&1 || { echo "[$(date '+%F %T')] $n TRAIN FAILED" >> results/done.log; exit 1; }
fi
while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -lt 14000 ]; do sleep 60; done
EV="$PY scripts/cuit_alpha.py --config=configs/$n.yaml --rand_seed=$s --eval_only=runs/$r/ckpt/best.pth"
if grep -q '^score_norm: mean' configs/$n.yaml; then
  for nm in mean sum; do
    $EV --betas=0 --score_norm=$nm --val_only=True --results_path=results/${r}_val_$nm.json > logs/${r}_val_$nm.log 2>&1 || { echo "[$(date '+%F %T')] $n VAL($nm) FAILED" >> results/done.log; exit 1; }
  done
  NORM=$($PY scripts/score_norm_select.py results/${r}_val_mean.json results/${r}_val_sum.json results/${r}_val_mean_items.npz results/${r}_norm.json | tail -1)
  echo "[$(date '+%F %T')] $n score norm: $(tr -d '\n ' < results/${r}_norm.json)" >> results/done.log
  $EV --betas=0 --score_norm=$NORM --buckets=True --results_path=results/${r}_final.json > logs/${r}_final.log 2>&1 \
    && echo "[$(date '+%F %T')] $n DONE" >> results/done.log || echo "[$(date '+%F %T')] $n EVAL FAILED" >> results/done.log
else
  $EV --buckets=True --results_path=results/${r}_final.json > logs/${r}_final.log 2>&1 \
    && echo "[$(date '+%F %T')] $n DONE" >> results/done.log || echo "[$(date '+%F %T')] $n EVAL FAILED" >> results/done.log
fi
