#!/usr/bin/env bash
# Supervised dense baselines on the shared features and folds.
# usage: GPUS="0 1 2 3" scripts/run_frame_baseline.sh {causal_tcn|mistsense} [SEED]
set -euo pipefail
METHOD=$1; SEED=${2:-0}; GPUS=${GPUS:-"0"}; BENCH=${BENCH:-data/impact_public/reassembly_a}
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY=${PYTHON_BIN:-python}; OUT=outputs/frame_baselines/${METHOD}_reassembly_a_seed$SEED; R=outputs/results; L=$OUT/logs
mkdir -p "$L" "$R"; read -ra G <<< "$GPUS"; NG=${#G[@]}
train() { local fold=$1 gpu=$2; [[ -s $OUT/fold_$fold/meta.json ]] && return 0
  CUDA_VISIBLE_DEVICES=$gpu $PY integrations/train_frame_baselines.py --benchmark "$BENCH" --method "$METHOD" \
    --fold "$fold" --seed "$SEED" --output-root "$OUT" --device cuda > "$L/train_fold$fold.log" 2>&1; }
pids=(); i=0
for fold in 0 1 2 3 4; do
  train $fold "${G[$((i % NG))]}" & pids+=($!); i=$((i+1))
  if (( i % NG == 0 )); then wait "${pids[@]}"; pids=(); fi
done
[[ ${#pids[@]} -gt 0 ]] && wait "${pids[@]}"
$PY integrations/evaluate_fully_predicted.py dense --root "$OUT" --benchmark "$BENCH" --timeline event \
  --decision validation_f1 --seed "$SEED" --output "$R/${METHOD}_seed${SEED}_frame.json" > "$L/eval.log" 2>&1
$PY integrations/recovery_at_validation_recall.py dense --root "$OUT" --benchmark "$BENCH" --timeline event \
  --seed "$SEED" --output "$R/${METHOD}_seed${SEED}_recovery.json" > "$L/recovery.log" 2>&1
$PY -c "import json;d=json.load(open('$R/${METHOD}_seed${SEED}_frame.json'))['pooled'];print('$METHOD seed $SEED: AUPRC %.3f F1 %.3f'%(d['anomaly_auprc'],d['anomaly_f1']))"
