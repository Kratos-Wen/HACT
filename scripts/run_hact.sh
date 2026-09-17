#!/usr/bin/env bash
# Train and evaluate one HACT configuration on IMPACT-ego Reassembly A:
# five participant-disjoint folds -> rescoring on predicted events -> two-state filter ->
# fully predicted evaluation -> recovery false-positive rate at the validation-selected operating point.
# usage: NAME=hact_reassembly_a GPUS="0 1 2 3" scripts/run_hact.sh [SEED]
set -euo pipefail
SEED=${1:-0}; NAME=${NAME:-hact_reassembly_a}; GPUS=${GPUS:-"0"}; VARIANT=${VARIANT:-hact}
BENCH=${BENCH:-data/impact_public/reassembly_a}
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=${PYTHON_BIN:-python}; CONFIG=configs/$NAME.json
CK=outputs/$NAME/$VARIANT; PH=outputs/$NAME/hact_phase; ST=outputs/$NAME/hact_state
R=outputs/results; L=outputs/$NAME/logs; mkdir -p "$L" "$R"
read -ra G <<< "$GPUS"; NG=${#G[@]}
train() { local fold=$1 gpu=$2; [[ -s $CK/seed_$SEED/fold_$fold/metrics.json ]] && return 0
  CUDA_VISIBLE_DEVICES=$gpu $PY pipeline/step2_bimanual_transition.py run --config "$CONFIG" \
    --variant "$VARIANT" --fold "$fold" --seed "$SEED" --device cuda > "$L/train_seed${SEED}_fold$fold.log" 2>&1; }
pids=(); i=0
for fold in 0 1 2 3 4; do
  train $fold "${G[$((i % NG))]}" & pids+=($!); i=$((i+1))
  if (( i % NG == 0 )); then wait "${pids[@]}"; pids=(); fi
done
[[ ${#pids[@]} -gt 0 ]] && wait "${pids[@]}"
CUDA_VISIBLE_DEVICES=${G[0]} $PY integrations/rescore_hact_transitions.py --checkpoint-root "$CK" \
  --benchmark "$BENCH" --output-root "$PH" --seed "$SEED" --device cuda > "$L/rescore_seed$SEED.log" 2>&1
$PY integrations/apply_hact_two_state_filter.py --prediction-root "$PH" --benchmark "$BENCH" \
  --output-root "$ST" --seed "$SEED" > "$L/filter_seed$SEED.log" 2>&1
$PY integrations/evaluate_fully_predicted.py hact --root "$ST" --benchmark "$BENCH" --timeline phase \
  --decision validation_f1 --seed "$SEED" --output "$R/${NAME}_seed${SEED}_frame.json" > "$L/eval_seed$SEED.log" 2>&1
$PY integrations/recovery_at_validation_recall.py hact --root "$ST" --benchmark "$BENCH" --timeline phase \
  --seed "$SEED" --output "$R/${NAME}_seed${SEED}_recovery.json" > "$L/recovery_seed$SEED.log" 2>&1
$PY -c "import json;d=json.load(open('$R/${NAME}_seed${SEED}_frame.json'))['pooled'];print('$NAME seed $SEED: AUPRC %.3f F1 %.3f'%(d['anomaly_auprc'],d['anomaly_f1']))"
