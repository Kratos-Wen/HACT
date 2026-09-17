#!/usr/bin/env bash
# Transfer to Reassembly B: the Reassembly A fold models are applied unchanged. Each B recording is scored
# by the fold model that excluded its participant; thresholds come from that fold's A validation participants.
# Run scripts/run_hact.sh (and scripts/run_frame_baseline.sh) on Reassembly A first.
# usage: NAME=hact_reassembly_a scripts/run_transfer_b.sh [SEED]
set -euo pipefail
SEED=${1:-0}; NAME=${NAME:-hact_reassembly_a}; VARIANT=${VARIANT:-hact}
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY=${PYTHON_BIN:-python}; BENCH=data/impact_public/reassembly_b_transfer; R=outputs/results
CK=outputs/$NAME/$VARIANT; PH=outputs/${NAME}_transfer_b/hact_phase; ST=outputs/${NAME}_transfer_b/hact_state; mkdir -p "$R"
$PY integrations/rescore_hact_transitions.py --checkpoint-root "$CK" --benchmark "$BENCH" --output-root "$PH" --seed "$SEED" --transfer-test
$PY integrations/apply_hact_two_state_filter.py --prediction-root "$PH" --benchmark "$BENCH" --output-root "$ST" --seed "$SEED"
$PY integrations/evaluate_fully_predicted.py hact --root "$ST" --benchmark "$BENCH" --timeline phase \
  --decision validation_f1 --seed "$SEED" --output "$R/${NAME}_transfer_b_seed${SEED}_frame.json"
for METHOD in causal_tcn mistsense; do
  SRC=outputs/frame_baselines/${METHOD}_reassembly_a_seed0; [[ -d $SRC ]] || continue
  OUT=outputs/frame_baselines_transfer_b/${METHOD}_seed0
  $PY integrations/score_frame_baselines_transfer.py --source-root "$SRC" --benchmark "$BENCH" --output-root "$OUT"
  $PY integrations/evaluate_fully_predicted.py dense --root "$OUT" --benchmark "$BENCH" --timeline event \
    --decision validation_f1 --seed 0 --output "$R/${METHOD}_transfer_b_seed0_frame.json"
done
