#!/usr/bin/env bash
set -euo pipefail

# Debug EVA01 baseline plumbing without loading/downloading EVA01, ShapeLLM, or VQVAE.
# Override knobs, e.g.:
#   RUN_CONFIG=configs/runs/eva01-mock.yaml NUM_SAMPLES=5 KEEP_RATIOS=0.75,0.5 OUT_ROOT=../output/eva01-debug bash scripts/run_eva01_baselines_debug.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_CONFIG="${RUN_CONFIG:-configs/runs/eva01-mock.yaml}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
KEEP_RATIOS="${KEEP_RATIOS:-1.0,0.75,0.5,0.25,0.1}"
OUT_ROOT="${OUT_ROOT:-../output/eva01-baseline-debug}"

# Keep this off by default so local debugging never downloads SBERT/SimCSE weights.
export SHAPELLM_ENABLE_SEMANTIC_METRICS="${SHAPELLM_ENABLE_SEMANTIC_METRICS:-0}"

PRUNERS=(
  no_pruning
  random
  uniform
  divprune
  apet
  otprune
  tome
  fastv_mesh
  loco3d
  octree_merge
  runlength_curve
  reconot
  loco3d_dpp
  loco3d_nonempty_dpp
)

mkdir -p "$OUT_ROOT"
INPUT_DIRS=()

for PRUNER in "${PRUNERS[@]}"; do
  OUT_DIR="$OUT_ROOT/$PRUNER"
  rm -rf "$OUT_DIR"
  mkdir -p "$OUT_DIR"

  if [[ "$PRUNER" == "no_pruning" ]]; then
    RATIOS="1.0"
  else
    RATIOS="$KEEP_RATIOS"
  fi

  echo "==== EVA01 mock baseline: $PRUNER keep_ratios=$RATIOS ===="
  "$PYTHON_BIN" -u -m eval.run_eval \
    --config "$RUN_CONFIG" \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$OUT_DIR" \
    --run-log-file "$OUT_DIR/eval_run.log" \
    --pruners "$PRUNER" \
    --keep-ratios "$RATIOS"

  INPUT_DIRS+=("$OUT_DIR")
done

MERGED_DIR="$OUT_ROOT/merged"
rm -rf "$MERGED_DIR"
"$PYTHON_BIN" -u -m eval.merge_eval_results   --inputs "${INPUT_DIRS[@]}"   --output-dir "$MERGED_DIR"   --dedupe   --indent 2

echo "==== done ===="
echo "Merged summary: $MERGED_DIR/summary.csv"
echo "Merged results: $MERGED_DIR/results.json"
