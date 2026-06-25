#!/usr/bin/env bash
set -euo pipefail

# Debug EVA01 baseline plumbing without loading/downloading EVA01, ShapeLLM, or VQVAE.
# Override knobs, e.g.:
#   NUM_SAMPLES=5 KEEP_RATIOS=0.75,0.5 OUT_ROOT=artifacts/eva01-debug bash scripts/run_eva01_baselines_debug.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_CSV="${DATA_CSV:-data/metadata.csv}"
GLB_DIR="${GLB_DIR:-data}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
KEEP_RATIOS="${KEEP_RATIOS:-1.0,0.75,0.5,0.25,0.1}"
OUT_ROOT="${OUT_ROOT:-artifacts/eva01-baseline-debug}"

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
  "$PYTHON_BIN" -u -m eval.run_eval     --model-backend eva01     --mock-model     --device cpu     --data-csv "$DATA_CSV"     --glb-dir "$GLB_DIR"     --num-samples "$NUM_SAMPLES"     --output-dir "$OUT_DIR"     --pruners "$PRUNER"     --keep-ratios "$RATIOS"     --mesh-prefetch-workers 0     2>&1 | tee "$OUT_DIR/eval_run.log"

  INPUT_DIRS+=("$OUT_DIR")
done

MERGED_DIR="$OUT_ROOT/merged"
rm -rf "$MERGED_DIR"
"$PYTHON_BIN" -u -m eval.merge_eval_results   --inputs "${INPUT_DIRS[@]}"   --output-dir "$MERGED_DIR"   --dedupe   --indent 2

echo "==== done ===="
echo "Merged summary: $MERGED_DIR/summary.csv"
echo "Merged results: $MERGED_DIR/results.json"
