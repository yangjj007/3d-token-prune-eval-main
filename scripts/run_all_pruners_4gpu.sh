#!/usr/bin/env bash
# Run all registered token pruners (including no_pruning baseline) on 4 GPUs in parallel,
# then merge partial results into one directory.
#
# Requirements: bash, Linux/WSL2 + NVIDIA driver; run from repo root ShapeLLM-Omni-main.
# VLM loads in FP16 via --vlm-torch-dtype float16 (omit that flag to use auto bf16/fp16).
#
# Usage:
#   chmod +x scripts/run_all_pruners_4gpu.sh
#   ./scripts/run_all_pruners_4gpu.sh
# Or override env:
#   DATA_CSV=data/metadata.csv GLB_DIR=data OUT_DIR=eval_results_fp16 ./scripts/run_all_pruners_4gpu.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"
export SHAPELLM_EVAL_LIGHT="${SHAPELLM_EVAL_LIGHT:-1}"

# Hugging Face weights cache: use persistent disk under /yangjunjie (override with HF_HOME).
if [[ -d "/yangjunjie" ]]; then
  export HF_HOME="${HF_HOME:-/yangjunjie/huggingface}"
fi

# --- User-tunable defaults ---
DATA_CSV="${DATA_CSV:-data/metadata.csv}"
GLB_DIR="${GLB_DIR:-data}"
OUT_DIR="${OUT_DIR:-eval_results_4gpu_fp16}"
EVAL_CONFIG_DIR="${EVAL_CONFIG_DIR:-configs/eval}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
# Comma-separated, must match README / registered pruners (baseline + proposed)
KEEP_RATIOS="${KEEP_RATIOS:-1.0,0.75,0.5,0.25,0.1}"
SEED="${SEED:-42}"
# Use TEMPERATURE=0 for deterministic diagnostic runs. Historical full runs used 0.7.
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.7}"
VLM_TORCH_DTYPE="${VLM_TORCH_DTYPE:-float16}"

# Physical GPU ids used in order (0..3). Override e.g. GPU_IDS="0,1,2,3"
IFS=',' read -r -a GPU_IDS_ARR <<< "${GPU_IDS:-0,1,2,3}"
if [[ "${#GPU_IDS_ARR[@]}" -ne 4 ]]; then
  echo "Expected 4 GPU ids in GPU_IDS, got: ${GPU_IDS:-}" >&2
  exit 1
fi

PART0="${OUT_DIR}/gpu0"
PART1="${OUT_DIR}/gpu1"
PART2="${OUT_DIR}/gpu2"
PART3="${OUT_DIR}/gpu3"
MERGED="${OUT_DIR}/merged"

mkdir -p "$OUT_DIR"

# Split pruners across 4 jobs (each process loads VLM once on its GPU).
# Order matches eval registration: baseline + proposed.
PRUNERS_0="no_pruning,random,uniform"
PRUNERS_1="divprune,apet,otprune"
PRUNERS_2="tome,fastv_mesh,loco3d"
PRUNERS_3="octree_merge,runlength_curve,reconot,loco3d_dpp,loco3d_nonempty_dpp"

run_one() {
  local gpu_slot="$1"   # 0-3 index into GPU_IDS_ARR
  local pruners="$2"
  local out_sub="$3"
  local phys="${GPU_IDS_ARR[$gpu_slot]}"
  mkdir -p "$out_sub" "$out_sub/logs"
  echo "[GPU slot $gpu_slot -> CUDA_VISIBLE_DEVICES=$phys] pruners=$pruners"
  echo "[GPU slot $gpu_slot] proposed-pruner logs (if any) -> $out_sub/logs/<method>.log and <method>.deep.jsonl"
  CUDA_VISIBLE_DEVICES="$phys" \
  SHAPELLM_EVAL_LOG_DIR="$out_sub/logs" \
  SHAPELLM_EVAL_LOG_DEEP_EVERY="${SHAPELLM_EVAL_LOG_DEEP_EVERY:-20}" \
  python -m eval.run_eval \
    --data-csv "$DATA_CSV" \
    --glb-dir "$GLB_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$out_sub" \
    --eval-config-dir "$EVAL_CONFIG_DIR" \
    --pruners "$pruners" \
    --keep-ratios "$KEEP_RATIOS" \
    --seed "$SEED" \
    --device cuda \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --vlm-torch-dtype "$VLM_TORCH_DTYPE"
}

run_one 0 "$PRUNERS_0" "$PART0" &
PID0=$!
run_one 1 "$PRUNERS_1" "$PART1" &
PID1=$!
run_one 2 "$PRUNERS_2" "$PART2" &
PID2=$!
run_one 3 "$PRUNERS_3" "$PART3" &
PID3=$!

ec=0
wait $PID0 || ec=1
wait $PID1 || ec=1
wait $PID2 || ec=1
wait $PID3 || ec=1
if [[ "$ec" -ne 0 ]]; then
  echo "One or more GPU jobs failed; skipping merge." >&2
  exit "$ec"
fi

echo "Merging into $MERGED ..."
# -u: unbuffered stdout so the [merge +Ns] progress lines stream live.
# Compact JSON by default (orjson if available) -> 5-20x faster on huge results.json.
# Override with MERGE_INDENT=2 for pretty output, MERGE_WORKERS=N for parallelism.
python -u -m eval.merge_eval_results \
  --inputs "$PART0" "$PART1" "$PART2" "$PART3" \
  --output-dir "$MERGED" \
  ${MERGE_DEDUPE:+--dedupe} \
  ${MERGE_INDENT:+--indent "$MERGE_INDENT"} \
  ${MERGE_WORKERS:+--workers "$MERGE_WORKERS"} \
  ${MERGE_SKIP_BIG_JSON:+--skip-merged-json}

mkdir -p "$MERGED/logs"
for slot in 0 1 2 3; do
  src="${OUT_DIR}/gpu${slot}/logs"
  if [[ -d "$src" ]]; then
    for f in "$src"/*; do
      [[ -e "$f" ]] || continue
      base="$(basename "$f")"
      cp -f "$f" "$MERGED/logs/gpu${slot}_${base}"
    done
  fi
done

echo "Done. Full table: $MERGED/summary.csv"
echo "Pruner logs collected under $MERGED/logs (also per-GPU at ${OUT_DIR}/gpu*/logs/)."
