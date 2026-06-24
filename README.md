# Token 剪枝评测（eval）

## 环境与前缀（Linux）

在仓库根目录 `eval-main` 下执行。若遇 OpenMP 与 MKL 冲突，可先设置：

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export SPCONV_ALGO=native
export SHAPELLM_EVAL_LIGHT=1
```

（`run_eval.py` 内已对后两项设默认，一般无需重复导出。）

### 硬件与精度

- 需要 **Linux + NVIDIA GPU + CUDA**，Python 环境与 `requirements.txt` 一致。
- **VLM 必须以 bfloat16 加载**（默认 `--vlm-torch-dtype bfloat16`）。**不支持 `--load-in-4bit`**（无需安装 `bitsandbytes`）。
- ShapeLLM-7B bf16 约需 **~15GB+** 显存；H20 等大卡可直接跑。显存紧张时可将 VQVAE 放到 CPU：`--vqvae-device cpu`。

## 数据准备

- `data/metadata.csv`：含 `file_identifier` 与 `captions`（JSON 数组字符串）。
- 将每个样本的网格放在 `--glb-dir` 下，文件名为 `{file_identifier}.glb`。
- `--glb-dir` 指向实际 `.glb` 目录即可（如 `data`）。

## 一键跑 baseline（双卡示例）

VLM 在 `cuda:0`，VQVAE 编码在 `cuda:1`（单卡时两者都用 `cuda:0`）：

```bash
python -m eval.run_eval \
  --data-csv data/metadata.csv \
  --glb-dir data \
  --num-samples 10 \
  --output-dir eval_results \
  --eval-config-dir configs/eval \
  --vlm-torch-dtype bfloat16 \
  --device cuda:0 \
  --vqvae-device cuda:1 \
  --pruners no_pruning,random,uniform,divprune,apet,otprune,tome,fastv_mesh \
  --keep-ratios 0.75,0.5,0.25,0.1

python -m eval.run_eval \
  --data-csv data/metadata.csv \
  --glb-dir data \
  --num-samples 10 \
  --output-dir eval_results \
  --eval-config-dir configs/eval \
  --vlm-torch-dtype bfloat16 \
  --device cuda:0 \
  --vqvae-device cuda:1 \
  --pruners no_pruning \
  --keep-ratios 1.0
```

说明：

- **`--output-dir`**：生成 `results.json` 与 `summary.csv`。
- **`--eval-config-dir`**：加载 `configs/eval/{pruner}.json` 中的超参。
- **`no_pruning`** 仅在 `keep_ratio=1.0` 时运行。
- 可选：`--vqvae-device cpu` 减轻 GPU 显存压力。

## 模型底座后端

默认后端仍是 ShapeLLM：

```bash
python -m eval.run_eval --model-backend shapellm ...
```

EVA01 第一版只接入公开 mesh understanding / caption 评估路径，不对 EVA01 内部 mesh feature 做剪枝：

```bash
python -m eval.run_eval \
  --model-backend eva01 \
  --data-csv data/metadata.csv \
  --glb-dir data \
  --num-samples 10 \
  --output-dir eval_results_eva01 \
  --pruners no_pruning \
  --keep-ratios 1.0 \
  --device cuda:0
```

Formal EVA01 baseline launch command (real model; loads OpenEVA and the EVA01 checkpoint):

```bash
mkdir -p "eval_results_eva01/logs"
SHAPELLM_ENABLE_SEMANTIC_METRICS=0 python -u -m eval.run_eval \
  --model-backend eva01 \
  --data-csv data/metadata.csv \
  --glb-dir data \
  --num-samples -1 \
  --output-dir "eval_results_eva01" \
  --pruners no_pruning \
  --keep-ratios 1.0 \
  --device cuda:0 \
  --vlm-torch-dtype bfloat16 \
  --seed 42 \
  2>&1 | tee "eval_results_eva01/eval_run.log"
```

If your local OpenEVA LoRA setup requires an explicit base model, add `--eva01-base-model-name-or-path /path/to/base-model`.

说明：

- EVA01 后端需要额外安装 OpenEVA，并确保 Python 可 `import eva01`。
- 默认 EVA01 checkpoint 为 `SEELE-AI/EVA01-2B-Instruct-LoRA`，可用 `--eva01-model-id` 覆盖。
- 如需 LoRA base model 路径，可传 `--eva01-base-model-name-or-path`。
- EVA01 后端当前只允许 `--pruners no_pruning --keep-ratios 1.0`；其他剪枝需要 EVA01 feature-pruning adapter。
- For local CLI/output-contract debugging across all EVA01 baselines, use mock mode; it does not load or download any model:

```bash
bash scripts/run_eva01_baselines_debug.sh
```

The script runs `no_pruning,random,uniform,divprune,apet,otprune,tome,fastv_mesh` and writes outputs to `artifacts/eva01-baseline-debug/`.

## Baseline 融合状态

| 类别 | 方法 | 状态 |
|------|------|------|
| 直接接入 | `no_pruning`, `random`, `uniform` | 已按 1024 mesh token index 选择接入 |
| 直接接入 | `divprune`, `apet`, `otprune`, `tome`, `fastv_mesh` | 已按 ShapeLLM VQ embedding / token 序列代理接入 |
| 需代理特征 | EVA01 内部 mesh feature pruning、attention-based FastV 变体 | 需要模型内部 feature/attention adapter 后再接入 |
| 暂不兼容 | 模型权重剪枝、点云网络压缩、场景级 3D pruning | 不满足 `BasePruner.prune(token_ids, voxel_grid, vq_embeddings=...)` 契约 |

## 精简命令（loco3d）

```bash
mkdir -p "eval_results_loco3d/logs"
CUDA_VISIBLE_DEVICES=0,1 SHAPELLM_EVAL_LOG_DIR="eval_results_loco3d/logs" python -u -m eval.run_eval   --data-csv data/metadata.csv   --glb-dir data   --num-samples -1   --output-dir "eval_results_loco3d"   --eval-config-dir configs/eval   --pruners loco3d   --keep-ratios 0.75,0.5,0.25,0.1   --vlm-torch-dtype bfloat16   --seed 42   --device cuda:0   --vqvae-device cuda:1   2>&1 | tee "eval_results_loco3d/eval_run.log"
```

## 依赖提示

可选安装 `nltk` 以获得更稳定的 BLEU 平滑实现。

若在 Windows 原生 Conda 下出现 `torch` / `shm.dll` 或 `numpy` 版本问题，请在 Linux 或 WSL2 中运行。

## 输出文件

- **`results.json`**：样本 × 剪枝器 × `keep_ratio` 明细。
- **`summary.csv`**：按 `pruner` + `keep_ratio` 聚合。
- **`by_pruner/<name>.json`**：按方法拆分。

## 指标说明

BLEU 对每条参考句分别计算句级分数，再对所有参考取最大值。ROUGE-L 同样取 max-over-reference。

语义相似度指标已接入 `results.json` / `summary.csv`：

- `sentence_bert`, `sentence_bert_mean`
- `simcse`, `simcse_mean`

默认 `eval_score` 仍只使用 ROUGE-L / BLEU-4 / BLEU-1，保证历史结果可比。可通过环境变量控制语义指标：

```bash
export SHAPELLM_ENABLE_SEMANTIC_METRICS=0      # 禁用 Sentence-BERT / SimCSE
export SHAPELLM_TEXT_METRIC_DEVICE=cpu         # cpu / cuda / auto
export SHAPELLM_SENTENCE_BERT_MODEL=sentence-transformers/all-MiniLM-L6-v2
export SHAPELLM_SIMCSE_MODEL=princeton-nlp/sup-simcse-roberta-base
```

详见 `eval/metrics.py` 与 `eval/score_eval_run.py`。
