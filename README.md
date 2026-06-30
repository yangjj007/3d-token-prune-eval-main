# Token 剪枝评测（eval）

## 文档索引

- [ShapeLLM 原始说明](docs/ShapeLLM-README.md)
- [项目概览](docs/overview.md)
- [如何适配其他评估及算法框架](docs/如何适配其他评估及算法框架.md)
- [归档 demo 说明](archive/demo/README.md)

## junjie_llm 顶层结构

`/data/xujinyi/junjie_llm` 是当前工作区根目录，主要目录如下：

- `3d-token-prune-eval-main/`：当前主项目，包含 3D token 剪枝评测代码、配置、脚本和项目内文档。
- `data/`：评测数据目录，包含 `metadata.csv`、`.glb` 网格文件、`mesh_voxel_cache/` 和 `raw/` 等数据资源。
- `output/`：跑测输出目录，保存 `results.json`、`summary.csv`、日志和分 pruner 结果。
- `model/`：模型与缓存目录，包含 `hf-cache/` 和 `OpenEVA/` 等本地模型资源。
- `algorithm/`：外部剪枝算法参考源码目录，用于对照和迁移思路；当前评测实际调用的是本项目 `eval/` 下已接入的实现。
- `docs/`：工作区级文档，包含项目进展、研究方案和相关工作整理。
- `github/`：外部辅助仓库目录，例如 autoresearch 系列实验/自动化相关代码。
- `.agents/`、`.codex/`、`.git/`：工具与版本管理元数据，一般无需手动修改。

## 当前目录结构

- `eval/`：评测主流程、模型后端、指标、pruner 注册与实现。
- `configs/runs/`：实验跑测 YAML，作为当前推荐入口。
- `configs/eval/`：各剪枝算法的 JSON 超参。
- `configs/vae/`：VAE/Trellis 相关配置，仍是运行依赖。
- `scripts/`：批量跑测、预计算、结果分析脚本。
- `tests/`：轻量单元测试与 mock 测试。
- `trellis/`、`extensions/`、`dataset_toolkits/`：ShapeLLM/Trellis 运行支撑代码。
- `docs/`：项目说明与迁移文档。
- `archive/demo/`：原始 Gradio demo、素材、示例图和 generation 配置；当前评测不依赖。
- `archive/manual/`：一次性手动检查脚本，不参与自动测试。

## 环境与前缀（Linux）

在评测目录 `3d-token-prune-eval-main` 下执行。若遇 OpenMP 与 MKL 冲突，可先设置：

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export SPCONV_ALGO=native
export SHAPELLM_EVAL_LIGHT=1
```

（`run_eval.py` 内已对后两项设默认，一般无需重复导出。）

如果当前 shell 里 `conda activate` 不可用，先按服务器实际安装位置初始化 conda，例如：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

### 硬件与精度

- 需要 **Linux + NVIDIA GPU + CUDA**，Python 环境与 `requirements.txt` 一致。
- **VLM 必须以 bfloat16 加载**（默认 `--vlm-torch-dtype bfloat16`）。**不支持 `--load-in-4bit`**（无需安装 `bitsandbytes`）。
- ShapeLLM-7B bf16 约需 **~15GB+** 显存；H20 等大卡可直接跑。显存紧张时可将 VQVAE 放到 CPU：`--vqvae-device cpu`。

## 数据准备

- `../data/metadata.csv`：含 `file_identifier` 与 `captions`（JSON 数组字符串）。
- 将每个样本的网格放在 `--glb-dir` 下，文件名为 `{file_identifier}.glb`。
- `--glb-dir` 指向实际 `.glb` 目录即可（当前为 `../data`）。

## YAML 跑测配置

实验参数统一放在 `configs/runs/*.yaml`，路径均以 `3d-token-prune-eval-main` 为根目录解析。常用配置：

- `configs/runs/shapellm-full.yaml`
- `configs/runs/eva01-full.yaml`
- `configs/runs/eva01-mock.yaml`
- `configs/runs/loco3d.yaml`

ShapeLLM 跑测：

```bash
cd /data/xujinyi/junjie_llm/3d-token-prune-eval-main
conda activate token-prune-shapellm
python -u -m eval.run_eval --config configs/runs/shapellm-full.yaml
```

说明：

- **`output_dir`**：生成 `results.json` 与 `summary.csv`。
- **`run_log_file`**：自动保存原来需要 `tee` 才能得到的 `eval_run.log`。
- **`eval_config_dir`**：加载 `configs/eval/{pruner}.json` 中的超参。
- **`no_pruning`** 仅在 `keep_ratio=1.0` 时运行。
- 可选：把 YAML 里的 `vqvae_device` 改为 `cpu` 可减轻 GPU 显存压力。

## 模型底座后端

默认后端仍是 ShapeLLM：

```bash
cd /data/xujinyi/junjie_llm/3d-token-prune-eval-main
conda activate token-prune-shapellm
python -u -m eval.run_eval --config configs/runs/shapellm-full.yaml
```

EVA01 后端现在支持同一套 token prune 注册表。普通 embedding 方法直接剪 EVA01 的 512 个 mesh patch token；空间方法会按需加载 VQVAE，先在 1024 latent token 上运行原 pruner，再映射回 EVA01 patch token：

```bash
cd /data/xujinyi/junjie_llm/3d-token-prune-eval-main
conda activate token-prune-eva01
python -u -m eval.run_eval --config configs/runs/eva01-full.yaml
```

If your local OpenEVA LoRA setup requires an explicit base model, set `eva01_base_model_name_or_path` in the YAML.

说明：

- EVA01 后端需要额外安装 OpenEVA，并确保 Python 可 `import eva01`。
- 默认 EVA01 checkpoint 为 `SEELE-AI/EVA01-2B-Instruct-LoRA`，可用 `--eva01-model-id` 覆盖。
- 如需 LoRA base model 路径，可传 `--eva01-base-model-name-or-path`。
- EVA01 后端会固定保留 cls mesh token，并在 512 个 patch token 上按 `keep_ratio` 剪枝；`no_pruning` 仍只在 `keep_ratio=1.0` 下运行。
- `loco3d*`、`octree_merge`、`runlength_curve`、`reconot` 需要 VQVAE 空间 token，请传 `--vqvae-device`；其他 EVA01 patch-embedding 方法不需要额外加载 VQVAE。
- For local CLI/output-contract debugging across all EVA01 baselines, use mock mode; it does not load or download any model:

```bash
cd /data/xujinyi/junjie_llm/3d-token-prune-eval-main
conda activate token-prune-shapellm
python -u -m eval.run_eval --config configs/runs/eva01-mock.yaml
```

The mock config runs all registered EVA01 baseline/proposed pruners and writes outputs to `../output/eva01-baseline-debug/`.

## Baseline 融合状态

| 类别 | 方法 | 状态 |
|------|------|------|
| ShapeLLM 直接接入 | `no_pruning`, `random`, `uniform`, `divprune`, `apet`, `otprune`, `tome`, `fastv_mesh` | 支持 1024 VQVAE token / embedding 序列 |
| EVA01 直接接入 | `no_pruning`, `random`, `uniform`, `divprune`, `apet`, `otprune`, `tome`, `fastv_mesh` | 支持 EVA01 原生 512 patch-like mesh embedding，cls token 固定保留 |
| EVA01 空间映射接入 | `loco3d`, `octree_merge`, `runlength_curve`, `reconot`, `loco3d_dpp`, `loco3d_nonempty_dpp` | 先跑 1024 VQVAE 空间 pruner，再最近邻映射到 EVA01 patch token |
| 需代理特征 | attention-based FastV 变体 | 需要模型内部 attention adapter 后再接入 |
| 暂不兼容 | 模型权重剪枝、点云网络压缩、场景级 3D pruning | 不满足 `BasePruner.prune(token_ids, voxel_grid, vq_embeddings=...)` 契约 |

## 精简命令（loco3d）

```bash
cd /data/xujinyi/junjie_llm/3d-token-prune-eval-main
conda activate token-prune-shapellm
python -u -m eval.run_eval --config configs/runs/loco3d.yaml
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
