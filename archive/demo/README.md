# Archived ShapeLLM Demo

This directory keeps the original ShapeLLM Gradio demo files that are not part of the current token-pruning evaluation workflow.

Contents:

- `app.py`: original Gradio demo entry, adjusted so it can be launched from the project root.
- `app_old.py`: older Gradio demo entry kept for reference.
- `assets/`: media used by the original README/demo.
- `examples/`: image examples used by the Gradio UI.
- `configs/generation/`: original generation JSON configs.
- `templates.txt`: prompt templates copied from the original project.
- `temper.glb`: legacy demo output/sample file.

Run from the project root if you need the demo:

```bash
cd /data/xujinyi/junjie_llm/3d-token-prune-eval-main
conda activate token-prune-shapellm
python archive/demo/app.py
```

The active evaluation entry remains:

```bash
cd /data/xujinyi/junjie_llm/3d-token-prune-eval-main
conda activate token-prune-shapellm
python -u -m eval.run_eval --config configs/runs/shapellm-full.yaml
```
