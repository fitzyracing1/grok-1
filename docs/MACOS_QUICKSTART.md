# macOS Quickstart

This guide helps you set up a development workflow on Apple Silicon.

## Reality Check

Grok-1 is a 314B model. Full local inference is generally not feasible on a single Mac due to memory limits.

You can still use this repo on macOS for:

- workflow setup,
- evaluation harness development,
- prompt iteration,
- and pre-flight validation before cloud GPU execution.

## 1) Create Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2) Install Dependencies

The default requirements target CUDA (`jax[cuda12-pip]`). On macOS, use CPU JAX for tool development:

```bash
pip install dm_haiku==0.0.12 numpy==1.26.4 sentencepiece==0.2.0 "jax==0.4.25"
```

## 3) Run Capacity Check

```bash
python scripts/system_check.py
```

## 4) Optional: Download Weights

Only do this if you have a concrete multi-GPU execution plan and storage budget.

```bash
pip install "huggingface_hub[hf_transfer]"
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
```

## 5) Next Step

Follow `docs/PERFECT_GROK1_PLAN.md` to improve quality systematically.

For immediate quality gains without weight tuning, start with local RAG:

```bash
export XAI_API_KEY=YOUR_KEY
python scripts/rag_starter.py --query "Summarize this repo" --corpus . --ext .md,.py --top-k 6 --show-context
```