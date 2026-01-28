# miniLLM (teaching edition)

A compact, educational **GPT-style (decoder-only)** language model in PyTorch.

This repo is meant for **learning**: you can train a tiny model end-to-end (tokenizer → model → sampling),
then inspect how generation decisions are made.

## What’s new in this update

- **True causal attention** (autoregressive mask) — not bidirectional
- Cleaner project layout (`src/`) with reusable modules
- CLI chat with optional **sampling details**
- Optional **Gradio UI** (sliders + “show top tokens”)

---

## Quickstart

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Put training text here
Create a plain text file:
```
data/your_data.txt
```

### 3) Train tokenizer + model
```bash
python -m src.train --train_tokenizer --data data/your_data.txt
```

This saves:
- `tokenizer/vocab.json`, `tokenizer/merges.txt`
- `minillm.pt` (model weights)

### 4) Chat (CLI)
```bash
python -m src.chat_cli
# or with details:
python -m src.chat_cli --show_steps
```

### 5) Friendly UI (Gradio)
```bash
python -m src.app_gradio
```

---

## Teaching notes

- **Temperature** controls randomness.
- **Top‑k** restricts the model to the k most likely next tokens.
- With `--show_steps` (CLI) or “Show sampling details” (UI) you can see
  the **top token probabilities** at each generation step.

---

by Alfonso G. Bastias, Ph.D.
