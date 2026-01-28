# Release Notes — miniLLM Teaching Edition

**Version:** Teaching Edition (tag: `v0.2.0`)  
**Release date:** 2026-01-27

This release refocuses the project into a **teaching-first, GPT-style (decoder-only) miniLLM** with a cleaner codebase, friendlier entry points, and an optional UI for interactive demos.

---

## Highlights

- **Decoder-only Transformer (GPT-style) with causal masking**
  - The model now uses **masked self-attention** so tokens can only attend to previous tokens during training and generation.
- **Teaching-friendly project structure**
  - New, clear modules under `src/` (model, data, training, generation, UI).
- **Friendly interface**
  - Adds a simple **Gradio UI** for prompt → generation, with sampling controls.

---

## What changed (compared to the previous version)

### Model / Architecture
- **Replaced bidirectional encoder-style flow with a causal decoder-only flow**
  - The previous approach was aligned with `TransformerEncoder`-style behavior (bidirectional attention).
  - This release implements a **true causal language model** suitable for “next-token prediction” and GPT-like generation.

### Training Pipeline
- Adds a cleaner training entry point (`src/train.py`) that follows the standard LM pattern:
  - tokenize text
  - build dataset with **shifted labels** (predict next token)
  - train with cross-entropy loss
  - save checkpoints + tokenizer artifacts

### Generation / Sampling
- Adds `src/generate.py` to centralize generation logic and sampling:
  - temperature
  - top-k
  - top-p (nucleus)
  - max_new_tokens
- Optional “teaching” hooks can be extended to show:
  - top-k candidates per step
  - probabilities/logits
  - selected token at each step

### User Experience
- Adds a **CLI chat** entry (`src/chat_cli.py`) for quick testing.
- Adds **Gradio app** (`src/app_gradio.py`) for live demos in class.

### Docs / Repo Organization
- README updated to reflect the recommended workflow:
  - prepare data → train → chat → UI
- Encourages a clear separation between:
  - `src/` (current teaching pipeline)
  - `legacy/` (optional place to keep original implementation if you want to preserve it)

---

## Backward compatibility

If your previous repo had root-level scripts (e.g., `train.py`, `chat.py`, `generate.py`), this release can keep wrappers so existing commands still work, but the recommended path is:

- `python -m src.train`
- `python -m src.chat_cli`
- `python -m src.app_gradio`

---

## Upgrade / Migration notes

1. **Install deps**
   - `pip install -r requirements.txt`

2. **Data**
   - Put training text into `data/your_data.txt` (or equivalent path you choose).

3. **Train**
   - `python -m src.train --train_tokenizer --data data/your_data.txt`

4. **Run**
   - CLI chat: `python -m src.chat_cli`
   - UI: `python -m src.app_gradio`

---

## Known limitations (teaching scope)

- This is a **mini** implementation intended for learning, not SOTA performance.
- No distributed training, mixed precision, or large-scale dataset streaming by default.
- Tokenizer and dataset utilities are intentionally simple.

---

## What’s next (nice add-ons for teaching)

- Token-by-token generation “explain mode” with a top-k table per step
- Attention visualization (layer/head) for a selected prompt
- A notebook with plots: loss curve, token frequency, example generations

