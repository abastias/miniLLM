from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer

from .model import MiniGPT


def get_device(prefer_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class StepInfo:
    step: int
    chosen_id: int
    chosen_token: str
    top_tokens: List[Tuple[str, float]]  # (token, prob)


@torch.no_grad()
def generate(
    model: MiniGPT,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    return_steps: bool = False,
) -> tuple[str, Optional[List[StepInfo]]]:
    model.eval()
    device = next(model.parameters()).device

    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    steps: List[StepInfo] = []

    for t in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size :]  # crop context
        logits = model(idx_cond)  # (B, T, vocab)
        logits = logits[:, -1, :]  # (B, vocab)

        # temperature
        logits = logits / max(temperature, 1e-8)

        # top-k filter
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            cutoff = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx = torch.cat([idx, next_id], dim=1)

        if return_steps:
            # decode top tokens for inspection
            topv, topi = torch.topk(probs[0], k=min(10, probs.size(-1)))
            top_tokens = []
            for p, tid in zip(topv.tolist(), topi.tolist()):
                tok = tokenizer.decode([tid])
                tok = tok.replace("\n", "\\n")
                top_tokens.append((tok, float(p)))
            chosen = int(next_id.item())
            chosen_tok = tokenizer.decode([chosen]).replace("\n", "\\n")
            steps.append(StepInfo(step=t+1, chosen_id=chosen, chosen_token=chosen_tok, top_tokens=top_tokens))

    out = tokenizer.decode(idx[0].tolist())
    return out, (steps if return_steps else None)


def load_model_and_tokenizer(
    checkpoint_path: str = "minillm.pt",
    tokenizer_dir: str = "tokenizer",
    *,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    dropout: float = 0.1,
) -> tuple[MiniGPT, ByteLevelBPETokenizer, str]:
    from .data import load_tokenizer

    device = get_device()
    tokenizer = load_tokenizer(tokenizer_dir)
    model = MiniGPT(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model, tokenizer, device
