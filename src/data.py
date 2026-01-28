from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer


@dataclass
class TokenizerPaths:
    vocab_json: str
    merges_txt: str


def load_tokenizer(tokenizer_dir: str = "tokenizer") -> ByteLevelBPETokenizer:
    td = Path(tokenizer_dir)
    vocab = td / "vocab.json"
    merges = td / "merges.txt"
    if not vocab.exists() or not merges.exists():
        raise FileNotFoundError(
            f"Tokenizer files not found in {tokenizer_dir}. "
            f"Run: python -m src.train --train_tokenizer"
        )
    return ByteLevelBPETokenizer(str(vocab), str(merges))


def train_tokenizer(
    input_files: List[str],
    tokenizer_dir: str = "tokenizer",
    vocab_size: int = 8000,
    min_frequency: int = 2,
) -> ByteLevelBPETokenizer:
    tok = ByteLevelBPETokenizer()
    tok.train(files=input_files, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=["<pad>"])
    Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)
    tok.save_model(tokenizer_dir)
    return ByteLevelBPETokenizer(str(Path(tokenizer_dir)/"vocab.json"), str(Path(tokenizer_dir)/"merges.txt"))


class TextDataset(Dataset):
    """
    Turns a text file into (x, y) training pairs for causal language modeling.

    We build contiguous token ids, then slice windows of length block_size:
      x = tokens[i : i+block_size]
      y = tokens[i+1 : i+block_size+1]
    """
    def __init__(self, text_path: str, tokenizer: ByteLevelBPETokenizer, block_size: int = 128):
        self.block_size = block_size
        text = Path(text_path).read_text(encoding="utf-8", errors="ignore")
        ids = tokenizer.encode(text).ids
        if len(ids) < block_size + 1:
            raise ValueError("Not enough tokens. Use more text or a smaller block_size.")
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        # number of windows
        return self.data.size(0) - (self.block_size + 1)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[i : i + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
