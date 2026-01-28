from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MiniGPT
from .data import TextDataset, load_tokenizer, train_tokenizer
from .generate import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="miniLLM training (educational GPT)")
    p.add_argument("--data", type=str, default="data/your_data.txt", help="Path to training text file")
    p.add_argument("--tokenizer_dir", type=str, default="tokenizer", help="Where vocab.json/merges.txt live")
    p.add_argument("--train_tokenizer", action="store_true", help="Train tokenizer before training the model")
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--min_frequency", type=int, default=2)

    # model
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # train
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--checkpoint", type=str, default="minillm.pt")
    p.add_argument("--device", type=str, default="", help="Force device: cuda|mps|cpu (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            f"Create it (plain text) or point --data to a valid file."
        )

    # device
    device = args.device.strip() or get_device()
    print(f"Using device: {device}")

    # tokenizer
    if args.train_tokenizer:
        print("Training tokenizer...")
        train_tokenizer(
            input_files=[str(data_path)],
            tokenizer_dir=args.tokenizer_dir,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
    tokenizer = load_tokenizer(args.tokenizer_dir)

    # dataset
    ds = TextDataset(str(data_path), tokenizer, block_size=args.block_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # model
    model = MiniGPT(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        last_loss = None
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            last_loss = float(loss.item())
            pbar.set_postfix(loss=f"{last_loss:.4f}")

        print(f"Epoch {epoch+1}: loss={last_loss:.4f}")

    torch.save(model.state_dict(), args.checkpoint)
    print(f"Saved checkpoint to: {args.checkpoint}")


if __name__ == "__main__":
    main()
