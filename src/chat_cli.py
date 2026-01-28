from __future__ import annotations

import argparse
from .generate import load_model_and_tokenizer, generate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="miniLLM chat (CLI)")
    p.add_argument("--checkpoint", type=str, default="minillm.pt")
    p.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    p.add_argument("--tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--show_steps", action="store_true", help="Print per-token sampling details")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, device = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
    )
    print(f"miniLLM ready on {device}. Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break

        out, steps = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            return_steps=args.show_steps,
        )

        print("\nBot:", out)
        if steps:
            print("\nSampling details (top-10 probabilities each step):")
            for s in steps[:10]:  # keep readable
                top = ", ".join([f"{tok}:{p:.2f}" for tok, p in s.top_tokens[:5]])
                print(f"  step {s.step:02d} -> '{s.chosen_token}' | {top}")
        print()


if __name__ == "__main__":
    main()
