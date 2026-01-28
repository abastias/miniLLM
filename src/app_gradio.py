from __future__ import annotations

import gradio as gr

from .generate import load_model_and_tokenizer, generate


def build_app(checkpoint: str = "minillm.pt", tokenizer_dir: str = "tokenizer") -> gr.Blocks:
    model, tokenizer, device = load_model_and_tokenizer(checkpoint_path=checkpoint, tokenizer_dir=tokenizer_dir)

    def run(prompt: str, tokens: int, temperature: float, top_k: int, show_steps: bool):
        out, steps = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=int(tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            return_steps=bool(show_steps),
        )
        details = ""
        if steps:
            lines = []
            for s in steps[: min(25, len(steps))]:
                top = ", ".join([f"`{tok}` {p:.2f}" for tok, p in s.top_tokens[:5]])
                lines.append(f"- **step {s.step:02d}** chose `{s.chosen_token}` | {top}")
            details = "\n".join(lines)
        return out, details

    with gr.Blocks(title="miniLLM — teaching UI") as demo:
        gr.Markdown(f"# miniLLM\nRunning on **{device}**\n\nA teaching-focused GPT-style mini model.")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=4, value="Once upon a time,")
        with gr.Row():
            tokens = gr.Slider(1, 300, value=80, step=1, label="Max new tokens")
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
            top_k = gr.Slider(1, 200, value=40, step=1, label="Top‑k")
        show_steps = gr.Checkbox(label="Show sampling details (top tokens)", value=False)
        btn = gr.Button("Generate")
        out = gr.Textbox(label="Output", lines=10)
        details = gr.Markdown()

        btn.click(run, inputs=[prompt, tokens, temperature, top_k, show_steps], outputs=[out, details])

        gr.Markdown(
            "### Teaching tips\n"
            "- Lower **temperature** to make output more deterministic.\n"
            "- Reduce **top‑k** to constrain choices (more repetitive).\n"
            "- Enable **sampling details** to see how the model chooses tokens."
        )
    return demo


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="minillm.pt")
    p.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    demo = build_app(args.checkpoint, args.tokenizer_dir)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
