from src.generate import load_model_and_tokenizer, generate

if __name__ == "__main__":
    model, tokenizer, device = load_model_and_tokenizer()
    prompt = "How are you?"
    out, _ = generate(model, tokenizer, prompt, max_new_tokens=80, temperature=0.8, top_k=40)
    print(out)
