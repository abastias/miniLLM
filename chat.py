import torch
import argparse
from model.minigpt import MiniGPT
from tokenizers import ByteLevelBPETokenizer
import torch.nn.functional as F


# mps -> Apple Silicon: M CPUs
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load tokenizer and model
tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
model = MiniGPT(vocab_size=tokenizer.get_vocab_size())
model.load_state_dict(torch.load("minillm.pt", map_location=device))
model.to(device)
model.eval()

def sample_logits(logits, temperature=1.0, top_k=10):
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_val = values[:, -1].unsqueeze(1)
        logits[logits < min_val] = -float("Inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def generate(prompt, max_new_tokens=50, temperature=1.0, top_k=10):
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_tensor)
        next_token_logits = logits[0, -1, :]
        next_token = sample_logits(next_token_logits.unsqueeze(0), temperature, top_k)
        input_tensor = torch.cat([input_tensor, next_token], dim=1)

    output = tokenizer.decode(input_tensor[0].cpu().numpy())
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="Text prompt to generate from")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=50)
    args = parser.parse_args()

    if args.prompt:
        output = generate(args.prompt, args.tokens, args.temperature, args.top_k)
        print(output)
    else:
        print("Welcome to MiniLLM Chat! Type 'exit' to quit.")
        while True:
            prompt = input("You: ")
            if prompt.lower() in ("exit", "quit"):
                break
            output = generate(prompt, args.tokens, args.temperature, args.top_k)
            print("Bot:", output)
