import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from tokenizers import ByteLevelBPETokenizer
from model.minigpt import MiniGPT
from tqdm import tqdm
import os

# Tokenizer training
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["data/your_data.txt"], vocab_size=8000, min_frequency=2)
os.makedirs("tokenizer", exist_ok=True)
tokenizer.save_model("tokenizer")
tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")

# Prepare training data
with open("data/your_data.txt", "r") as f:
    text = f.read()

tokens = tokenizer.encode(text).ids
block_size = 128

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.block_size+1]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

dataset = TextDataset(tokens, block_size)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = MiniGPT(vocab_size=tokenizer.get_vocab_size()).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(3):
    for xb, yb in tqdm(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "minillm.pt")
