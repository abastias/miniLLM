import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=4, n_layer=4, block_size=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, batch_first=True)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_embedding(idx) + self.position_embedding[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
