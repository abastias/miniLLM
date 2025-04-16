# 🧠 Fundamentals of Training a Mini Language Model (LLM)

This document walks you through the conceptual and technical foundations of building a simple transformer-based language model from scratch. This is a learning-focused implementation that covers all major steps: tokenization, model design, training, and inference.

I use a small GPT-style architecture designed to be trained on modest hardware, such as a Mac with an M2 chip.

---

## 🔧 `train.py` – How We Train the Model

Training a language model involves teaching it to predict the next token in a sequence, given a context of previous tokens. This is known as a **causal language modeling** task.

### 🔤 1. Tokenizer Training

We first train a **Byte-Pair Encoding (BPE)** tokenizer. Tokenization is the process of converting text into a sequence of numerical identifiers that a model can understand.

```python
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["data/your_data.txt"], vocab_size=8000, min_frequency=2)
tokenizer.save_model("tokenizer")
```

Why train your own tokenizer?
- It's tailored to the dataset vocabulary.
- Efficient tokenization leads to better generalization and reduced training time.

### 📚 2. Dataset Preparation

Once the text is tokenized, we need to prepare it into sequences of tokens to be fed into the model.

```python
class TextDataset(Dataset):
    ...
```

For example, if our input is:

```
"The sun is shining today."
```

The model sees inputs like:

```
["The", "sun", "is"] → target: "shining"
```

This is implemented as a sliding window over the tokenized dataset.

### 🧠 3. Model Training

```python
model = MiniGPT(...)
```

We instantiate a transformer and use **cross-entropy loss** to measure how close the model's predictions are to the actual next token. We optimize this with **AdamW**, a variant of Adam optimized for training transformers.

We also detect the best device for training:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

This enables fast training using Apple Silicon GPUs on M1/M2 Macs.

---

## 🧠 `minigpt.py` – The Model: A Transformer Decoder

Transformers are the dominant architecture in NLP. This model is a **decoder-only** transformer similar to OpenAI’s GPT.

### Model Components:

- `Embedding`: Converts token indices to dense vectors.
- `Positional Embedding`: Injects sequence order information.
- `Transformer Blocks`: These perform self-attention to capture relationships across tokens.
- `LayerNorm`: Helps with training stability.
- `Head`: Outputs a vector of probabilities over the vocabulary.

### Key Hyperparameters:

```python
n_embd = 256       # Embedding size
n_head = 4         # Number of attention heads
n_layer = 4        # Number of transformer blocks
block_size = 128   # Max context length
```

All these can be tuned to balance model size, training time, and quality.

This architecture enables the model to build up contextual understanding from scratch, learning structure like grammar and syntax just from exposure to the training data.

---

## ✍️ `generate.py` – Generating Text from the Model

Once the model is trained, we want it to generate text. This involves:

1. Encoding a prompt with the tokenizer.
2. Feeding it to the model to get a prediction for the next token.
3. Appending the predicted token to the sequence.
4. Repeating the process.

### 🔁 Sampling Logic

We use **top-k sampling with temperature** to avoid repetitive, deterministic outputs:

```python
def sample_logits(logits, temperature=1.0, top_k=10):
    ...
```

- `temperature`: Controls randomness. Higher values → more creativity.
- `top_k`: Limits predictions to the top K most likely tokens → more diversity.

This balances creativity and coherence.

---

## 🧪 Summary Table

| Component     | Role                                                 |
|---------------|------------------------------------------------------|
| `train.py`    | Builds tokenizer, prepares data, trains model        |
| `minigpt.py`  | Implements GPT-style transformer                     |
| `generate.py` | Generates text via top-k sampling                    |

---

## 🌱 What's Next?

This project is a minimal but complete foundation. You can extend it by:

- Adding attention masking to prevent peeking ahead.
- Training on a larger or more structured dataset.
- Adding model checkpointing and validation loss tracking.
- Exporting to ONNX or other formats for serving models.

Understanding these components helps you grasp how GPT-like models work at a low level — a great first step before working with full-scale LLaMA, Mistral, or GPT-2 models.

---

Built with ❤️ to demystify transformers and language models.

---

## 💬 `chat.py` – A Command-Line Chat Interface

Once your model is trained, you can talk to it directly via the terminal!

### Modes

- **Single prompt**:
  ```bash
  python chat.py --prompt "What is AI?"
  ```

- **Interactive chatbot mode**:
  ```bash
  python chat.py
  ```

Type messages and receive responses until you type `exit`.

### Under the Hood

- Loads your trained model and tokenizer.
- Generates tokens one-by-one with top-k sampling.
- Allows real-time interaction with your model's knowledge.

---

## 📥 Data Tips

If you're starting out, we suggest using small and well-formatted datasets like:

- [`TinyShakespeare`](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- Public domain books from [Project Gutenberg](https://www.gutenberg.org/)
- Exported conversations, technical documentation, or domain-specific guides

Ensure your dataset is plain `.txt` and placed in:
```
my_mini_llm/data/your_data.txt
```

Then run:

```bash
python train.py
```

Train longer (10+ epochs) for better performance.



by Alfonso G. Bastias, Ph.D.