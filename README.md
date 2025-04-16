# 🧠 Mini LLM

Build your own GPT-style language model (LLM) from scratch using PyTorch.  
This project is designed to work efficiently on a Mac M CPU or any other machine with modest resources.

---

## 🚀 Features

- ✅ Train your own tokenizer (BPE)
- ✅ Implement a decoder-only Transformer (GPT-style)
- ✅ Generate text using top-k sampling
- ✅ Interact with your model using a command-line chat interface
- ✅ Run on Apple Silicon (`mps`) or CPU

---

## 🗂️ Project Structure

```
mini_llm/
├── README.md
├── requirements.txt
├── fundamentals.md
├── train.py
├── generate.py
├── chat.py
├── model/
│   └── minigpt.py
├── tokenizer/
│   ├── vocab.json
│   └── merges.txt
├── data/
│   └── your_data.txt      ← Your training dataset
```

---

## 📥 Dataset Required

Before training, you must add your own dataset in plain text format.

I recommend starting with:

```bash
curl -o data/your_data.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Or you can use any `.txt` file of your choosing. The better and more varied the text, the smarter your model.

---

## ⚙️ Setup Instructions

### 1. Create Python Environment

```bash
python3.11 -m venv myllm-env
source myllm-env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Train the Model

```bash
python train.py
```

This will:
- Train a tokenizer using your dataset
- Train the transformer model
- Save weights to `minillm.pt`

---

## ✍️ Generate Text

```bash
python generate.py
```

Change the prompt inside the file or modify the function to generate different samples.

---

## 💬 Chat with Your Model

```bash
python chat.py --prompt "Tell me a joke"
```

Or start an interactive loop:

```bash
python chat.py
```

Options:
- `--temperature` (default: 1.0)
- `--top_k` (default: 20)
- `--tokens` (default: 50)

---

## 🧪 Summary

This repo provides a compact and educational GPT-style model pipeline: from tokenizer, to training, to sampling and chat interaction.

Built with ❤️ to demystify LLMs.


by Alfonso G. Bastias, Ph.D.

