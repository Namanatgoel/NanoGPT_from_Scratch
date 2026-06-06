# Custom Decoder-Only Transformer

A production-grade, end-to-end custom Decoder-Only Transformer pipeline built from scratch in PyTorch. This project encompasses the full lifecycle of a Large Language Model (LLM): from raw data scraping and custom Byte-Pair Encoding (BPE) tokenization to base pre-training, architecture ablation studies, scaling experiments, and domain-specific fine-tuning (e.g., stylized rap lyrics).

## 📖 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Configuration Guide](#-configuration-guide)
- [Execution Pipeline](#-execution-pipeline)
  - [Phase 1: Data Preparation](#phase-1-data-preparation)
  - [Phase 2: Base Model Pre-Training](#phase-2-base-model-pre-training)
  - [Phase 3: Experimentation & Scaling](#phase-3-experimentation--scaling)
  - [Phase 4: Behavioral Profiling](#phase-4-behavioral-profiling)
  - [Phase 5: Domain Fine-Tuning](#phase-5-domain-fine-tuning)
  - [Phase 6: Multi-Strategy Inference](#phase-6-multi-strategy-inference)
- [Architecture Details](#-architecture-details)

## ✨ Features

- **From-Scratch Transformer Architecture:** A clean, heavily optimized PyTorch implementation of a GPT-2 style causal language model (`model/transformer.py`).
- **Custom Tokenizers:** Includes both a custom Byte-Pair Encoding (BPE) tokenizer and a Character-level tokenizer, built completely without external tokenization libraries (`tokenizer/`).
- **Decoupled Configurations:** Centralized configuration registry (`configs/experiment_configs.py`) allowing for reproducible scaling, ablation sweeps, and hyperparameter tuning.
- **Robust Data Pipeline:** Highly efficient memory-mapped dataset handling supporting millions of tokens with O(1) random-access batching for training (`data/prepare.py`).
- **Multi-Strategy Inference Engine:** Supports Greedy, Temperature, Top-K, and Top-P (Nucleus) sampling mechanisms, featuring a custom "Ghost Byte Blocker" for robust UTF-8 token decoding (`evaluation/generate.py`, `evaluation/inference.py`).
- **Automated Stress Testing:** Adversarial robustness profiling to systematically test the model against context overflows, repetition loops, and high-temperature hallucinations (`evaluation/stress_test.py`).

## 📁 Project Structure

```text
Project/
├── configs/
│   └── experiment_configs.py      # Master configuration registry for all models and ablations
├── data/
│   ├── prepare.py                 # Master orchestrator: tokenizes corpora into memory-mapped arrays
│   └── fetching_dataset_scripts/
│       ├── fetch_arxiv.py         # Extracts scientific abstracts from the ArXiv API
│       ├── prepare_training_data.py # Flattens CSVs into heavily conditioned text sequences
│       └── updatedLyricsScraping.py # Scrapes raw lyrics seamlessly using the Genius API
├── evaluation/
│   ├── generate.py                # Multi-strategy generation (Greedy, Temp, Top-K, Top-P)
│   ├── inference.py               # Prompt-conditioned inference pipeline with Ghost Byte Masking
│   ├── stress_test.py             # Adversarial automated prompt testing framework
│   └── visualize_part3.py         # PCA embedding visualizations and attention head heatmaps
├── experiments/                   # Directory where trained model weights/checkpoints are saved
├── logs/                          # Directory for training metrics and historical loss trajectories
├── model/
│   └── transformer.py             # Core PyTorch Decoder-only Transformer implementation (LayerNorm, Block, GPT)
├── notebooks/                     # Jupyter notebooks for exploratory data analysis (EDA)
│   ├── EDA.ipynb                  
│   └── inspect_data.ipynb         
├── output/                        # Visualizations, charts, and generated sample artifacts
├── reports/                       # Formal reports and technical documentation output
├── tokenizer/
│   ├── bpe_tokenizer.py           # Custom Byte-Pair Encoding (BPE) implementation
│   ├── char_tokenizer.py          # Character-level tokenizer implementation
│   └── *_meta.pkl                 # Serialized vocabularies and learned merge tables
├── training/
│   ├── train.py                   # Standard pre-training loop with cross-entropy loss tracking
│   └── finetune.py                # Specialized loop for domain adaptation (reduced LR, localized batches)
├── debug.py                       # Helper script for interactive checkpoint inspection and debugging
├── env.py                         # Environment and hardware validation (CUDA/MPS detection)
└── README.md                      # This file
```

*(Note: Data files, model checkpoints, `.json`/`.jsonl`/`.csv`/`.bin` artifacts, and `__pycache__` directories are dynamically generated and omitted for clarity).*

## ⚙️ Installation & Setup

1. **Clone the Repository**
2. **Setup Python Environment (Python 3.8+ Recommended)**
   ```bash
   pip install torch numpy scikit-learn matplotlib seaborn pandas lyricsgenius
   ```
3. **Verify Environment**
   ```bash
   python env.py
   ```
   *Expected Output: PyTorch Version, CUDA Availability, and GPU Device Name.*

## 🛠 Configuration Guide

All model hyperparameters, network architectures, and training loop parameters are entirely decoupled into a central registry located at `configs/experiment_configs.py`. 

**Base Template Example:**
```python
BASE_TEMPLATE = {
    'vocab_size': 512,      # BPE vocabulary ceiling
    'block_size': 128,      # Maximum context window (sequence capacity)
    'bias': False,          # Disable biases in Linear/LayerNorm to reduce parameters
    'dropout': 0.1,         # Regularization probability
    'ffn_multiplier': 4,    # FeedForward expansion ratio (n_embd -> 4 * n_embd)
    'learning_rate': 6e-4,  # AdamW optimization learning rate
    'batch_size': 32,       # Sequences per gradient step
    'max_iters': 15000,     # Total training steps
    'eval_interval': 100,   # Steps between validation evaluations
    'eval_iters': 20        # Batches averaged per evaluation step
}
```
To run an experiment, simply pass the target configuration dictionary key (e.g., `task_4_1_baseline`, `task_4_3_large`) as a CLI argument to the training scripts.

## 🚀 Execution Pipeline

### Phase 1: Data Preparation
Ensure your raw corpora (e.g., clean poems and rap JSON/JSONL) are present in the `data/` directory. If they aren't, use the scripts inside `data/fetching_dataset_scripts/`.

Explore the data first:
```bash
jupyter notebook notebooks/EDA.ipynb
jupyter notebook notebooks/inspect_data.ipynb
```

Compile the text into memory-mapped binary files (train/val) and train the custom BPE/Char tokenizers:
```bash
python data/prepare.py
```
*This generates `train_bpe.bin`, `val_bpe.bin`, and tokenizer `.pkl` state files required for training.*

### Phase 2: Base Model Pre-Training
Train a randomly initialized foundational model on the prepared dataset using the baseline configuration.
```bash
python training/train.py task_4_1_baseline
```
*Logs (JSON metrics) are saved to `logs/`, and the checkpoint `.pt` is saved to `experiments/`.*

### Phase 3: Experimentation & Scaling
Execute hyperparameter sweeps and structural scaling experiments by referencing predefined configuration IDs.

**Hyperparameter Sweeps:**
```bash
python training/train.py task_4_2_high_lr
python training/train.py task_4_2_large_context
```

**Scaling Laws:**
```bash
python training/train.py task_4_3_tiny
python training/train.py task_4_3_medium
python training/train.py task_4_3_large
```

### Phase 4: Behavioral Profiling
Evaluate the pre-trained base model generation capabilities and robustness.
```bash
# Test various decoding strategies (Greedy, Top-K, Top-P, Temperature)
python evaluation/generate.py

# Subject the model to automated adversarial prompts and analyze failure modes
python evaluation/stress_test.py
```

### Phase 5: Domain Fine-Tuning
Adapt the pre-trained foundation model to a specific domain (e.g., Rap lyrics) using explicit conditioning envelopes (`<|type|> rap`, `[mood: emotional]`) and a significantly reduced learning rate (`1e-4`).
```bash
python training/finetune.py
```
*(Requires `task_4_1_baseline_checkpoint.pt` to exist prior to running).*

### Phase 6: Multi-Strategy Inference
Run prompt-conditioned generation using the fine-tuned model checkpoint. This script specifically implements a **Ghost Byte Blocker** to permanently ban the model from outputting untrained UTF-8 symbols.
```bash
python evaluation/inference.py --prompt "Deep in the shadows of the concrete jungle," --temperature 0.7 --top_k 40
```

## 🧠 Architecture Details

The core language model (`model/transformer.py`) is a Decoder-only architecture utilizing standard modern practices:
- **Pre-Norm Architecture**: `LayerNorm` is strictly applied *before* the Multi-Head Attention and FeedForward blocks, rather than after, stabilizing deep gradients.
- **Fused QKV Projections**: Concatenates Query, Key, and Value projections into a single Matrix Multiplication (`3 * n_embd`), heavily optimizing CUDA kernel launches.
- **GELU Activations**: Prevents the "dying neuron" problem found in standard ReLU layers inside the MLP Block.
- **Weight Tying**: Shares weights between the token embedding matrix and the final Language Model head to save structural parameters (`vocab_size * n_embd`) and improve semantic consistency.
- **No Biases**: Biases are completely disabled across `Linear` and `LayerNorm` layers, serving as a mild regularizer and improving training efficiency.
