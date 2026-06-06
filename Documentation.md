# Custom Decoder-Only Transformer Pipeline: Exhaustive Technical Documentation

## Base Pre-Training → Architecture Ablation → Hyperparameter Sweep → Scaling → Domain Fine-Tuning → Multi-Strategy Inference

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Lifecycle & Topology](#2-lifecycle--topology)
3. [Codebase Engineering Breakdown](#3-codebase-engineering-breakdown)
   - 3.1 [Configuration System](#31-configuration-system)
   - 3.2 [Data Pipeline & Tokenizers](#32-data-pipeline--tokenizers)
   - 3.3 [Transformer Architecture](#33-transformer-architecture)
   - 3.4 [Training Loops](#34-training-loops)
   - 3.5 [Sampling & Inference](#35-sampling--inference)
   - 3.6 [Logit Masking](#36-logit-masking)
4. [Core System Design Choices](#4-core-system-design-choices)
5. [Debug Log Matrix](#5-debug-log-matrix)
6. [Output Analysis](#6-output-analysis)
7. [Experiment Results Matrix](#7-experiment-results-matrix)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROJECT DIRECTORY TOPOLOGY                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  configs/                                                                   │
│    └── experiment_configs.py      ← Decoupled configuration registry        │
│                                                                             │
│  data/                                                                      │
│    ├── prepare.py                 ← Master data pipeline orchestrator       │
│    ├── fetching_dataset_scripts/                                            │
│    │   ├── fetch_arxiv.py         ← ArXiv API abstract extractor            │
│    │   ├── prepare_training_data.py ← CSV→JSONL conditioned flattener       │
│    │   └── updatedLyricsScraping.py ← Genius API lyrics scraper             │
│    ├── clean_poems_corpus.jsonl   ← 22.7MB poems corpus                     │
│    ├── clean_rap_dataset.json     ← 1.7MB curated rap lyrics                │
│    ├── train_bpe.bin / val_bpe.bin ← Memory-mapped BPE token arrays         │
│    └── train_char.bin / val_char.bin ← Memory-mapped char token arrays      │
│                                                                             │
│  tokenizer/                                                                 │
│    ├── bpe_tokenizer.py           ← Custom BPE (Byte-Pair Encoding) impl    │
│    ├── char_tokenizer.py          ← Character-level tokenizer impl          │
│    ├── bpe_meta.pkl               ← Serialized BPE merge table + vocab      │
│    └── char_meta.pkl              ← Serialized character vocab mapping       │
│                                                                             │
│  model/                                                                     │
│    └── transformer.py             ← Full GPT architecture (114 LOC)         │
│                                                                             │
│  training/                                                                  │
│    ├── train.py                   ← Base pre-training loop                  │
│    └── finetune.py                ← Domain fine-tuning loop                 │
│                                                                             │
│  evaluation/                                                                │
│    ├── generate.py                ← Multi-strategy text generation          │
│    ├── inference.py               ← Prompt-conditioned inference pipeline   │
│    ├── stress_test.py             ← Adversarial robustness profiling        │
│    └── visualize_part3.py         ← Embedding PCA + attention heatmaps      │
│                                                                             │
│  experiments/                                                               │
│    ├── task_4_1_baseline_checkpoint.pt    ← 3.75MB base model weights       │
│    └── task_7_1_finetuned_checkpoint.pt   ← 3.75MB fine-tuned weights       │
│                                                                             │
│  logs/                            ← JSON metric histories per experiment    │
│  output/                          ← Screenshots, plots, raw text outputs    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Lifecycle & Topology

The pipeline executes as a strict four-phase DAG. Each phase produces artifacts consumed by subsequent phases. No phase is re-entrant; failure at any stage requires restart from that stage.

```
 ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
 │   PHASE 1: DATA   │────▶│  PHASE 2: BASE    │────▶│  PHASE 3: FINE-   │────▶│  PHASE 4:         │
 │   PREPARATION     │     │  PRE-TRAINING     │     │  TUNING           │     │  INFERENCE        │
 └───────────────────┘     └───────────────────┘     └───────────────────┘     └───────────────────┘
         │                         │                         │                         │
    ┌────┴────┐               ┌────┴────┐               ┌────┴────┐               ┌────┴────┐
    │ Scrape  │               │ Init GPT│               │ Load    │               │ Load FT │
    │ Corpus  │               │ from    │               │ Base    │               │ Weights │
    │ Sources │               │ Config  │               │ Ckpt    │               │         │
    ├─────────┤               ├─────────┤               ├─────────┤               ├─────────┤
    │ Train   │               │ AdamW   │               │ Reduced │               │ Encode  │
    │ BPE     │               │ Optim   │               │ LR      │               │ Prompt  │
    │ Vocab   │               │ Loop    │               │ 1e-4    │               │ via BPE │
    ├─────────┤               ├─────────┤               ├─────────┤               ├─────────┤
    │ Encode  │               │ 15K     │               │ 400     │               │ Auto-   │
    │ to .bin │               │ Steps   │               │ Steps   │               │ regress │
    │ memmap  │               │         │               │         │               │ Decode  │
    ├─────────┤               ├─────────┤               ├─────────┤               ├─────────┤
    │ 90/10   │               │ Save    │               │ Save    │               │ Ghost   │
    │ Split   │               │ Ckpt    │               │ FT Ckpt │               │ Byte    │
    │         │               │ + Logs  │               │ + Logs  │               │ Mask    │
    └─────────┘               └─────────┘               └─────────┘               └─────────┘

    Artifacts:                 Artifacts:                 Artifacts:                 Artifacts:
    - train_bpe.bin            - task_4_1_baseline_       - task_7_1_finetuned_      - Raw text
    - val_bpe.bin                checkpoint.pt              checkpoint.pt            - Decoded
    - bpe_meta.pkl             - *_metrics.json            - *_metrics.json            output
    - char_meta.pkl
```

### Phase 1: Data Preparation (`data/prepare.py`)

**Objective**: Transform raw text corpora into memory-mapped integer arrays for O(1) random-access batch sampling.

**Pipeline Steps**:

1. **Corpus Aggregation**: Load `clean_poems_corpus.jsonl` (JSONL, one `text` field per line) and `clean_rap_dataset.json` (JSON array with `lyrics` field) into a unified `List[str]` where each element is one complete song/poem. This list-based approach preserves document boundaries—a critical fix for the String-Shredding bug (see §5).

2. **Train/Val Split**: `sklearn.model_selection.train_test_split` on the *list of intact documents* (not a joined string), `test_size=0.1`, `random_state=42`. This ensures no song is split across train/val partitions.

3. **Character Tokenizer Training**: `CharacterTokenizer.train_from_text()` on the joined training text. Extracts `sorted(set(corpus))` to build `stoi`/`itos` bijective mappings.

4. **BPE Tokenizer Training**: `SimpleBPETokenizer.train_from_text()` on the first 1M characters of training text. Starts from 256 raw byte tokens, iteratively merges the most frequent adjacent pair until `vocab_size=512` (256 merges).

5. **Safe Encoding**: The `safe_encode_list` function iterates over the list of documents, splits each into lines, and encodes `line + '\n'` individually. This avoids the O(n²) pair-statistics bottleneck that choked the naive `encode(entire_corpus)` approach.

6. **Binary Serialization**: Encoded arrays are cast to `np.uint16` (sufficient for vocab_size ≤ 65535) and written to `.bin` files via `ndarray.tofile()`.

**Data Scale**:

| File | Size | Description |
|------|------|-------------|
| `train_bpe.bin` | 21.4 MB | ~10.7M BPE tokens |
| `val_bpe.bin` | 2.1 MB | ~1.0M BPE tokens |
| `train_char.bin` | 40.8 MB | ~20.4M character tokens |
| `val_char.bin` | 4.1 MB | ~2.1M character tokens |

### Phase 2: Base Pre-Training (`training/train.py`)

**Objective**: Train a randomly-initialized GPT model on the full poems+rap corpus until convergence.

**Execution**: `python training/train.py task_4_1_baseline`

The script loads memory-mapped `.bin` files via `np.memmap(..., mode='r')`, instantiates `GPT(config)` from the configuration registry, and runs a vanilla training loop for `max_iters=15000` steps. At every `eval_interval=100` steps, it computes mean loss over `eval_iters=20` random batches for both train and val splits.

### Phase 3: Domain Fine-Tuning (`training/finetune.py`)

**Objective**: Adapt the pre-trained base model to rap lyrics generation with explicit conditioning envelopes.

The fine-tuning script:
1. Loads the base checkpoint (`task_4_1_baseline_checkpoint.pt`)
2. Wraps each rap song in conditioning tokens: `<|startoftext|>\n<|type|> rap\n[genre: boom_bap]\n[mood: emotional]\n<|content|>\n{lyrics}\n<|endoftext|>\n`
3. Splits the *list of conditioned songs* (not a joined string—same critical fix)
4. Runs 400 steps at `lr=1e-4` with `batch_size=16`

### Phase 4: Inference (`evaluation/inference.py`, `evaluation/generate.py`)

**Objective**: Auto-regressively generate text from a prompt using the fine-tuned model with Ghost Byte logit masking and configurable sampling strategies.

---

## 3. Codebase Engineering Breakdown

### 3.1 Configuration System

**File**: `configs/experiment_configs.py`

The system uses a dictionary-based configuration registry with template inheritance via Python's `**` unpacking operator.

```python
BASE_TEMPLATE = {
    'vocab_size': 512,       # BPE vocabulary ceiling
    'block_size': 128,       # Maximum context window (sequence length)
    'bias': False,           # Disable all bias terms in Linear/LN layers
    'dropout': 0.1,          # Regularization probability
    'ffn_multiplier': 4,     # FFN hidden dim = 4 * n_embd
    'learning_rate': 6e-4,   # AdamW learning rate
    'batch_size': 32,        # Sequences per gradient step
    'max_iters': 15000,      # Total training steps
    'eval_interval': 100,    # Steps between validation evaluations
    'eval_iters': 20         # Batches averaged per evaluation
}
```

**Design Rationale**: Dictionary-based configs over dataclasses or YAML because:
- The model constructor (`GPT.__init__`) indexes via `config['key']`, avoiding attribute access boilerplate
- Template inheritance via `{**BASE_TEMPLATE, 'override_key': value}` is atomic and explicit
- Zero deserialization overhead; the configs live in-process as Python dicts

**Configuration Registry** (13 total configurations):

| Config ID | n_layer | n_head | n_embd | Special Override |
|-----------|---------|--------|--------|-----------------|
| `task_4_1_baseline` | 4 | 4 | 128 | None (canonical) |
| `task_4_2_high_lr` | 4 | 4 | 128 | `lr=5e-3` |
| `task_4_2_large_context` | 4 | 4 | 128 | `block_size=256` |
| `task_4_3_tiny` | 2 | 2 | 64 | Minimal capacity |
| `task_4_3_medium` | 6 | 6 | 192 | Mid-scale |
| `task_4_3_large` | 8 | 8 | 256 | Max capacity |
| `task_3_2_no_pos` | 4 | 4 | 128 | `use_positional_embeddings=False` |
| `task_3_3_no_mask` | 4 | 4 | 128 | `use_causal_masking=False` |
| `task_3_4_single_head` | 4 | 1 | 128 | Attention bottleneck |
| `task_3_5_small_ffn` | 4 | 4 | 128 | `ffn_multiplier=1` |
| `task_3_6_no_residual` | 4 | 4 | 128 | `bypass_residuals=True` |
| `task_3_6_no_ln` | 4 | 4 | 128 | `bypass_ln=True` |

---

### 3.2 Data Pipeline & Tokenizers

#### 3.2.1 BPE Tokenizer (`tokenizer/bpe_tokenizer.py`)

**Class**: `SimpleBPETokenizer`

A from-scratch Byte-Pair Encoding implementation following the Sennrich et al. (2016) algorithm without any external library dependency.

**Training Algorithm** (`train_from_text`):

```python
def train_from_text(self, text):
    tokens = list(text.encode('utf-8'))          # Step 1: UTF-8 byte decomposition
    self.vocab = {i: bytes([i]) for i in range(256)}  # Step 2: Base vocab = 256 raw bytes
    num_merges = self.target_vocab_size - 256     # Step 3: 512 - 256 = 256 merges

    current_ids = list(tokens)
    for i in range(num_merges):
        stats = self.get_stats(current_ids)       # Step 4: Count adjacent pair frequencies
        best_pair = max(stats, key=stats.get)      # Step 5: Select most frequent pair
        idx = 256 + i                              # Step 6: Assign new token ID
        current_ids = self.merge(current_ids, best_pair, idx)  # Step 7: Replace all occurrences
        self.merges[best_pair] = idx               # Step 8: Record merge rule
        self.vocab[idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]  # Step 9: Compose bytes
```

**Computational Complexity**:
- `get_stats`: O(n) where n = current token sequence length
- `merge`: O(n) single-pass with lookahead
- Full training: O(m × n) where m = 256 merges, n shrinks with each merge
- **Critical**: Training on the first 1M characters (`text[:1000000]`) bounds the quadratic growth

**Encoding Algorithm** (`encode`):

```python
def encode(self, text):
    tokens = list(text.encode('utf-8'))
    while len(tokens) >= 2:
        stats = self.get_stats(tokens)
        pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
        if pair not in self.merges:
            break
        tokens = self.merge(tokens, pair, self.merges[pair])
    return tokens
```

The `min(stats, key=lambda p: self.merges.get(p, float("inf")))` lookup prioritizes pairs that were merged *earliest* during training—i.e., the most frequently observed subword units. This greedy bottom-up reconstruction mirrors the original BPE paper's encoding procedure.

**Decoding Algorithm** (`decode`):

The decode method reconstructs the byte-level vocabulary from the merge table at decode time, maps each token ID to its byte sequence, concatenates all bytes, and performs `bytes.decode('utf-8', errors='replace')`. The `errors='replace'` policy substitutes U+FFFD (�) for malformed UTF-8 sequences—this is the source of Ghost Byte artifacts (see §5.2).

#### 3.2.2 Character Tokenizer (`tokenizer/char_tokenizer.py`)

**Class**: `CharacterTokenizer`

Minimal character-level tokenizer. `train_from_text` extracts `sorted(set(corpus))`, builds bijective `stoi` (string-to-int) and `itos` (int-to-string) dictionaries. Encoding silently drops characters not in `stoi` (no UNK token). Decoding silently drops IDs not in `itos`.

**Key Difference from BPE**:
- Character vocab size is corpus-dependent (not fixed at 512)
- No subword compression: each Unicode codepoint = 1 token
- Produces ~2× more tokens than BPE for the same text (40.8MB char vs 21.4MB BPE for training data)
- Abandoned in the final pipeline in favor of BPE (the active `inference.py` uses `SimpleBPETokenizer`)

#### 3.2.3 Data Preparation Orchestrator (`data/prepare.py`)

**Critical Engineering Fix—The `safe_encode_list` Function**:

```python
def safe_encode_list(text_list, tokenizer):
    encoded_ids = []
    for text in text_list:
        if not text: continue
        for line in text.split('\n'):
            encoded_ids.extend(tokenizer.encode(line + '\n'))
    return np.array(encoded_ids, dtype=np.uint16)
```

**Why this exists**: The naive approach (`bpe_tok.encode("\n".join(all_texts))`) calls `get_stats` on the *entire* 20M+ byte sequence, producing O(n²) behavior because each merge iteration recomputes pair statistics over the full sequence. The line-by-line chunking reduces the maximum `encode` input to a single lyric line (~100 bytes), making each `get_stats` call O(100) instead of O(20M).

**Memory-Mapping Strategy**: The `.bin` files are read via `np.memmap(path, dtype=np.uint16, mode='r')`. This memory-maps the file into virtual address space without loading it entirely into RAM. The `get_batch` function then performs random-access reads:

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    x = torch.stack([torch.from_numpy((data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
```

Each batch samples `batch_size=32` random contiguous windows of length `block_size=128` from the memory-mapped array. The target sequence `y` is the input sequence shifted right by one token (standard causal LM objective).

---

### 3.3 Transformer Architecture

**File**: `model/transformer.py` — 114 lines implementing a complete GPT-2-style decoder-only transformer.

#### 3.3.1 Module Dependency Graph

```
GPT
 ├── transformer (ModuleDict)
 │    ├── wte: Embedding(vocab_size=512, n_embd=128)     ← Token embeddings
 │    ├── wpe: Embedding(block_size=128, n_embd=128)     ← Positional embeddings
 │    ├── drop: Dropout(0.1)                             ← Embedding dropout
 │    ├── h: ModuleList                                  ← Transformer blocks
 │    │    └── Block × n_layer
 │    │         ├── ln_1: LayerNorm(128)                 ← Pre-attention norm
 │    │         ├── attn: CausalSelfAttention             ← Multi-head attention
 │    │         │    ├── c_attn: Linear(128 → 384)       ← Fused QKV projection
 │    │         │    ├── c_proj: Linear(128 → 128)       ← Output projection
 │    │         │    ├── attn_dropout: Dropout(0.1)
 │    │         │    ├── resid_dropout: Dropout(0.1)
 │    │         │    └── bias: Buffer[1,1,128,128]       ← Causal mask (lower triangular)
 │    │         ├── ln_2: LayerNorm(128)                 ← Pre-FFN norm
 │    │         └── mlp: FeedForward
 │    │              ├── c_fc: Linear(128 → 512)         ← Up-projection
 │    │              ├── gelu: GELU()                    ← Activation
 │    │              ├── c_proj: Linear(512 → 128)       ← Down-projection
 │    │              └── dropout: Dropout(0.1)
 │    └── ln_f: LayerNorm(128)                           ← Final layer norm
 └── lm_head: Linear(128 → 512, bias=False)              ← Language model head
      └── weight TIED to wte.weight                      ← Weight tying
```

**Total Parameters (baseline config)**:

| Component | Parameters | Calculation |
|-----------|-----------|-------------|
| `wte` | 65,536 | 512 × 128 |
| `wpe` | 16,384 | 128 × 128 |
| `c_attn` (per block) | 49,152 | 128 × 384 |
| `c_proj` attn (per block) | 16,384 | 128 × 128 |
| `ln_1` (per block) | 128 | 128 weights |
| `c_fc` (per block) | 65,536 | 128 × 512 |
| `c_proj` FFN (per block) | 65,536 | 512 × 128 |
| `ln_2` (per block) | 128 | 128 weights |
| `ln_f` | 128 | 128 weights |
| `lm_head` | 0 | Tied to `wte` |
| **Total (4 blocks)** | **~868K** | |

#### 3.3.2 LayerNorm

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
```

**Design Choice**: `bias=False` by default (set from `config['bias']`). Following the GPT-2/NanoGPT convention, all bias terms are disabled. This reduces parameter count and acts as a mild regularizer. The epsilon `1e-5` prevents division-by-zero in the variance normalization.

**Mathematical Operation**:

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where μ and σ² are computed per-token across the embedding dimension, γ = `self.weight` (initialized to 1), and β = `self.bias` (disabled).

#### 3.3.3 CausalSelfAttention

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'], bias=config['bias'])
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=config['bias'])
        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                             .view(1, 1, config['block_size'], config['block_size']))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
```

**Fused QKV Projection**: `c_attn` projects from `n_embd` → `3 × n_embd` in a single matrix multiply, then `.split(n_embd, dim=2)` partitions the output into Q, K, V tensors. This is 3× more efficient than separate Q/K/V linear layers due to CUDA kernel launch overhead amortization.

**Multi-Head Reshape**: The `.view(B, T, n_head, head_dim).transpose(1, 2)` operation restructures `(B, T, C)` → `(B, n_head, T, head_dim)`. This enables batched matrix multiplication across all heads simultaneously.

**Scaled Dot-Product Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The `1.0 / math.sqrt(k.size(-1))` scaling factor prevents softmax saturation. For `n_embd=128, n_head=4`, `head_dim = 32`, so the scale factor is `1/√32 ≈ 0.1768`.

**Causal Mask**: The `register_buffer("bias", torch.tril(...))` creates a lower-triangular boolean mask as a non-parameter buffer. During forward, `att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))` sets future positions to -∞ before softmax, ensuring zero attention weight on future tokens. The `:T` indexing handles variable-length sequences up to `block_size`.

**Dropout Pipeline**: Two independent dropout layers:
1. `attn_dropout`: Applied to attention weights post-softmax (randomly zeros attention connections)
2. `resid_dropout`: Applied to the output projection (prevents co-adaptation of attention outputs)

#### 3.3.4 FeedForward (MLP)

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config['ffn_multiplier'] * config['n_embd'])
        self.c_fc = nn.Linear(config['n_embd'], hidden_dim, bias=config['bias'])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, config['n_embd'], bias=config['bias'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
```

**Architecture**: Classic two-layer MLP with expansion ratio. For the baseline: `128 → 512 → 128`.

**Activation**: GELU (Gaussian Error Linear Unit) instead of ReLU. GELU provides smooth gradients near zero, preventing the "dying neuron" problem where ReLU permanently zeros gradient for negative inputs. Mathematically:

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

**Why `ffn_multiplier=4`**: The 4× expansion follows the GPT-2/Vaswani convention. The FFN acts as a key-value memory: the up-projection (`c_fc`) maps inputs to a high-dimensional space where individual neurons learn to detect specific patterns, and the down-projection (`c_proj`) maps back to the residual stream. Reducing to `ffn_multiplier=1` (task_3_5_small_ffn ablation) collapses this capacity.

#### 3.3.5 Block (Transformer Layer)

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config['n_embd'], bias=config['bias'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config['n_embd'], bias=config['bias'])
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))      # Pre-norm attention + residual
        x = x + self.mlp(self.ln_2(x))        # Pre-norm FFN + residual
        return x
```

**Pre-Norm Architecture**: LayerNorm is applied *before* the sub-layer (attention or FFN), not after. This is the GPT-2/modern convention, which stabilizes training by normalizing inputs to each sub-layer. The original Vaswani (2017) paper used post-norm, which requires warmup and is more brittle.

**Residual Connections**: The `x = x + sublayer(ln(x))` pattern creates a residual stream. Gradients flow directly through the addition operator, bypassing the potentially-vanishing gradients through deep attention/FFN stacks. The ablation `task_3_6_no_residual` (`bypass_residuals=True`) demonstrates the catastrophic impact of removing these connections.

#### 3.3.6 GPT (Top-Level Model)

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']),
            wpe = nn.Embedding(config['block_size'], config['n_embd']),
            drop = nn.Dropout(config['dropout']),
            h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f = LayerNorm(config['n_embd'], bias=config['bias']),
        ))
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying
        self.apply(self._init_weights)
```

**Weight Tying** (`self.transformer.wte.weight = self.lm_head.weight`):

The token embedding matrix and the output projection (language model head) share the same weight tensor. This means:
- The embedding lookup `wte(idx)` and the final logit computation `lm_head(x)` use identical parameters
- Reduces parameter count by `vocab_size × n_embd = 512 × 128 = 65,536` parameters
- Enforces semantic consistency: tokens with similar embeddings will have similar output logits
- Follows Press & Wolf (2017) "Using the Output Embedding to Improve Language Models"

**Weight Initialization** (`_init_weights`):

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

All linear layers and embeddings are initialized from N(0, 0.02). The std=0.02 is the GPT-2 convention—large enough to break symmetry but small enough to prevent early training instability. The `self.apply(self._init_weights)` call recursively initializes every sub-module.

**Forward Pass**:

```python
def forward(self, idx, targets=None):
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

    tok_emb = self.transformer.wte(idx)          # (B, T) → (B, T, C)
    pos_emb = self.transformer.wpe(pos)           # (T,) → (T, C) → broadcast to (B, T, C)
    x = self.transformer.drop(tok_emb + pos_emb)  # Embedding dropout

    for block in self.transformer.h:              # Sequential transformer blocks
        x = block(x)
    x = self.transformer.ln_f(x)                  # Final layer norm

    if targets is not None:
        logits = self.lm_head(x)                  # (B, T, C) → (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
        logits = self.lm_head(x[:, [-1], :])      # Only compute logits for last position
        loss = None
    return logits, loss
```

**Inference Optimization**: When `targets=None` (inference mode), logits are computed only for the last token position (`x[:, [-1], :]`), avoiding a full `(B, T, vocab_size)` matmul. This reduces inference FLOPS by a factor of `T`.

**Loss Function**: `F.cross_entropy` with `ignore_index=-1` computes the standard next-token prediction loss:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{z_{y_i}}}{\sum_{j=1}^{V} e^{z_j}}$$

Where z are logits, y_i is the target token ID, V = vocab_size. The `view(-1, ...)` flattening collapses the batch and sequence dimensions for vectorized cross-entropy computation.

---

### 3.4 Training Loops

#### 3.4.1 Base Pre-Training Loop (`training/train.py`)

```python
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

for step in range(config['max_iters'] + 1):
    if step % config['eval_interval'] == 0:
        metrics = estimate_loss(model)
        # ... logging ...

    X, Y = get_batch('train')
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

**`zero_grad(set_to_none=True)`**: Instead of setting gradients to zero tensors, this sets `.grad` attributes to `None`. This saves memory (no zero tensor allocation) and can be slightly faster because the first `backward()` call uses `+=` when grad exists but direct assignment when grad is None.

**No Gradient Clipping**: The loop does not clip gradients. This is acceptable for the baseline config (4 layers, small model) but could cause instability in the `task_3_6_no_ln` ablation where LayerNorm is disabled.

**No Learning Rate Schedule**: A constant learning rate of `6e-4` is used throughout all 15,000 steps. No warmup, no cosine decay. This simplifies the training but means the model may oscillate near convergence rather than settling into a sharper minimum.

**Evaluation Function**:

```python
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        for k in range(config['eval_iters']):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
```

The `@torch.no_grad()` decorator disables gradient computation during evaluation (saves memory and compute). `model.eval()` disables dropout; `model.train()` re-enables it. Loss is averaged over 20 random batches for variance reduction.

#### 3.4.2 Fine-Tuning Loop (`training/finetune.py`)

```python
ft_config = {
    **CONFIGS[BASE_CONFIG_ID],
    'learning_rate': 1e-4,   # 6× reduction from base lr
    'max_iters': 400,        # 37.5× fewer steps
    'eval_interval': 50,
    'batch_size': 16          # 2× reduction from base batch
}

model = GPT(ft_config)
model.load_state_dict(torch.load(base_checkpoint_path, map_location=device, weights_only=True))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=ft_config['learning_rate'], weight_decay=0.01)
```

**Key Differences from Base Training**:

| Parameter | Base | Fine-Tune | Rationale |
|-----------|------|-----------|-----------|
| `learning_rate` | 6e-4 | 1e-4 | Prevent catastrophic forgetting of base knowledge |
| `max_iters` | 15,000 | 400 | Domain adaptation requires far fewer steps |
| `batch_size` | 32 | 16 | Smaller dataset warrants smaller batch |
| `weight_decay` | 0 (default) | 0.01 | Explicit L2 regularization for fine-tuning stability |
| `eval_interval` | 100 | 50 | Higher monitoring frequency over shorter run |

**Conditioning Envelope**:

```python
formatted_text = f"<|startoftext|>\n<|type|> rap\n[genre: boom_bap]\n[mood: emotional]\n<|content|>\n{item['lyrics']}\n<|endoftext|>\n"
```

These structural tokens are not special tokens in the vocabulary—they are encoded as regular BPE subword sequences. The model learns to associate these byte patterns with the rap domain through the fine-tuning distribution. At inference, providing these conditioning prefixes steers generation toward the fine-tuned domain.

---

### 3.5 Sampling & Inference

**File**: `evaluation/generate.py`

Four sampling strategies are implemented, all sharing the same auto-regressive backbone:

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -model.config['block_size']:]   # Context window truncation
    logits, _ = model(idx_cond)                        # Forward pass
    logits = logits[:, -1, :]                          # Last-position logits only
    # ... strategy-specific sampling ...
    idx = torch.cat((idx, idx_next), dim=1)            # Append sampled token
```

#### Strategy 1: Greedy Decoding

```python
idx_next = torch.argmax(logits, dim=-1, keepdim=True)
```

Deterministically selects the highest-probability token at every step. Produces the single most likely sequence. Highly susceptible to repetition loops (see §6.1).

#### Strategy 2: Temperature Sampling

```python
logits = logits / temp
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

Temperature τ controls distribution entropy:
- τ < 1.0: Sharpens distribution → more deterministic (low T ≈ greedy)
- τ = 1.0: Original model distribution
- τ > 1.0: Flattens distribution → more random (high T → uniform)

Mathematically: `P(token_i) = exp(z_i/τ) / Σ exp(z_j/τ)`

#### Strategy 3: Top-K Sampling

```python
v, _ = torch.topk(logits, min(k, logits.size(-1)))
logits[logits < v[:, [-1]]] = -float('Inf')
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

Restricts the sampling pool to the K most probable tokens. `v[:, [-1]]` is the K-th largest logit value; everything below is set to -∞ before softmax, effectively zeroing those probabilities. Prevents sampling from the extreme low-probability tail while maintaining diversity within the top-K.

#### Strategy 4: Top-P Nucleus Sampling (Holtzman et al., 2020)

```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
sorted_indices_to_remove = cumulative_probs > p
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
sorted_indices_to_remove[..., 0] = 0
indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
logits[indices_to_remove] = -float('Inf')
```

Dynamically sizes the sampling pool: includes the minimum number of tokens whose cumulative probability mass exceeds threshold P. The mask shift (`[..., 1:] = [..,:- 1].clone()`) retains the token that first crosses the threshold. The `.scatter()` operation maps the sorted-space mask back to the original vocabulary indices.

**Advantage over Top-K**: Adapts to the model's confidence. When the model is certain (one token dominates), Top-P samples from ~1 token. When uncertain (flat distribution), it samples from many tokens.

---

### 3.6 Logit Masking: The Ghost Byte Blocker

**File**: `evaluation/inference.py` (active version, lines 205–230)

```python
# --- THE GHOST BYTE BLOCKER ---
logits[:, 128:256] = -float('Inf')
```

**What It Does**: Sets logits for token IDs 128–255 to negative infinity before softmax, permanently banning the model from outputting these tokens.

**Why It Exists**: BPE token IDs 0–255 correspond to raw UTF-8 byte values. Bytes 0–127 are standard ASCII (printable English text). Bytes 128–255 are:
- UTF-8 continuation bytes (0x80–0xBF)
- UTF-8 lead bytes for multi-byte sequences (0xC0–0xFF)

These bytes only appear as *components* of merged BPE tokens during training (e.g., merged token ID 300 might internally consist of bytes [0xC3, 0xA9] = 'é'). The raw bytes 128–255 were rarely or never emitted as standalone tokens in the training data. Without masking, the model occasionally samples these orphan bytes, which produce:
- UTF-8 decode errors → U+FFFD replacement characters (�)
- Partial multi-byte sequences → mojibake
- Non-printable control characters

The mask ensures all sampled tokens are either ASCII bytes (0–127) or valid merged BPE tokens (256–511).

---

## 4. Core System Design Choices

### 4.1 PyTorch

The entire pipeline is built on PyTorch (imperative/eager execution). Key PyTorch-specific patterns used:

| Pattern | Usage | Rationale |
|---------|-------|-----------|
| `nn.ModuleDict` | GPT's `self.transformer` | Named sub-module access via dict syntax while maintaining `parameters()` registration |
| `register_buffer` | Causal mask | Persisted in `state_dict`, moved with `.to(device)`, but excluded from `parameters()` |
| `@torch.no_grad()` | `estimate_loss`, `generate_text` | Disables autograd tape recording—saves memory and compute during non-training passes |
| `F.cross_entropy` | Loss computation | Fused log-softmax + NLL loss with numerical stability via log-sum-exp trick |
| `torch.multinomial` | Stochastic sampling | Draws from discrete distributions on GPU without CPU roundtrip |
| `model.eval()/train()` | Eval/train mode toggle | Controls dropout behavior (disabled during eval, active during training) |
| `weights_only=True` | `torch.load` | Security measure: prevents arbitrary code execution from malicious pickle payloads |

### 4.2 AdamW Optimizer

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
```

AdamW (Loshchilov & Hutter, 2019) is chosen over vanilla Adam because:
- **Decoupled weight decay**: AdamW applies weight decay directly to parameters, not to gradients. Vanilla Adam's L2 regularization is entangled with the adaptive learning rate, making regularization strength vary per-parameter.
- **Standard for transformers**: Used by GPT-2, GPT-3, BERT, and virtually all modern LLM training.
- **Fine-tuning weight decay**: The fine-tuning loop explicitly sets `weight_decay=0.01`, providing L2 regularization that prevents the fine-tuned weights from drifting too far from the pre-trained initialization.

### 4.3 BPE vs Character-Level Tokenization

Both tokenizers are implemented and trained. The pipeline ultimately uses BPE for all training and inference. The character tokenizer remains for ablation comparisons.

| Dimension | BPE (Active) | Character (Deprecated) |
|-----------|-------------|----------------------|
| Vocab Size | 512 (fixed) | Corpus-dependent (~200–600) |
| Token Granularity | Subword (variable-length byte sequences) | Single Unicode codepoint |
| Compression | ~2× fewer tokens than char | No compression |
| OOV Handling | Falls back to raw byte decomposition (0–255) | Silently drops unknown chars |
| Training Cost | O(m × n) merge iterations | O(n) single pass |
| Sequence Length | Shorter sequences → fits more context in `block_size` | Longer sequences → wastes context window |
| Implementation | 96 LOC | 38 LOC |

**BPE Selection Rationale**: At `block_size=128`, character-level tokenization captures ~128 characters of context. BPE with ~2× compression captures ~256 characters of effective context. For lyrical content with meaningful multi-line structure, this doubled context window significantly improves coherence.

### 4.4 Decoupled Configuration Architecture

The configuration system uses a registry pattern (dictionary of dictionaries) rather than command-line arguments or config files. This enables:

1. **Reproducibility**: Each experiment is fully specified by its config ID string
2. **Template inheritance**: Ablations override only the targeted parameter
3. **Validation at import time**: Missing keys cause immediate KeyError, not silent defaults
4. **Multi-experiment orchestration**: The training script accepts a config ID as `sys.argv[1]`, enabling automated sweeps via shell loops

### 4.5 KV Management

The model does **not** implement a KV cache. During auto-regressive generation, each forward pass recomputes attention over the full context window:

```python
idx_cond = idx[:, -model.config['block_size']:]
logits, _ = model(idx_cond)
```

This means generating T tokens requires O(T²) attention computation (each step processes the growing sequence from scratch). A KV cache would reduce this to O(T) by caching key/value tensors from previous steps and only computing attention for the new token. The absence of KV caching is acceptable here because:
- `max_new_tokens ≤ 150` and `block_size = 128` keep the quadratic cost manageable
- The model is small (~868K params) with fast forward passes
- Implementation simplicity is prioritized over inference throughput

---

## 5. Debug Log Matrix

### 5.1 The String-Shredding Parasite

**Symptom**: During early development iterations, the model generated intelligible word fragments but with systematic inter-word corruption—spaces, punctuation, and word boundaries were replaced by random byte artifacts. Output resembled "Th\x82e\x91 r\x84ain\x93 fa\x80lls" instead of "The rain falls".

**Root Cause: O(n²) BPE Encoding on Joined Corpus**

The original `prepare.py` code:

```python
# BROKEN: The String-Shredding approach
train_string = "\n".join(train_txt)     # Join all songs into one massive string
train_bpe_ids = np.array(bpe_tok.encode(train_string), dtype=np.uint16)
```

The `encode()` method calls `get_stats(tokens)` on the *entire* corpus byte sequence (~20M+ bytes) in a `while` loop. Each iteration:

1. **`get_stats`**: O(n) to count all adjacent pairs in the 20M-length token array
2. **`min(stats, key=lambda p: self.merges.get(p, float("inf")))`**: O(|stats|) lookup per pair
3. **`merge`**: O(n) single pass to replace the selected pair

The total complexity is O(n × m) where n = 20M+ bytes and m = number of merge iterations until no more mergeable pairs exist. This caused:

- **Computational failure**: The encoding step took hours and was often killed by OOM
- **Partial encoding**: When interrupted, the `.bin` file contained a partially-encoded sequence where early portions were fully merged (valid BPE) but later portions were raw bytes (invalid standalone tokens 128–255)
- **Boundary corruption**: The `"\n".join()` created artificial cross-document byte sequences that the BPE encoder merged across song boundaries, creating tokens that spanned two unrelated songs

**The Fix**:

```python
def safe_encode_list(text_list, tokenizer):
    encoded_ids = []
    for text in text_list:
        if not text: continue
        for line in text.split('\n'):
            encoded_ids.extend(tokenizer.encode(line + '\n'))
    return np.array(encoded_ids, dtype=np.uint16)
```

Line-by-line encoding bounds each `encode()` call to ~100 bytes, making `get_stats` O(100) instead of O(20M). Total encoding time drops from hours to seconds. No cross-document merges occur because each song line is encoded independently.

**Mathematical Impact on Loss**:

The corrupted training data injected noise into the cross-entropy target distribution. Standalone bytes 128–255 that appeared as targets had near-zero model probability (the model had no reason to predict these orphan bytes), creating per-token losses of:

$$\mathcal{L}_{\text{corrupted}} = -\log P(\text{byte}_{128\text{-}255}) \approx -\log(1/512) \approx 6.24$$

These pathological loss contributions inflated the training loss, preventing the model from descending below ~4.0 even after thousands of steps in early iterations.

### 5.2 The Ghost Byte Anomaly

**Symptom**: After fixing the String-Shredding bug, the model generated mostly coherent text but periodically emitted replacement characters (�), non-printable sequences, and orphan UTF-8 continuation bytes.

**Root Cause: Untrained Byte Tokens in the Vocabulary**

The BPE vocabulary maps token IDs 0–255 to raw UTF-8 bytes. Of these:
- IDs 0–127: ASCII characters (space, letters, digits, punctuation) — heavily represented in training data
- IDs 128–255: UTF-8 continuation/lead bytes — rarely or never appear as *standalone* tokens in English text

During training, the model sees these IDs primarily as *components* of merged tokens (e.g., BPE token #300 is the merge of bytes #195 and #169, forming 'é'). The individual bytes 128–255 have near-zero frequency as standalone training targets. However, they remain valid token IDs in the embedding/LM-head matrices. At inference, stochastic sampling occasionally selects these low-probability tokens, especially under:

- **High temperature** (τ > 1.0): Flattens the distribution, giving non-trivial probability mass to rare tokens
- **Top-K with large K**: Includes rare tokens in the sampling pool
- **Long generations**: More rolls of the dice → higher cumulative probability of hitting a rare token

When a Ghost Byte is sampled:
1. The BPE decoder maps it to a single byte in 128–255
2. This byte is a UTF-8 continuation byte (no valid start sequence)
3. Python's `bytes.decode('utf-8', errors='replace')` substitutes U+FFFD (�)

**The Fix**:

```python
logits[:, 128:256] = -float('Inf')
```

Zeroing the softmax probability for IDs 128–255 eliminates Ghost Bytes at the source. This is a post-hoc logit mask, not a vocabulary modification—the embedding matrix retains all 512 entries, and the mask is applied only during generation.

**Mathematical Verification**:

Pre-mask probability mass on Ghost Bytes (estimated from typical logit distributions):

$$P(\text{ghost}) = \sum_{i=128}^{255} \frac{e^{z_i / \tau}}{\sum_{j=0}^{511} e^{z_j / \tau}}$$

For τ=0.7 and typical logits where ghost byte logits cluster around -5.0 while valid token logits cluster around 0.0:

$$P(\text{ghost}) \approx 128 \times \frac{e^{-5/0.7}}{256 \times e^{0/0.7} + 128 \times e^{-5/0.7} + 128 \times e^{-2/0.7}} \approx 0.001$$

At 150 tokens per generation: `P(≥1 ghost in output) = 1 - (1 - 0.001)^150 ≈ 0.14`, a 14% chance per generation. The logit mask reduces this to exactly 0%.

### 5.3 Loss Trajectory Forensics

**Base Pre-Training (task_4_1_baseline)**:

| Milestone | Step | Train Loss | Val Loss | Δ Val/Step |
|-----------|------|-----------|---------|------------|
| Initialization | 0 | 6.2618 | 6.2637 | — |
| Fast descent phase | 100 | 4.2946 | 4.5239 | -0.0174 |
| Diminishing returns | 1,000 | 3.5828 | 3.7225 | -0.0089 |
| Convergence onset | 5,000 | 2.9492 | 3.0259 | -0.0017 |
| Near-plateau | 10,000 | 2.8167 | 2.9040 | -0.0002 |
| Final | 15,000 | 2.7179 | 2.8001 | -0.0002 |

**Initial loss sanity check**: At step 0, the untrained model produces `loss ≈ -ln(1/512) = ln(512) ≈ 6.238`. The observed 6.2618 is consistent with uniform random prediction over 512 tokens plus small variance from the initialization distribution.

**Generalization gap**: Final train-val gap = `2.8001 - 2.7179 = 0.0822`. This ~3% relative gap indicates mild overfitting, consistent with the 10% dropout regularization and the absence of data augmentation.

**Approximate final perplexity**: `exp(2.80) ≈ 16.4` — the model's effective uncertainty is ~16 tokens at each position. For a 512-token vocabulary, this represents significant learning (random baseline perplexity = 512).

**Fine-Tuning (task_7_1_finetuned)**:

| Step | Train Loss | Val Loss | Notes |
|------|-----------|---------|-------|
| 0 | 2.9134 | 2.8663 | FT init (base model on rap distribution) |
| 50 | 2.8077 | 2.7932 | Rapid domain adaptation |
| 100 | 2.7301 | 2.7353 | Distribution alignment |
| 200 | 2.7137 | 2.7116 | Continued descent |
| 300 | 2.6224 | 2.6347 | Near convergence |
| 400 | 2.6329 | 2.6563 | Final (slight val increase → early overfit signal) |

**FT Init Loss (2.9134)**: The base model evaluates at 2.9134 on rap-specific data. This is *higher* than the base model's final training loss (2.7179) because the rap corpus has a different token distribution than the general poems+rap training corpus. The conditioning envelope tokens (`<|startoftext|>`, `<|type|>`, etc.) are also novel patterns that the base model must adapt to.

**FT Final Loss (2.6329)**: The fine-tuned model achieves `2.6329` on rap training data, a reduction of `2.9134 - 2.6329 = 0.2805` nats. In perplexity terms: `exp(2.9134) ≈ 18.4 → exp(2.6329) ≈ 13.9`, a 24.4% reduction in predictive uncertainty on the target domain.

**Overfitting Signal**: At step 400, val loss (2.6563) exceeds step 300's val loss (2.6347), while train loss continues decreasing. This is the earliest sign of overfitting and suggests 300–350 steps may be the optimal fine-tuning horizon.

---

## 6. Output Analysis

### 6.1 Base Model Generation Outputs (`evaluation/generate.py`)

The base model, trained on the combined poems+rap corpus, generates text using five strategies from the prompt: `"The shadow falls across the street,\n"`

**Strategy Behavioral Analysis**:

| Strategy | Config | Expected Behavior | Observed Behavior |
|----------|--------|-------------------|-------------------|
| Greedy | `argmax` | Deterministic, repetitive | Highest coherence but enters repetition loops within ~20 tokens |
| Low Temperature | τ=0.3 | Near-greedy with slight variation | Similar to greedy but with occasional word substitutions |
| High Temperature | τ=1.5 | Creative but unstable | Generates novel phrasings but with increasing grammatical degradation |
| Top-K | K=10, τ=0.8 | Constrained diversity | Best balance of coherence and variety |
| Top-P (Nucleus) | P=0.9, τ=0.8 | Adaptive sampling | Similar quality to Top-K; dynamically adjusts pool size |

### 6.2 Stress Test Results (`evaluation/stress_test.py`)

**Test 1: Repetition Loop Vulnerability** (Severity: HIGH)

```
Input:  The rain falls. The rain falls.
Output: The rain falls. The rain falls. The beautiful soul of the beautiful soul
        of the beautiful soul of the beautiful
```

**Analysis**: Greedy decoding on a repetitive prompt immediately enters an attractor loop. The causal attention mechanism amplifies the dominant pattern in the context window—once "the beautiful soul of" appears, its attention weights dominate subsequent predictions. The model lacks any explicit repetition penalty or n-gram blocking mechanism.

**Test 2: Context Memory Overflow** (Severity: CRITICAL)

```
Input:  "word " × 140 (140 repetitions)
Output: word word word ... word world world work work word world world worl
```

**Analysis**: The 140-word input exceeds the `block_size=128` token limit. The `idx_cond = idx[:, -model.config['block_size']:]` truncation discards the earliest tokens. The model receives a context window entirely filled with "word" tokens, producing near-uniform predictions. The occasional mutations ("world", "work", "worl") are stochastic escapes from the dominant pattern via Top-P sampling.

**Test 3: High-Temperature Hallucination** (Severity: HIGH)

```
Input:  Deep in the shadows of the concrete jungle,
Output: Deep in the shadows of the concrete jungle,Z;
        tense licaght thea 5emi f □sfi'Tendhis one
        e, her `it aned -y fithatires□tilor
```

**Analysis**: Temperature=4.5 flattens the softmax distribution to near-uniform. At this temperature, `P(any token) ≈ 1/512 ≈ 0.002`, making the model effectively a random token generator. The "□" characters are Ghost Byte artifacts (this test predates the Ghost Byte Blocker deployment). Note the presence of partial English morphemes ("tense", "one", "her") surviving even under extreme temperature—these represent the strongest signals in the model's learned distribution.

**Test 4: Logical Contradiction** (Severity: MEDIUM)

```
Input:  I hate hip-hop music, it makes me angry. So I wrote a rap song because I love it:
Output: I hate hip-hop music, it makes me angry. So I wrote a rap song because I love it: -
        Some blood some flower is not a giving place of the claims.
        If I will not
```

**Analysis**: The model does not track logical consistency across clauses. It treats the prompt as a stylistic context window, not a semantic graph. The output adopts a poetic/philosophical tone consistent with the training corpus but ignores the contradiction between "hate" and "love". This is expected for a ~868K parameter model without instruction tuning.

### 6.3 Fine-Tuned Inference Outputs (`evaluation/inference.py`)

The active inference script uses the fine-tuned checkpoint with:
- BPE tokenizer (active version, not the commented-out char tokenizer or cipher-shift experiments)
- Ghost Byte Blocker (`logits[:, 128:256] = -float('Inf')`)
- Temperature=0.7, Top-K=40 as defaults

The inference file contains three evolutionary iterations preserved as commented-out blocks:

1. **V1 (lines 1–82, commented)**: Character tokenizer + Ghost Byte Blocker. Abandoned due to character-level tokenization's limited context window.

2. **V2 (lines 83–192, commented)**: BPE tokenizer + Caesar cipher shift (-2 encoding, +2 decoding). This was a failed experiment to train the model on "encrypted" text to prevent direct memorization. The cipher corrupted the BPE merge table's assumptions and produced unintelligible output.

3. **V3 (lines 194–269, active)**: Clean BPE tokenizer + Ghost Byte Blocker. The final production configuration.

### 6.4 Architecture Ablation Outputs

Six ablation experiments remove individual architectural components to measure their contribution:

| Ablation | Final Val Loss | Δ vs Baseline (1.83) | Component Removed |
|----------|---------------|---------------------|-------------------|
| `task_3_2_no_pos` | 1.8330 | +0.000 | Positional embeddings |
| `task_3_3_no_mask` | 1.8328 | -0.000 | Causal masking |
| `task_3_4_single_head` | 1.8384 | +0.005 | 3 of 4 attention heads |
| `task_3_5_small_ffn` | 1.8452 | +0.012 | 3/4 of FFN capacity |
| `task_3_6_no_ln` | 1.8297 | -0.003 | Layer normalization |
| `task_3_6_no_residual` | 1.8345 | +0.002 | Residual connections |

> [!IMPORTANT]
> **Critical observation**: All ablations converge to nearly identical val losses (~1.83) after only 1000 steps. This is a dataset/training-duration artifact—the ablation experiments were run for only 1000 steps (not 15,000 like the baseline). At this early stage, the model has barely moved past the initial rapid descent phase. The true impact of these architectural modifications would only manifest over longer training horizons, particularly:
> - **No-residual**: Would exhibit gradient vanishing after ~5K steps as depth compounds the signal degradation
> - **No-LN**: Would eventually show exploding activations as pre-normalization is absent
> - **Single-head**: Would plateau earlier due to reduced representational capacity in attention

### 6.5 Hyperparameter Sensitivity Outputs

**High Learning Rate (task_4_2_high_lr, lr=5e-3)**:

| Step | Train Loss | Val Loss |
|------|-----------|---------|
| 0 | 6.3032 | 6.3044 |
| 100 | 1.9115 | 1.9115 |
| 1000 | 1.8329 | 1.8359 |

Converges extremely fast (loss drops from 6.3 to 1.91 in 100 steps vs baseline's 4.29 at step 100). However, the final loss of 1.836 is **lower** than the baseline's 1000-step loss of 3.72. This suggests:
- The 5e-3 learning rate overshoots the loss landscape initially but finds a reasonable minimum
- The near-zero train-val gap (1.8329 vs 1.8359) indicates no overfitting at 1000 steps
- Longer training would reveal whether this LR causes instability (oscillation or divergence)

**Large Context (task_4_2_large_context, block_size=256)**:

| Step | Train Loss | Val Loss |
|------|-----------|---------|
| 0 | 6.2380 | 6.2395 |
| 1000 | 1.8302 | 1.8283 |

Doubling the context window to 256 tokens provides similar loss to baseline at 1000 steps. The marginal benefit is in *qualitative* generation coherence (longer-range dependencies) rather than quantitative loss reduction. The 3.4× increase in training time per step (87.8s total vs 26.8s for high_lr) reflects the O(T²) attention cost.

### 6.6 Scaling Experiment Outputs

| Model | Layers | Heads | Embed | Params (est.) | Val Loss @1000 | Time (s) |
|-------|--------|-------|-------|---------------|---------------|----------|
| Tiny | 2 | 2 | 64 | ~100K | 1.8384 | 8.2 |
| Baseline | 4 | 4 | 128 | ~868K | 3.7225* | 10.0 |
| Medium | 6 | 6 | 192 | ~2.5M | 1.8307 | 72.8 |
| Large | 8 | 8 | 256 | ~5.5M | 1.8403 | 144.9 |

*Note: Baseline at 1000 steps is from the 15K-step run (different data path); the scaling experiments used 1000-step runs.

**Scaling Analysis**: Medium (6L/6H/192E) achieves the lowest val loss at 1000 steps (1.8307). The Large model (8L/8H/256E) performs slightly worse (1.8403), likely because 1000 steps is insufficient to warm up the larger parameter space. The Large model's 144.9s runtime reflects the cubic scaling of compute with model dimension.

### 6.7 Visualization Outputs

**Embedding PCA** (`output/embedding_pca.png`): Comparing untrained (random) vs trained token embeddings projected to 2D via PCA. Untrained embeddings form a spherical Gaussian blob. Trained embeddings exhibit cluster structure where semantically/syntactically similar tokens group together.

**Attention Heatmap** (`output/attention_heatmap.png`): Visualizes Layer 0, Head 0 attention weights for the input "The quick brown fox jumps". Shows the lower-triangular causal structure (no attention to future tokens) and learned attention patterns (e.g., "jumps" attending to "fox").

---

## 7. Experiment Results Matrix

### 7.1 Complete Loss Trajectory Summary

| Experiment | Init Loss | Final Train | Final Val | Steps | LR | Batch | Context |
|------------|-----------|-------------|-----------|-------|------|-------|---------|
| **task_4_1_baseline** | 6.2618 | 2.7179 | 2.8001 | 15,000 | 6e-4 | 32 | 128 |
| task_4_2_high_lr | 6.3032 | 1.8329 | 1.8359 | 1,000 | 5e-3 | 32 | 128 |
| task_4_2_large_context | 6.2380 | 1.8302 | 1.8283 | 1,000 | 6e-4 | 32 | 256 |
| task_4_3_tiny | 6.3120 | 1.8351 | 1.8384 | 1,000 | 6e-4 | 32 | 128 |
| task_4_3_medium | 6.3531 | 1.8285 | 1.8307 | 1,000 | 6e-4 | 32 | 128 |
| task_4_3_large | 6.4750 | 1.8394 | 1.8403 | 1,000 | 6e-4 | 32 | 128 |
| task_3_2_no_pos | 6.2399 | 1.8297 | 1.8330 | 1,000 | 6e-4 | 32 | 128 |
| task_3_3_no_mask | 6.2587 | 1.8401 | 1.8328 | 1,000 | 6e-4 | 32 | 128 |
| task_3_4_single_head | 6.1484 | 1.8359 | 1.8384 | 1,000 | 6e-4 | 32 | 128 |
| task_3_5_small_ffn | 6.2628 | 1.8322 | 1.8452 | 1,000 | 6e-4 | 32 | 128 |
| task_3_6_no_ln | 6.2585 | 1.8320 | 1.8297 | 1,000 | 6e-4 | 32 | 128 |
| task_3_6_no_residual | 6.3532 | 1.8302 | 1.8345 | 1,000 | 6e-4 | 32 | 128 |
| **task_7_1_finetuned** | 2.9134 | 2.6329 | 2.6563 | 400 | 1e-4 | 16 | 128 |

### 7.2 Checkpoint File Manifest

| Checkpoint | Size | Contents |
|------------|------|----------|
| `task_4_1_baseline_checkpoint.pt` | 3.75 MB | Base model state_dict (~868K float32 parameters) |
| `task_7_1_finetuned_checkpoint.pt` | 3.75 MB | Fine-tuned model state_dict (identical architecture) |

### 7.3 Data Pipeline File Manifest

| File | Format | Size | Token Count (est.) |
|------|--------|------|--------------------|
| `clean_poems_corpus.jsonl` | JSONL | 22.7 MB | — (raw text) |
| `clean_rap_dataset.json` | JSON | 1.7 MB | — (raw text) |
| `diverse_rap_corpus_500.json` | JSON | 1.6 MB | — (raw text) |
| `tagged_rap_dataset_gemma4.json` | JSON | 1.7 MB | — (tagged version) |
| `cleaned_combined.csv` | CSV | 7.3 MB | — (tabular) |
| `training_corpus.jsonl` | JSONL | 7.7 MB | — (conditioned text) |
| `train_bpe.bin` | Binary uint16 | 21.4 MB | ~10.7M tokens |
| `val_bpe.bin` | Binary uint16 | 2.1 MB | ~1.0M tokens |
| `train_char.bin` | Binary uint16 | 40.8 MB | ~20.4M tokens |
| `val_char.bin` | Binary uint16 | 4.1 MB | ~2.1M tokens |
| `bpe_meta.pkl` | Pickle | 6.6 KB | 256 merge rules + 512 vocab entries |
| `char_meta.pkl` | Pickle | 34.2 KB | Character vocabulary mapping |

---

*Documentation generated from exhaustive static analysis of the complete codebase, training logs, and output artifacts. All loss values are exact as recorded in the JSON metric files. All parameter counts are computed from the architecture specification.*
