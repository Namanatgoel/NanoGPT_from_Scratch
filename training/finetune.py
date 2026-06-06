import os
import sys
import time
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Clean path alignment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import GPT
from configs.experiment_configs import CONFIGS
from tokenizer.bpe_tokenizer import SimpleBPETokenizer

print("=========================================")
print("Initializing Domain Fine-Tuning Pipeline")
print("=========================================")

# 1. Hardware and Environment Configurations
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_CONFIG_ID = 'task_4_1_baseline'
FINETUNE_CONFIG_ID = 'task_7_1_finetuned'

# Setup dedicated fine-tuning configurations inline
ft_config = {
    **CONFIGS[BASE_CONFIG_ID],
    'learning_rate': 1e-4,   # Drastically reduced to prevent catastrophic forgetting
    'max_iters': 400,        # Fewer steps are needed for domain adaptation
    'eval_interval': 50,
    'batch_size': 16         # Smaller localized batch optimization footprint
}

# 2. Load Tokenizer and Clean Rap Corpus
tokenizer = SimpleBPETokenizer()
tokenizer.load('tokenizer/bpe_meta.pkl')

if not os.path.exists('data/clean_rap_dataset.json'):
    print("Error: Target rap dataset is missing from data/ directory.")
    sys.exit(1)

with open('data/clean_rap_dataset.json', 'r', encoding='utf-8') as f:
    rap_data = json.load(f)
'''
# 3. Apply Explicit Conditioning Envelopes
print("Processing text data with target conditioning prefixes...")
conditioned_texts = []
for item in rap_data:
    # Build explicit token tracking architecture
    formatted_text = f"<|startoftext|>\n<|type|> rap\n<|content|>\n{item['lyrics']}\n<|endoftext|>\n"
    conditioned_texts.append(formatted_text)

# Consolidate and execute localized domain split
full_rap_corpus = "".join(conditioned_texts)
train_text, val_text = train_test_split(full_rap_corpus, test_size=0.1, random_state=42)

# Flatten the lists of strings into a single continuous string block for BPE tokenization
train_string = "\n".join(train_text)
val_string = "\n".join(val_text)

# Convert strings directly into dynamic token arrays
train_ids = np.array(tokenizer.encode(train_string), dtype=np.uint16)
val_ids = np.array(tokenizer.encode(val_string), dtype=np.uint16)
print(f"Fine-Tuning Token Footprint -> Train: {len(train_ids)} tokens | Val: {len(val_ids)} tokens")
'''

# 3. Apply Explicit Conditioning Envelopes
print("Processing text data with target conditioning prefixes...")
conditioned_texts = []
for item in rap_data:
    formatted_text = f"<|startoftext|>\n<|type|> rap\n[genre: boom_bap]\n[mood: emotional]\n<|content|>\n{item['lyrics']}\n<|endoftext|>\n"
    conditioned_texts.append(formatted_text)

# CRITICAL FIX: Split the LIST, not a joined string!
train_text, val_text = train_test_split(conditioned_texts, test_size=0.1, random_state=42)

# Safely encode the intact songs line-by-line (just like prepare.py)
def safe_encode(text_list, tokenizer):
    encoded = []
    for text in text_list:
        for line in text.split('\n'):
            encoded.extend(tokenizer.encode(line + '\n'))
    return encoded

train_ids = np.array(safe_encode(train_text, tokenizer), dtype=np.uint16)
val_ids = np.array(safe_encode(val_text, tokenizer), dtype=np.uint16)
print(f"Fine-Tuning Token Footprint -> Train: {len(train_ids)} tokens | Val: {len(val_ids)} tokens")


# 4. Localized Data Batch Ingestion Hub
def get_ft_batch(split):
    data = train_ids if split == 'train' else val_ids
    ix = torch.randint(len(data) - ft_config['block_size'], (ft_config['batch_size'],))
    x = torch.stack([torch.from_numpy((data[i:i+ft_config['block_size']]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+ft_config['block_size']]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_ft_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(ft_config['eval_iters'])
        for k in range(ft_config['eval_iters']):
            X, Y = get_ft_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# 5. Load Pre-Trained Foundation Weights
print(f"Loading base parameters from profile: {BASE_CONFIG_ID}")
model = GPT(ft_config)
base_checkpoint_path = f"experiments/{BASE_CONFIG_ID}_checkpoint.pt"

if not os.path.exists(base_checkpoint_path):
    print(f"Critical Error: Base checkpoint missing at {base_checkpoint_path}. Train the base model first!")
    sys.exit(1)

# Inject pre-trained weights into empty model architecture
model.load_state_dict(torch.load(base_checkpoint_path, map_location=device, weights_only=True))
model.to(device)
print("Base weights injected successfully. Ready to adapt layers.")

# Initialize fine-tuning optimizer configuration
optimizer = torch.optim.AdamW(model.parameters(), lr=ft_config['learning_rate'], weight_decay=0.01)

# 6. Run Fine-Tuning Loop
ft_history = []
start_time = time.time()

for step in range(ft_config['max_iters'] + 1):
    if step % ft_config['eval_interval'] == 0:
        metrics = estimate_ft_loss(model)
        elapsed = time.time() - start_time
        print(f"Fine-Tune Step {step:3d} | Train Loss: {metrics['train']:.4f} | Target Domain Val Loss: {metrics['val']:.4f} | Time: {elapsed:.1f}s")
        
        ft_history.append({
            'step': step,
            'train_loss': metrics['train'],
            'val_loss': metrics['val']
        })

    X, Y = get_ft_batch('train')
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Export tracking assets cleanly
log_path = f"logs/{FINETUNE_CONFIG_ID}_metrics.json"
checkpoint_path = f"experiments/{FINETUNE_CONFIG_ID}_checkpoint.pt"

with open(log_path, 'w') as f:
    json.dump(ft_history, f, indent=4)
torch.save(model.state_dict(), checkpoint_path)

print(f"\nFine-tuning session ended cleanly.")
print(f"Domain Logs Saved -> {log_path}")
print(f"Adapted Model Checkpoint Exported -> {checkpoint_path}\n")