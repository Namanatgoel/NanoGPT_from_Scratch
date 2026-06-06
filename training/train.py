import os
import sys
import time
import json
import torch
import numpy as np

# Append root project paths cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import GPT
from configs.experiment_configs import CONFIGS

if len(sys.argv) < 2:
    print("Error: Specify a configuration profile name to run.")
    print(f"Available profiles: {list(CONFIGS.keys())}")
    sys.exit(1)

config_id = sys.argv[1]
assert config_id in CONFIGS, f"Configuration ID '{config_id}' not found."
config = CONFIGS[config_id]

print(f"=========================================")
print(f"Launching Experiment Engine: {config_id}")
print(f"=========================================")

# Detect hardware environment details cleanly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Targeting compute accelerator: {device}")

# Memory map datasets prepared during Task 2 steps
train_data = np.memmap('data/train_bpe.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('data/val_bpe.bin', dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    x = torch.stack([torch.from_numpy((data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

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

# Initialize tracking model instances dynamically
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

history_logs = []
start_time = time.time()

for step in range(config['max_iters'] + 1):
    # Periodic assessment evaluation boundaries
    if step % config['eval_interval'] == 0:
        metrics = estimate_loss(model)
        elapsed = time.time() - start_time
        print(f"Step {step:4d} | Train Loss: {metrics['train']:.4f} | Val Loss: {metrics['val']:.4f} | Time Elapsed: {elapsed:.1f}s")
        
        history_logs.append({
            'step': step,
            'train_loss': metrics['train'],
            'val_loss': metrics['val'],
            'elapsed_seconds': elapsed
        })
        
    X, Y = get_batch('train')
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Export tracking logs directly out to project/logs/
log_filepath = f"logs/{config_id}_metrics.json"
with open(log_filepath, 'w') as f:
    json.dump(history_logs, f, indent=4)

# Export optimized weights directly out to project/experiments/
checkpoint_filepath = f"experiments/{config_id}_checkpoint.pt"
torch.save(model.state_dict(), checkpoint_filepath)

print(f"Experiment finished cleanly.")
print(f"Logs -> {log_filepath} | Weights -> {checkpoint_filepath}\n")