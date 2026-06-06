import os
import sys
import torch
from torch.nn import functional as F

# Append root paths cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import GPT
from configs.experiment_configs import CONFIGS
from tokenizer.bpe_tokenizer import SimpleBPETokenizer

def load_model_and_tokenizer(config_id, checkpoint_path):
    print(f"Loading Configuration: {config_id}")
    config = CONFIGS[config_id]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Initialize empty architecture
    model = GPT(config)
    
    # 2. Load trained weights
    assert os.path.exists(checkpoint_path), f"Checkpoint missing: {checkpoint_path}"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # CRITICAL: Disables dropout during inference
    
    # 3. Load associated Tokenizer
    tokenizer = SimpleBPETokenizer()
    tokenizer.load('tokenizer/bpe_meta.pkl')
    
    return model, tokenizer, device

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=100, strategy='greedy', **kwargs):
    device = next(model.parameters()).device
    # Encode the starting prompt
    input_ids = tokenizer.encode(prompt)
    idx = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0) # Shape: (1, T)

    for _ in range(max_new_tokens):
        # Ensure context length doesn't exceed block size
        idx_cond = idx[:, -model.config['block_size']:]
        
        # Forward pass to get logits for the *last* token only
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] # Shape: (1, vocab_size)

        if strategy == 'greedy':
            # The most probable token, 100% of the time. Highly repetitive.
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        elif strategy == 'temperature':
            # Softens or sharpens the distribution before rolling the dice
            temp = kwargs.get('temperature', 1.0)
            logits = logits / temp
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
        elif strategy == 'top_k':
            # Restricts the dice roll to only the top K most likely tokens
            k = kwargs.get('top_k', 10)
            temp = kwargs.get('temperature', 1.0)
            logits = logits / temp
            # Find the value of the Kth largest logit
            v, _ = torch.topk(logits, min(k, logits.size(-1)))
            # Force everything lower than the Kth logit to -infinity
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
        elif strategy == 'top_p':
            # Nucleus Sampling: Sort probabilities and dynamically cut off the tail once mass hits P.
            p = kwargs.get('top_p', 0.9)
            temp = kwargs.get('temperature', 1.0)
            logits = logits / temp
            
            # Sort logits descending
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Create a mask for tokens whose cumulative probability exceeds p
            sorted_indices_to_remove = cumulative_probs > p
            # Shift the mask to the right to keep the token that officially pushes the mass over P
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter the mask back to the original index positions
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
        # Append sampled token to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    # Decode binary array back to human text
    generated_text = tokenizer.decode(idx[0].tolist())
    return generated_text

# --- Execution Block ---
if __name__ == "__main__":
    CONFIG_NAME = 'task_4_1_baseline'
    CHECKPOINT_PATH = f'experiments/{CONFIG_NAME}_checkpoint.pt'
    
    try:
        model, tokenizer, device = load_model_and_tokenizer(CONFIG_NAME, CHECKPOINT_PATH)
    except Exception as e:
        print(f"Error loading model. Did you run Part 4 first? {e}")
        sys.exit(1)

    prompt = "The shadow falls across the street,\n"
    print(f"\n--- Base Prompt ---\n{prompt}")

    print("\n[Strategy 1: Greedy Decoding]")
    print(generate_text(model, tokenizer, prompt, max_new_tokens=50, strategy='greedy'))

    print("\n[Strategy 2: Low Temperature (T=0.3)]")
    print(generate_text(model, tokenizer, prompt, max_new_tokens=50, strategy='temperature', temperature=0.3))

    print("\n[Strategy 3: High Temperature (T=1.5)]")
    print(generate_text(model, tokenizer, prompt, max_new_tokens=50, strategy='temperature', temperature=1.5))

    print("\n[Strategy 4: Top-K (K=10, T=0.8)]")
    print(generate_text(model, tokenizer, prompt, max_new_tokens=50, strategy='top_k', top_k=10, temperature=0.8))

    print("\n[Strategy 5: Top-P Nucleus (P=0.9, T=0.8)]")
    print(generate_text(model, tokenizer, prompt, max_new_tokens=50, strategy='top_p', top_p=0.9, temperature=0.8))