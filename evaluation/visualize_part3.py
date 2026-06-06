import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import GPT
from configs.experiment_configs import CONFIGS
from tokenizer.bpe_tokenizer import SimpleBPETokenizer

def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = CONFIGS['task_4_1_baseline']
    
    # Untrained Model (Random Weights)
    untrained_model = GPT(config).to(device)
    
    # Trained Model
    trained_model = GPT(config)
    checkpoint_path = 'experiments/task_4_1_baseline_checkpoint.pt'
    if os.path.exists(checkpoint_path):
        trained_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    trained_model.to(device).eval()
    
    tokenizer = SimpleBPETokenizer()
    tokenizer.load('tokenizer/bpe_meta.pkl')
    
    return untrained_model, trained_model, tokenizer, device

def visualize_embeddings(untrained, trained):
    print("Generating Embedding PCA plots...")
    # Extract weights
    w_un = untrained.transformer.wte.weight.detach().cpu().numpy()
    w_tr = trained.transformer.wte.weight.detach().cpu().numpy()
    
    pca = PCA(n_components=2)
    un_pca = pca.fit_transform(w_un)
    tr_pca = pca.fit_transform(w_tr)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(un_pca[:, 0], un_pca[:, 1], alpha=0.5, color='red')
    axes[0].set_title('Untrained Embeddings (Task 3.1)')
    
    axes[1].scatter(tr_pca[:, 0], tr_pca[:, 1], alpha=0.5, color='blue')
    axes[1].set_title('Trained Embeddings (Task 3.1)')
    
    plt.savefig('reports/embedding_pca.png')
    print("Saved to reports/embedding_pca.png")

def visualize_attention(model, tokenizer, text):
    print("Generating Attention Heatmap using Hooks...")
    device = next(model.parameters()).device
    tokens = tokenizer.encode(text)
    idx = torch.tensor([tokens], dtype=torch.long).to(device)
    decoded_tokens = [tokenizer.decode([t]) for t in tokens]
    
    attention_maps = []
    
    # PyTorch Hook to intercept the attention matrix without modifying the model code
    def hook_fn(module, input, output):
        # Reconstruct the attention matrix math internally for the hook
        x = input[0]
        B, T, C = x.size()
        q, k, v = module.c_attn(x).split(module.n_embd, dim=2)
        k = k.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
        q = q.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        '''if module.use_causal_masking:
            att = att.masked_fill(module.bias[:,:,:T,:T] == 0, float('-inf'))'''
            
        # Dynamically check for the presence of the causal mask buffer
        if hasattr(module, 'bias'):
            att = att.masked_fill(module.bias[:,:,:T,:T] == 0, float('-inf'))
        elif hasattr(module, 'tril'):
            att = att.masked_fill(module.tril[:,:,:T,:T] == 0, float('-inf'))
            
            
        att = torch.nn.functional.softmax(att, dim=-1)
        attention_maps.append(att.detach().cpu())

    # Attach hook to the first transformer block's attention layer
    hook_handle = model.transformer.h[0].attn.register_forward_hook(hook_fn)
    
    # Forward pass to trigger the hook
    with torch.no_grad():
        model(idx)
        
    hook_handle.remove() # Clean up
    
    # Plotting Head 0 of Layer 0
    att_matrix = attention_maps[0][0, 0].numpy() 
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(att_matrix, xticklabels=decoded_tokens, yticklabels=decoded_tokens, cmap='viridis')
    plt.title("Self-Attention Weights (Task 3.3)")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('reports/attention_heatmap.png')
    print("Saved to reports/attention_heatmap.png")

if __name__ == "__main__":
    untrained_model, trained_model, tokenizer, device = load_models()
    visualize_embeddings(untrained_model, trained_model)
    
    sample_text = "The quick brown fox jumps"
    visualize_attention(trained_model, tokenizer, sample_text)