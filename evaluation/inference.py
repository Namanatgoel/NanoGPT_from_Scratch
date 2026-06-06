'''
import os
import sys
import argparse
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import GPT
from configs.experiment_configs import CONFIGS
#from tokenizer.bpe_tokenizer import SimpleBPETokenizer
from tokenizer.char_tokenizer import CharacterTokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=40):
    device = next(model.parameters()).device
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block size
            idx_cond = idx[:, -model.config['block_size']:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Ban the Ghost Bytes (128-255) by setting their probability to negative infinity
            logits[:, 128:256] = -float('Inf')
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
    return tokenizer.decode(idx[0].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Fine-Tuned Model")
    parser.add_argument('--prompt', type=str, required=True, help="Input conditioning prompt")
    parser.add_argument('--temperature', type=float, default=0.7, help="Softmax temperature")
    parser.add_argument('--top_k', type=int, default=40, help="Top-K truncation")
    # Make sure this points to whatever your finetune.py saved its checkpoint as!
    parser.add_argument('--checkpoint', type=str, default='experiments/task_7_1_finetuned_checkpoint.pt') 
    
    args = parser.parse_args()
    
    # TO CORRECT THE OLD MISTAKE, SOLUTION -> 100% CONFIRMED conversion of terminal escaped strings (like \n) into real structural line breaks
    args.prompt = args.prompt.encode('utf-8').decode('unicode_escape')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Config and Tokenizer
    print(f"Loading checkpoint: {args.checkpoint}...")
    config = CONFIGS['task_4_1_baseline'] # Uses the baseline architecture footprint
   # tokenizer = SimpleBPETokenizer()
   # tokenizer.load('tokenizer/bpe_meta.pkl')
    tokenizer = CharacterTokenizer()
    tokenizer.load('tokenizer/char_meta.pkl')
    
    # Load Model Weights
    model = GPT(config)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        model.to(device)
        print("Model loaded successfully.\n")
    else:
        print(f"ERROR: Could not find {args.checkpoint}. Did your finetuning script save it under a different name?")
        sys.exit(1)
        
    print("=========================================")
    print(f"PROMPT:\n{args.prompt}")
    print("=========================================")
    print("GENERATING...\n")
    
    output = generate_text(model, tokenizer, args.prompt, max_new_tokens=150, temperature=args.temperature, top_k=args.top_k)
    print(output)
    print("\n=========================================")
'''
'''
import os
import sys
import argparse
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import GPT
from configs.experiment_configs import CONFIGS
from tokenizer.bpe_tokenizer import SimpleBPETokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=40):
    device = next(model.parameters()).device
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.config['block_size']:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # --- THE GHOST BYTE BLOCKER ---
            # Permanently bans the model from outputting untrained UTF-8 symbols
            logits[:, 128:256] = -float('Inf')
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
    return tokenizer.decode(idx[0].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Fine-Tuned Model")
    parser.add_argument('--prompt', type=str, required=True, help="Input conditioning prompt")
    parser.add_argument('--temperature', type=float, default=0.7, help="Softmax temperature")
    parser.add_argument('--top_k', type=int, default=40, help="Top-K truncation")
    parser.add_argument('--checkpoint', type=str, default='experiments/task_7_1_finetuned_checkpoint.pt') 
    
    args = parser.parse_args()
    
    # 1. Format the prompt line breaks
    args.prompt = args.prompt.encode('utf-8').decode('unicode_escape')
    
    # --- PROMPT ENCRYPTION (-2 SHIFT) ---
    # The model only understands the encrypted language, so we must translate the English prompt to cipher
    encrypted_prompt = ""
    for char in args.prompt:
        if 'a' <= char <= 'z':
            encrypted_prompt += chr(((ord(char) - ord('a') - 2) % 26) + ord('a'))
        elif 'A' <= char <= 'Z':
            encrypted_prompt += chr(((ord(char) - ord('A') - 2) % 26) + ord('A'))
        elif char == ' ':
            encrypted_prompt += '>'
        else:
            encrypted_prompt += char
            
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading checkpoint: {args.checkpoint}...")
    config = CONFIGS['task_4_1_baseline'] 
    
    tokenizer = SimpleBPETokenizer(vocab_size=512)
    tokenizer.load('tokenizer/bpe_meta.pkl')
    
    model = GPT(config)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        model.to(device)
        print("Model loaded successfully.\n")
    else:
        print(f"ERROR: Could not find {args.checkpoint}.")
        sys.exit(1)
        
    print("=========================================")
    print(f"ORIGINAL PROMPT:\n{args.prompt}")
    print(f"ENCRYPTED PROMPT FED TO MODEL:\n{encrypted_prompt}")
    print("=========================================")
    print("GENERATING...\n")
    
    # Pass the ENCRYPTED prompt to the model
    raw_output = generate_text(model, tokenizer, encrypted_prompt, max_new_tokens=150, temperature=args.temperature, top_k=args.top_k)
    
    print("--- RAW ENCRYPTED OUTPUT ---")
    print(raw_output)
    
    # --- OUTPUT DECRYPTION (+2 SHIFT) ---
    # Translate the model's encrypted output back to English for you to read
    print("\n--- FINAL DECRYPTED LYRICS ---")
    decrypted_text = ""
    for char in raw_output:
        if 'a' <= char <= 'z':
            decrypted_text += chr(((ord(char) - ord('a') + 2) % 26) + ord('a'))
        elif 'A' <= char <= 'Z':
            decrypted_text += chr(((ord(char) - ord('A') + 2) % 26) + ord('A'))
        elif char == '>':
            decrypted_text += ' '
        else:
            decrypted_text += char
            
    print(decrypted_text)
    print("\n=========================================")
    '''

import os
import sys
import argparse
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import GPT
from configs.experiment_configs import CONFIGS
from tokenizer.bpe_tokenizer import SimpleBPETokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7, top_k=40):
    device = next(model.parameters()).device
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.config['block_size']:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # --- THE GHOST BYTE BLOCKER ---
            # Permanently bans the model from outputting untrained UTF-8 symbols
            logits[:, 128:256] = -float('Inf')
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
    return tokenizer.decode(idx[0].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Fine-Tuned Model")
    parser.add_argument('--prompt', type=str, required=True, help="Input conditioning prompt")
    parser.add_argument('--temperature', type=float, default=0.7, help="Softmax temperature")
    parser.add_argument('--top_k', type=int, default=40, help="Top-K truncation")
    # DEFAULT TO YOUR FINETUNED CHECKPOINT
    parser.add_argument('--checkpoint', type=str, default='experiments/task_7_1_finetuned_checkpoint.pt') 
    
    args = parser.parse_args()
    
    # 1. Format the prompt line breaks properly
    args.prompt = args.prompt.encode('utf-8').decode('unicode_escape')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading checkpoint: {args.checkpoint}...")
    config = CONFIGS['task_4_1_baseline'] 
    
    tokenizer = SimpleBPETokenizer(vocab_size=512)
    tokenizer.load('tokenizer/bpe_meta.pkl')
    
    model = GPT(config)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        model.to(device)
        print("Model loaded successfully.\n")
    else:
        print(f"ERROR: Could not find {args.checkpoint}.")
        sys.exit(1)
        
    print("=========================================")
    print(f"PROMPT:\n{args.prompt}")
    print("=========================================")
    print("GENERATING...\n")
    
    output = generate_text(model, tokenizer, args.prompt, max_new_tokens=150, temperature=args.temperature, top_k=args.top_k)
    print(output)
    print("\n=========================================")