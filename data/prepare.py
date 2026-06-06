import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import sys

# Ensure project root is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.char_tokenizer import CharacterTokenizer
from tokenizer.bpe_tokenizer import SimpleBPETokenizer

print("1. Loading clean datasets...")
all_texts = []

# Load intact songs/poems into a list (DO NOT join them yet!)
with open('/home/naman/Cryptonite-JM-Tasks/GenAI/Task-4/Project/data/clean_poems_corpus.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        all_texts.append(json.loads(line)['text'])

with open('/home/naman/Cryptonite-JM-Tasks/GenAI/Task-4/Project/data/clean_rap_dataset.json', 'r', encoding='utf-8') as f:
    raps = json.load(f)
    for r in raps:
        all_texts.append(r['lyrics'])

# Split the list of INTACT songs (This keeps words and sentences completely intact)
train_texts, val_texts = train_test_split(all_texts, test_size=0.1, random_state=42)

# Join only for tokenizer training
train_string_for_training = "\n".join(train_texts)

print("2. Processing Character Tokenizer...")
char_tok = CharacterTokenizer()
char_tok.train_from_text(train_string_for_training)
char_tok.save('tokenizer/char_meta.pkl')

# Safely encode the joined text for character level
train_char_ids = np.array(char_tok.encode("\n".join(train_texts)), dtype=np.uint16)
val_char_ids = np.array(char_tok.encode("\n".join(val_texts)), dtype=np.uint16)
train_char_ids.tofile('data/train_char.bin')
val_char_ids.tofile('data/val_char.bin')
print("Saved character-level binary files to data/")

print("3. Processing BPE Tokenizer...")
bpe_tok = SimpleBPETokenizer(vocab_size=512)
bpe_tok.train_from_text(train_string_for_training[:1000000]) 
bpe_tok.save('tokenizer/bpe_meta.pkl')

print("4. Fast-encoding BPE tokens...")
def safe_encode_list(text_list, tokenizer):
    encoded_ids = []
    # text_list is a list of full songs, not letters
    for text in text_list:
        if not text: continue
        # Split the song into actual lyrical lines
        for line in text.split('\n'):
            # Safely encode the line and its natural newline
            encoded_ids.extend(tokenizer.encode(line + '\n'))
    return np.array(encoded_ids, dtype=np.uint16)

train_bpe_ids = safe_encode_list(train_texts, bpe_tok)
val_bpe_ids = safe_encode_list(val_texts, bpe_tok)

train_bpe_ids.tofile('data/train_bpe.bin')
val_bpe_ids.tofile('data/val_bpe.bin')
print("Saved BPE-level binary files to data/")
print("PIPELINE CLEANSED: Tokenizer and data are 100% healthy.")

'''BETTER OPTION
# 5. Fast-Encoding BPE (Aggressive Line-by-Line Chunking)
print("Fast-encoding BPE tokens in chunks...")
def fast_encode_list(text_list, tokenizer):
    encoded_ids = []
    # Iterate through the intact list of songs/poems
    for text in text_list:
        if not text:
            continue
        # AGGRESSIVE CHUNKING: Split the song into individual lines
        for line in text.split('\n'):
            # Encoding 10 words at a time is instantaneous
            encoded_ids.extend(tokenizer.encode(line + '\n'))
            
    return np.array(encoded_ids, dtype=np.uint16)
'''

'''OLD MISTAKE REASON IN CODE
# Flatten the lists of strings into a single continuous string block for BPE tokenization
train_string = "\n".join(train_txt)
val_string = "\n".join(val_txt)

# Encode using the unified string blocks
train_bpe_ids = np.array(bpe_tok.encode(train_string), dtype=np.uint16)
val_bpe_ids = np.array(bpe_tok.encode(val_string), dtype=np.uint16)
'''
'''O(N^2) choking the new run so chunking now
# --- New Corrected Code ---
# Safely handle the data whether it was read as a list of lines or a single giant string
train_string = "".join(train_txt) if isinstance(train_txt, list) else train_txt
val_string = "".join(val_txt) if isinstance(val_txt, list) else val_txt

train_bpe_ids = np.array(bpe_tok.encode(train_string), dtype=np.uint16)
val_bpe_ids = np.array(bpe_tok.encode(val_string), dtype=np.uint16)
'''
