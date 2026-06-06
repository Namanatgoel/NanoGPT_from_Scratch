import json
import pickle
import os

class CharacterTokenizer:
    def __init__(self):
        self.vocab = []
        self.stoi = {}
        self.itos = {}
    
    def train_from_text(self, text_corpus):
        # Extract all unique characters from the training corpus
        unique_chars = sorted(list(set(text_corpus)))
        
        #Always ensure special control tokens are handled if present
        self.vocab = unique_chars
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)} 
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        print(f"[CharTokenizer] Trained with vocab size: {len(self.vocab)}")
    
    def encode(self, text):
        # Fallback to a space or unknown flag if a character isn't recognized
        return [self.stoi[c] for c in text if c in self.stoi]
    
    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids if i in self.itos])
    
    def save(self, filepath):
        meta = {'vocab':self.vocab, 'stoi':self.stoi, 'itos':self.itos}
        with open(filepath, 'wb') as f:
            pickle.dump(meta, f)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            meta = pickle.load(f)
            self.vocab = meta['vocab']
            self.stoi = meta['stoi']
            self.itos = meta['itos']