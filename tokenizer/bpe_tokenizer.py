import pickle

class SimpleBPETokenizer:
    def __init__(self,vocab_size=512):
        self.target_vocab_size = vocab_size
        self.merges = {} # (int, int) -> int
        self.vocab = {} # int -> bytes
        
    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train_from_text(self, text):
        print(f"[BPETokenizer] Training BPE up to vocab size {self.target_vocab_size}...")
        # Initialize with standard raw bytes (0-255)
        tokens = list(text.encode('utf-8'))
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        num_merges = self.target_vocab_size - 256
        current_ids = list(tokens)
        
        for i in range(num_merges):
            stats = self.get_stats(current_ids)
            if not stats:
                print("[BPETokenizer] No more pairs to merge.")
                break
            best_pair = max(stats, key=stats.get)
            idx = 256 + i
            current_ids = self.merge(current_ids, best_pair, idx)
            self.merges[best_pair] = idx
            self.vocab[idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            if (i + 1) % 50 == 0 or i == num_merges - 1:
                print(f"[BPETokenizer] Merge {i + 1}/{num_merges}: Merged pair {best_pair} into new token ID {idx}.")
                
        print(f"[BPETokenizer] Training complete. Final vocabulary size: {len(self.vocab)}")
        
    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            
            # find the pair that was merged earlier in training
            
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = self.merge(tokens, pair, self.merges[pair])
        return tokens
    '''
    def decode(self, ids):
        tokens = b"".join(self.vocab.get(idx,b"") for idx in ids)
        return tokens.decode("utf-8", errors="replace")
    '''
    
    def decode(self, ids):
        # 1. Rebuild the dictionary mapping ID -> raw bytes
        # Base vocabulary is standard 0-255 bytes
        vocab = {i: bytes([i]) for i in range(256)}
        
        # 2. Add the learned merges to the dictionary
        # This tells the tokenizer that ID 305 is actually composed of smaller bytes
        if hasattr(self, 'merges'):
            for (p0, p1), idx in self.merges.items():
                # Reconstruct the merged token from its parts
                vocab[idx] = vocab[p0] + vocab[p1]
                
        # 3. Translate the model's output IDs back into a continuous byte string
        raw_bytes = b"".join(vocab.get(i, b'') for i in ids)
        
        # 4. Safely decode the bytes into readable English text
        return raw_bytes.decode('utf-8', errors='replace')
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'merges': self.merges, 'vocab': self.vocab}, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.merges = data['merges']
        self.vocab = data['vocab']