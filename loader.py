import tiktoken
import torch

class DataLoader:
    def __init__(self, B, T, file_path):
        self.B = B
        self.T = T
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        
        self.current_idx = 0
        
    def __len__(self):
        return len(self.tokens)

    def get_batch(self):
        B, T = self.B, self.T
        
        buf = self.tokens[self.current_idx : self.current_idx + B * T + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # labels
        
        self.current_idx += B * T
        if self.current_idx + (B * T + 1) > len(self):
            self.current_idx = 0
        
        return x, y