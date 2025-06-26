import tiktoken
import torch

class DataLoader:
    def __init__(self, B, T, process_rank, num_of_processes, file_path):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_of_processes = num_of_processes
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        
        self.current_idx = self.starting_point()
        
    def __len__(self):
        return len(self.tokens)
    
    def starting_point(self):
        return self.B * self.T * self.process_rank
    
    def stride(self):
        return self.B * self.T * self.num_of_processes

    def get_batch(self):
        B, T = self.B, self.T
        
        buf = self.tokens[self.current_idx : self.current_idx + B * T + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # labels
        
        self.current_idx += self.stride()
        if self.current_idx + (self.stride() + 1) > len(self):
            self.current_idx = self.starting_point()
        
        return x, y