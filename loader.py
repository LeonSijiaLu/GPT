import os
import torch
import numpy as np

def load_tokens(filename): # no need to tokenize, as the load has already done that
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(data=npt, dtype=torch.long)
    return ptt

class DataLoader:
    def __init__(self, B, T, process_rank, num_of_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_of_processes = num_of_processes
        assert split in {'train', 'val'}
        
        data_root = "data/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split")
        
        self.reset()
        
    def __len__(self):
        return len(self.tokens)
    
    def starting_point(self):
        return self.B * self.T * self.process_rank
    
    def stride(self):
        return self.B * self.T * self.num_of_processes
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[0])
        self.current_idx = self.starting_point()

    def get_batch(self):
        B, T = self.B, self.T
        
        buf = self.tokens[self.current_idx : self.current_idx + B * T + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # labels
        
        self.current_idx += self.stride()
        if self.current_idx + (self.stride() + 1) > len(self):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_idx = self.starting_point()
        
        return x, y