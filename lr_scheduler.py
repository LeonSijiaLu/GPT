import math

class LRScheduler:
    def __init__(self, max_lr, min_lr, warmup_steps, max_steps):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        
        if step > self.max_steps:
            return self.min_lr
        
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # cosine decay of learning rate
        return self.min_lr + coef * (self.max_lr - self.min_lr)