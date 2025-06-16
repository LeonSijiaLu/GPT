import os
import tiktoken
import torch
import time

from loader import DataLoader
from model import GPT, GPTConfig
from lr_scheduler import LRScheduler
from torch.nn import functional as F

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    
# sets the precision used for matrix multiplication,
# default is "highest", "high" is less precision, but good enough
torch.set_float32_matmul_precision("high")

data_dir = os.path.join('data', "tinyshakespeare", "tinyshakespeare.txt")

num_return_sequences = 5
max_tokens = 30
max_steps = 50
enc = tiktoken.get_encoding("gpt2")

# tokens = enc.encode("Hello, I am a language model")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.squeeze(0).repeat(num_return_sequences, 1)

model = GPT(GPTConfig())
model.to(device=device)
# model = torch.compile(model)
print("worked!")

train_loader = DataLoader(B=8, T=512, file_path=data_dir)

lr_scheduler = LRScheduler(max_lr=3e-4, min_lr=3e-4 * 0.1, warmup_steps=10, max_steps=50)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for i in range(max_steps):
    t0 = time.time()
    x, y = train_loader.get_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()

    # bfloat16 and tf32 has the same range, just different precision
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    
    # to clip the global norm of the gradient at 1.0
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # GPT-3 Paper
    
    # learning rate scheduling, GPT-3 Paper
    lr = lr_scheduler.get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step() # update parameters, decrease loss
    torch.cuda.synchronize() # gpu operations are async
    
    t1 = time.time()
    dts = t1 - t0
    tokens_per_sec = (train_loader.B * train_loader.T) / dts

    print(f"step {i}, loss: {loss.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt:{dts * 1000:.2f}ms, tok/sec:{tokens_per_sec:.2f}")


import sys; sys.exit(0)

while x.size()[1] < max_tokens:
    with torch.no_grad():
        logits, loss = model(x) # (B, T, vocab_size)
        
        logits = logits[:, -1, :] # (B, vocab_size), focuses on the last token

        probas = F.softmax(logits, dim=-1)

        # get top k, topk_probs (B, 50), topk_idx (B, 50)
        # topk_probs is sorted in descending order, topk_idx is the corresponding index of the entire vocab space
        topk_probs, topk_idx = torch.topk(probas, 50, dim=-1)

        # sample one token from top k
        ix = torch.multinomial(topk_probs, num_samples=1)

        x_next = torch.gather(topk_idx, -1, ix)

        x = torch.concat((x, x_next), dim=-1)

for i in range(num_return_sequences):
    tokens = x[i].tolist()
    decoded = enc.decode(tokens)
    print(">" + decoded)