import os
import tiktoken
import torch

from model import GPT, GPTConfig
from torch.nn import functional as F

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"using device: {device}")

num_return_sequences = 5
max_tokens = 30

enc = tiktoken.get_encoding("gpt2")

tokens = enc.encode("Hello, I am a language model")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.squeeze(0).repeat(num_return_sequences, 1)

# data_dir = os.path.join('data', "tinyshakespeare", "tinyshakespeare.txt")
# with open(data_dir, 'r', encoding='utf-8') as f:
#     shakespeare_text = f.read()
# tokens = enc.encode(shakespeare_text)
# buf = torch.tensor(tokens[:4 * 32 + 1])
# x = buf[:-1].view(4, 32).to(device=device)
# y = buf[1:].view(4, 32).to(device=device)

# print(x.size(), y.size())


x = tokens.to(device=device)

model = GPT.from_pretrained("gpt2")
print("worked!")

model.eval()
model.to(device=device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

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