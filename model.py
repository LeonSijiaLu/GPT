import torch
import torch.nn as nn
import math

from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    seq_size: int = 1024 # sequence size
    vocab_size: int = 50257
    n_layer: int = 12 # number of hidden layers
    n_head: int = 12 # number of heads
    n_embed: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed), # token --> embedding
            wpe = nn.Embedding(config.seq_size, config.n_embed), # position --> embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))

        # hidden layer --> vocab_size, with no bias
        # this is the final stage before softmax, the output is logits
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
        # "weight sharing", meaning certain parameters are shared between components
        # the motivation is similar tokens should have similar embeddings, it should enhance generic, 
        # efficiency (removed tons of parameters), and stability
        # 
        # self.transformer.wte.weight: (Token --> Embedding)
        # self.lm_head.weight: (Embedding --> Token)
        self.transformer.wte.weight = self.lm_head.weight
        
        # initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm): # redundant
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, target = None):
        # idx and target both are (B, T)
        # ex. idx = [I like apple], target = [like apple and]

        B, T = idx.size()
        assert T <= self.config.seq_size, f"Cannot forward sequence length of {T}, sequence size is only {self.config.seq_size}"
        
        if target is not None:
            assert idx.size() == target.size(), f"Invalid format of target {idx.size()} != {target.size()}"

        t_e = self.transformer.wte(idx) # token embedding, (B, T, C)

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        p_e = self.transformer.wpe(pos) # positional embedding, (T, C)

        x = t_e + p_e # (B, T, C)

        for h_l in self.transformer.h:
            x = h_l(x)

        x = self.transformer.ln_f(x) # normalizes

        logits = self.lm_head(x) # (B, T, vocab_size)
        
        # calculate loss
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        """
        Loads pretrained gpt-2 model weights from huggingface
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained model {model_type}")

        config_args = { # gpt-2 config info
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),
            "gpt2-large": dict(n_layer=36, n_head=120, n_embed=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024 # seq_size

        # create a from-scratch initialized GPT model
        config = GPTConfig()
        model = GPT(config=config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard mask / buffers, as these are not learnable parameters

        # create hugging face model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)

        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith('.attn.bias') and not k.endswith('.attn.masked_bias')]

        assert len(sd_keys) == len(sd_hf_keys), f"mismatches keys: {len(sd_keys)} != {len(sd_hf_keys)}"

        # assign weights
        # some weights need to be transposed, as hugging face model is developed by tensorflow
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_keys:
            if any(k.endswith(t) for t in transposed):
                assert sd[k].shape == sd_hf[k].shape[::-1], f"mismatches keys: {sd[k].shape} != {sd_hf[k].shape[::-1]}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].transpose(0, 1))
            else:
                assert sd[k].shape == sd_hf[k].shape, f"mismatches keys: {sd[k].shape} != {sd_hf[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        # in the original paper "Attention is all you need", layer normalization is inside the residual path way,
        # this is not very desirable. It's usually preferred to have clean residual stream all the way from 
        # supervision to the inputs token; 
        #
        # Why?
        #
        # Additions distribute gradients equally to branches, and we want that gradient stream to flow straight into
        # the inputs through residual pathways during backpropagation. 
        # If we include layer normalization in residual paths, it will change weights overtime.

        x = x + self.attn(self.ln_1(x)) # tokens exchange information
        x = x + self.mlp(self.ln_2(x)) # each token thinks individually about the information they obtained
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        
        # using tanh in self.gelu is a legacy issue, just following gpt-2's initial design
        # gelu is smoother than relu at position 0, better at avoiding gradient vanishing

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention uses multiple attention heads to focus on different parts of the input sequence.
    Each head learns different patterns, and the output from all the heads are combined to produce the final result
    """
    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # q, k, v projection for all heads, but in a batch;
        # 1. The model learns the weight matrix W, which transforms raw embeddings to q, k, v
        # 2. Projection controls how much attention each word should pay to each other
        # 3. Learned weights define how much tokens interact in self-attention
        # 4. Attention's goal is to learn relationships between tokens
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)

        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        # mask, but following openai's naming
        self.register_buffer("bias", torch.tril(torch.ones(config.seq_size, config.seq_size))
                            .view(1, 1, config.seq_size, config.seq_size))
        
    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x) #(B, T, 3 * C)
        q, k, v = qkv.split(self.n_embed, dim=-1) # each (B, T, C)

        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)

        # in later computations, we want pytorch to treat (B, self.n_head) as batches,
        # the operations will be performed on (T, C // self.n_head) in parallel,
        q = q.transpose(1, 2)
        k = k.transpose(1, 2) # (B, self.n_head, T, C // self.n_head)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(2, 3)) * (1.0 / math.sqrt(k.size(-1))) # (B, self.n_head, T, T)
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf")) # ensures tokens interact only with old tokens
        attn = F.softmax(attn, dim=-1) # normalizes the attention, probability sums to 1

        y = attn @ v # (B, self.n_head, T, C // self.n_head), weighted sum of tokens we found interesting
        y = y.transpose(1, 2).contiguous() # (B, T, self.n_head, C // self.n_head)
        y = y.view(B, T, C) # (B, T, C)
        
        # note:
        # y = x.transpose(), y and x are indexing the same array; 
        # however, y's memory is not continuous, thus it throws an error if we call .view() on y;
        # solition is y = x.transpose().contiguous(), it creates a brand new, continuous array, and y index on this new array

        # print(x.data_ptr() == y.data_ptr()) # True
        # y_contig = y.contiguous()
        # # print(y_contig.data_ptr() == y.data_ptr()) # False
        
        y = self.c_proj(y)

        return y
