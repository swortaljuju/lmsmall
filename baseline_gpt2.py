from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from components import AttentionMlpBlock
from base_trainer import BaseTrainer
import sys
from prepare_math_reasoning_data import MATH_REASONING_DATA_NAME

# -----------------------------------------------------------------------------

# Baseline GPT2 model from https://github.com/karpathy/build-nanogpt with some modifications
# Assume the model is only trained on CUDA devices

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 6  # number of heads. 6 instead of 12 to reduce training time
    n_embd: int = 142  # embedding dimension. 142 instead of 768 to reduce training time

# total 8.6m parameters
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([AttentionMlpBlock(config.n_embd, config.n_head) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
B = 64  # micro batch size
T = 1024  # sequence length
# create model
model = GPT(GPTConfig(vocab_size=50304))
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = (
    19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
)
weight_decay = 0.1
learning_rate = 6e-4

if __name__ == '__main__':
    data_name = len(sys.argv) > 1 and sys.argv[1] or MATH_REASONING_DATA_NAME
    resume_from_checkpoint = len(sys.argv) > 2 and sys.argv[2] or False
    trainer = BaseTrainer(
        "baseline_gpt2",
        model,
        total_batch_size=total_batch_size,
        B=B,
        T=T,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        data_name=data_name,
    )
    trainer.train_and_test(resume_from_checkpoint, warmup_steps, max_steps, 10000)
