from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from components import AttentionMlpBlock
from base_trainer import BaseTrainer
from common_utils import setup_args_parser, setup_logger

# -----------------------------------------------------------------------------

# Baseline GPT2 model from https://github.com/karpathy/build-nanogpt with some modifications
# Assume the model is only trained on CUDA devices

B = 8 #64  # micro batch size
T = 256 # 1024  # sequence length
@dataclass
class GPTConfig:
    block_size: int = T  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 6  # number of heads. 6 instead of 12 to reduce training time
    n_embd: int = 192  # embedding dimension. 142 instead of 768 to reduce training time

model_name = "baseline_gpt2"

# total 15m parameters
class GPT(nn.Module):

    def __init__(self, config: GPTConfig, log_level: int = 0):
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
        
        self.__logger = setup_logger("baseline_gpt2", model_name, log_level)
        self.__logger.info(f"model structure {self}")
        self.__logger.info(f"model config {config}")

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

    def forward(self, idx, targets=None, training_progress: float = 0.0):
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
        self.__logger.debug(f"shape after embedding {x.shape}")
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        self.__logger.debug(f"shape after transformer {x.shape}")    
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
training_steps = (
    20 # 10000
)
testing_steps = 250000
weight_decay = 0.1
learning_rate = 6e-4

if __name__ == '__main__':
    parser = setup_args_parser()
    args = parser.parse_args()
    data_name = args.data_name
    resume_from_checkpoint = args.resume_from_checkpoint
    trainer = BaseTrainer(
        "baseline_gpt2",
        GPT(GPTConfig(vocab_size=50304), args.loglevel),
        total_batch_size=total_batch_size,
        B=B,
        T=T,
        max_lr=max_lr,
        min_lr=min_lr,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        data_name=data_name,
        log_level=args.loglevel,
    )
    trainer.train_and_test(resume_from_checkpoint, warmup_steps, training_steps, testing_steps)
