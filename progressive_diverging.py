from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from components import (
    AttentionMlpBlock,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK,
    CausalSelfAttentionLoRA,
    CausalSelfAttention,
    MLP,
)
from base_trainer import BaseTrainer
from common_utils import setup_args_parser, setup_logger

# -----------------------------------------------------------------------------
# Progressive Diverging Model
# The training process is divided into three stages:
# Each stage is trained on 1/3 of the training data
# In each stage, the components' parameters are progressively diverged.
# This model is similar to GPT2 model with 12 layers of Attention and FeedForward blocks.
# The core idea is to train fewer parameters in the beginning and then diverge the parameters as training goes on.
# At Stage 1, the 12 layers are divided into 3 groups, each group has 4 layers.
# Each group shares same Attention and FeedForward block and hench shares same parameters while the block is called 4 times in serial.
# And the block's parameters' gradients are updated four times in a backward pass.
# At Stage 2, the 12 layers are again divided into 6 groups, each group has 2 layers.
# Each group shares same Attention and FeedForward block whose initial parameters are duplicated from corresponding Stage 1's block.
# Each block's attention parameters are updated using LORA technique and the feedforward parameters are updated as usual.
# And the block's parameters' gradients are updated 2 times in a backward pass.
# At Stage 3, each of 12 layers has its own Attention and FeedForward block.
# Each block's parameters duplicated from corresponding Stage 2's block.
# And attention parameters are updated using LORA technique and the feedforward parameters are updated as usual.
# Assume the model is only trained on CUDA devices

model_name = "progressive_diverging"

class Stage2AttentionMlpLoRABlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        pretrained_attention_mlp_block: AttentionMlpBlock,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.load_state_dict(pretrained_attention_mlp_block.state_dict())
        # Freeze pretrained attention weights
        self.attn.weight.requires_grad = False
        self.attn_lora = CausalSelfAttentionLoRA(n_embd, n_head)

    def forward(self, x):
        x = (
            x
            + self.attn(self.ln_1(x))
            + (DEFAULT_LORA_ALPHA / DEFAULT_LORA_RANK) * self.attn_lora(self.ln_1(x))
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class Stage3AttentionMlpLoRABlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        stage2_pretrained_attention_mlp_block: Stage2AttentionMlpLoRABlock,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.attn_lora = CausalSelfAttentionLoRA(n_embd, n_head)
        self.load_state_dict(stage2_pretrained_attention_mlp_block.state_dict())
        self.attn.weight.requires_grad = False
        self.attn_lora.weight.requires_grad = False
        self.attn_lora_2 = CausalSelfAttentionLoRA(n_embd, n_head)

    def forward(self, x):
        x = (
            x
            + self.attn(self.ln_1(x))
            + (DEFAULT_LORA_ALPHA / DEFAULT_LORA_RANK) * self.attn_lora(self.ln_1(x))
            + (DEFAULT_LORA_ALPHA / DEFAULT_LORA_RANK) * self.attn_lora_2(self.ln_1(x))
        )
        x = x + self.mlp(self.ln_2(x))
        return x

B = 8  # micro batch size
T = 256  # sequence length
@dataclass
class Config:
    block_size: int = T  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 6  # number of heads. 6 instead of 12 to reduce training time
    n_embd: int = 192  # embedding dimension. 142 instead of 768 to reduce training time


# total 11m parameters
class ProgressiveDiverging(nn.Module):

    def __init__(self, config: Config, log_level: int = 0):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.stage_1 = nn.ModuleList(
            [AttentionMlpBlock(config.n_embd, config.n_head) for _ in range(3)]
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
        self.__logger = setup_logger("progressive_diverging", model_name, log_level)[1]

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
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        self.__logger.debug(f"input embeddings shape: {x.shape}")
        # forward the blocks of the transformer
        if training_progress < 1 / 3:
            for block in self.stage_1:
                for _ in range(4):
                    x = block(x)
            self.__logger.debug(f"after stage 1: {x.shape}")
        elif training_progress < 2 / 3:
            if self.stage_2 is None:
                self.stage_2 = nn.ModuleList(
                    [
                        Stage2AttentionMlpLoRABlock(
                            self.config.n_embd,
                            self.config.n_head,
                            self.stage_1[idx // 2],
                        )
                        for idx in range(6)
                    ]
                )
                self.__logger.debug(f"stage 2 model: {self.stage_2}")
            for block in self.stage_2:
                for _ in range(2):
                    x = block(x)
            self.__logger.debug(f"after stage 2: {x.shape}")
        else:
            if self.stage_3 is None:
                self.stage_3 = nn.ModuleList(
                    [
                        Stage3AttentionMlpLoRABlock(
                            self.config.n_embd,
                            self.config.n_head,
                            self.stage_2[idx // 2],
                        )
                        for idx in range(12)
                    ]
                )
                self.__logger.debug(f"stage 3 model: {self.stage_3}")
            for block in self.stage_3:
                x = block(x)
            self.__logger.debug(f"after stage 3: {x.shape}")

        # forward the final layernorm and the classifier
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens

# create model
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = (
    19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
)
weight_decay = 0.1
learning_rate = 6e-4

if __name__ == "__main__":
    parser = setup_args_parser()
    args = parser.parse_args()
    data_name = args.data_name
    resume_from_checkpoint = args.resume_from_checkpoint
    trainer = BaseTrainer(
        "baseline_gpt2",
        ProgressiveDiverging(Config(), args.loglevel),
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
    trainer.train_and_test(resume_from_checkpoint, warmup_steps, max_steps, 10000)
