from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from components import AttentionMlpBlock
from base_trainer import BaseTrainer
from common_utils import setup_args_parser, setup_logger

# -----------------------------------------------------------------------------

# ConvAttention model.
# Layers:
# ------ Input ------
# ------ Vocabulary to embedding ------
# ------ Attention and FeedForward layer  ------
# ------ (embedding = initial_n_embedding, sequence length = initial_block_size) ------
# ------ Compressor layer ------
# ------ (embedding = initial_n_embedding * 2, sequence length = initial_block_size / 4) ------
# ------ Attention and FeedForward layer X 2 ------
# ------ (embedding = initial_n_embedding * 2, sequence length = initial_block_size / 4) ------
# ------ Compressor layer ------
# ------ (embedding = initial_n_embedding * 4, sequence length = initial_block_size / 16) ------
# ------ Attention and FeedForward layer X 2 ------
# ------ (embedding = initial_n_embedding * 4, sequence length = initial_block_size / 16) ------
# ------ Compressor layer ------
# ------ (embedding = initial_n_embedding * 8, sequence length = initial_block_size / 64) ------
# ------ Attention and FeedForward layer X 2  ------
# ------ (embedding = initial_n_embedding * 8, sequence length = initial_block_size / 64) ------
# ------ Expander layer ------
# ------ (embedding = initial_n_embedding * 4, sequence length = initial_block_size / 16) ------
# ------ Attention and FeedForward layer X 2 ------
# ------ (embedding = initial_n_embedding * 4, sequence length = initial_block_size / 16) ------
# ------ Expander layer ------
# ------ (embedding = initial_n_embedding * 2, sequence length = initial_block_size / 4) ------
# ------ Attention and FeedForward layer X 2 ------
# ------ (embedding = initial_n_embedding * 2, sequence length = initial_block_size / 4) ------
# ------ Expander layer ------
# ------ (embedding = initial_n_embedding, sequence length = initial_block_size) ------
# ------ Attention and FeedForward layer  ------
# ------ (embedding = initial_n_embedding, sequence length = initial_block_size) ------
# ------ Embedding to vocabulary ------
# Assume the model is only trained on CUDA devices

EMBEDDING_FAN_IN_FACTOR = 2
SEQUENCE_FAN_IN_FACTOR = 4

model_name = "conv_attention"

# Compress input from long sequence with small embedding into short sequence with large embedding
class Compressor(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.conv = nn.Conv1d(
            n_embd,
            n_embd * EMBEDDING_FAN_IN_FACTOR,
            SEQUENCE_FAN_IN_FACTOR,
            stride=SEQUENCE_FAN_IN_FACTOR,
        )

    def forward(self, x):
        return self.conv(self.ln(x).transpose(1, 2)).transpose(1, 2)


class Expander(nn.Module):
    def __init__(self, sequence_length: int, n_embd: int):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.conv = nn.Conv1d(
            sequence_length,
            sequence_length * SEQUENCE_FAN_IN_FACTOR,
            EMBEDDING_FAN_IN_FACTOR,
            stride=EMBEDDING_FAN_IN_FACTOR,
        )

    def forward(self, x):
        return self.conv(self.ln(x))

B = 8  # micro batch size
T = 1024  # sequence length
@dataclass
class Config:
    initial_block_size: int = T  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_head: int = 6  # number of heads. 6 instead of 12 to reduce training time
    initial_n_embedding: int = (
        48  # embedding dimension. 142 instead of 768 to reduce training time
    )
    n_layer: int = 12  # number of layers


# total params: 9.24m
class ConvAttentionModel(nn.Module):
    def __init__(self, config: Config, log_level: int = 0):
        super().__init__()
        assert (
            config.initial_block_size % pow(SEQUENCE_FAN_IN_FACTOR, 3) == 0
        ), f"initial_block_size must be divisible by ${SEQUENCE_FAN_IN_FACTOR} ^ 3"
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.initial_n_embedding),
                wpe=nn.Embedding(config.initial_block_size, config.initial_n_embedding),
                conv_attention=nn.Sequential(*self.__add_conv_attention_blocks(config)),
                ln_f=nn.LayerNorm(config.initial_n_embedding),
            )
        )
        self.lm_head = nn.Linear(config.initial_n_embedding, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
        self.__logger = setup_logger("ConvAttentionModel", model_name, log_level)
        self.__logger.debug(f"conv attention layers: {self.transformer.conv_attention}")

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

    def __add_conv_attention_blocks(self, config: Config) -> list[nn.Module]:
        layers = []
        n_embedding = config.initial_n_embedding
        block_size = config.initial_block_size
        for _ in range(3):
            layers.append(AttentionMlpBlock(n_embedding, config.n_head))
            layers.append(Compressor(n_embedding))
            n_embedding *= EMBEDDING_FAN_IN_FACTOR
            block_size //= SEQUENCE_FAN_IN_FACTOR
            layers.append(AttentionMlpBlock(n_embedding, config.n_head))

        for _ in range(3):
            layers.append(AttentionMlpBlock(n_embedding, config.n_head))
            layers.append(Expander(block_size, n_embedding))
            n_embedding //= EMBEDDING_FAN_IN_FACTOR
            block_size *= SEQUENCE_FAN_IN_FACTOR
            layers.append(AttentionMlpBlock(n_embedding, config.n_head))
        return layers

    def forward(self, idx, targets=None, training_progress: float = 0.0):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert (
            T == self.config.initial_block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.initial_block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        self.__logger.debug(f"shape after embedding {x.shape}")
        x = self.transformer.conv_attention(x)
        self.__logger.debug(f"shape after conv_attention {x.shape}")
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

if __name__ == "__main__":
    parser = setup_args_parser()
    args = parser.parse_args()
    data_name = args.data_name
    resume_from_checkpoint = args.resume_from_checkpoint
    trainer = BaseTrainer(
        model_name,
        ConvAttentionModel(Config(vocab_size=50304), args.loglevel),
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
