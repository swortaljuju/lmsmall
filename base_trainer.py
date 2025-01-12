import inspect
import torch
import torch.nn as nn
import os
import math
import time
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataloader import DataLoaderLite, Split
from prepare_math_reasoning_data import MATH_REASONING_DATA_NAME
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken


# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        total_batch_size: int,
        B: int,  # micro batch size
        T: int,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
        weight_decay: float,
        learning_rate: float,
    ):
        """
        Args:
            model: the model to train
            total_batch_size: the total batch size in number of tokens
            B: the micro batch size
            T: the sequence length
            max_lr: the maximum learning rate
            min_lr: the minimum learning rate
            warmup_steps: the number of warmup steps
            max_steps: the maximum number of steps
            weight_decay: the weight decay
            learning_rate: the learning rate
        """
        assert torch.cuda.is_available()
        self.__initialize_ddp()
        self.__device_type = "cuda"
        torch.cuda.manual_seed(1337)
        self.__enc = tiktoken.get_encoding("gpt2")
        self.__total_batch_size = total_batch_size
        self.__micro_batch_size = B
        self.__sequence_length = T
        assert (
            self.__total_batch_size
            % (self.__micro_batch_size * self.__sequence_length * self.__ddp_world_size)
            == 0
        ), "make sure total_batch_size is divisible by B * T * ddp_world_size"

        self.__grad_accum_steps = self.__total_batch_size // (
            B * T * self.__ddp_world_size
        )
        if self.__master_process:
            print(f"total desired batch size: {total_batch_size}")
            print(
                f"=> calculated gradient accumulation steps: {self.__grad_accum_steps}"
            )

        self.__train_loader = DataLoaderLite(
            micro_batch_size=B,
            sequence_length=T,
            process_rank=self.__ddp_rank,
            num_processes=self.__ddp_world_size,
            master_process=self.__master_process,
            split=Split.TRAIN,
            data_name=MATH_REASONING_DATA_NAME,
        )
        self.__test_loader = DataLoaderLite(
            micro_batch_size=B,
            sequence_length=T,
            process_rank=self.__ddp_rank,
            num_processes=self.__ddp_world_size,
            master_process=self.__master_process,
            split=Split.TEST,
            data_name=MATH_REASONING_DATA_NAME,
        )

        torch.set_float32_matmul_precision("high")
        model.to(self.__device)
        self.__model = model
        if self.__ddp:
            self.__model = DDP(self.__model, device_ids=[self.__ddp_local_rank])
        self.__raw_model = (
            self.__model.module if self.__ddp else model
        )  # always contains the "raw" unwrapped model

        self.__max_lr = max_lr
        self.__min_lr = min_lr
        self.__warmup_steps = warmup_steps
        self.__max_steps = max_steps

        # optimize!
        self.__optimizer = self.__configure_optimizers(weight_decay, learning_rate)

        self.__prepare_log_file()

    def __initialize_ddp(self):
        # set up DDP (distributed data parallel).
        # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
        self.__ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
        if self.__ddp:
            # use of DDP atm demands CUDA, we set the device appropriately according to rank
            assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
            init_process_group(backend="nccl")
            self.__ddp_rank = int(os.environ["RANK"])
            self.__ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.__ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.__device = f"cuda:{self.__ddp_local_rank}"
            torch.cuda.set_device(self.__device)
            self.__master_process = (
                self.__ddp_rank == 0
            )  # this process will do logging, checkpointing etc.
        else:
            # vanilla, non-DDP run
            self.__ddp_rank = 0
            self.__ddp_local_rank = 0
            self.__ddp_world_size = 1
            self.__master_process = True
            self.__device = "cuda"
            print(f"using device: {self._device}")

    def __configure_optimizers(self, weight_decay: float, learning_rate: float):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.__raw_model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self._master_process:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=fused_available,
        )
        return optimizer

    def __get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.__warmup_steps:
            return self.__max_lr * (it + 1) / self.__warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.__max_steps:
            return self.__min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.__warmup_steps) / (
            self.__max_steps - self.__warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff starts at 1 and goes to 0
        return self.__min_lr + coeff * (self.__max_lr - self.__min_lr)

    def __prepare_log_file(self):
        # create the log directory we will write checkpoints to and log to
        self.__log_dir = "log"
        os.makedirs(self.__log_dir, exist_ok=True)
        self.__log_file = os.path.join(self.__log_dir, f"log.txt")
        with open(self.__log_file, "w") as f:  # open for writing to clear the file
            pass

    def train(self):
        for step in range(self.__max_steps):
            t0 = time.time()
            last_step = step == self.__max_steps - 1

            # once in a while evaluate our validation loss
            if step % 250 == 0 or last_step:
                self.__model.eval()
                val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    val_loss_steps = 20
                    for _ in range(val_loss_steps):
                        x, y = val_loader.next_batch()
                        x, y = x.to(self.__device), y.to(self.__device)
                        with torch.autocast(
                            device_type=self.__device_type, dtype=torch.bfloat16
                        ):
                            logits, loss = self.__model(x, y)
                        loss = loss / val_loss_steps
                        val_loss_accum += loss.detach()
                if self.__ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if self.__master_process:
                    print(f"validation loss: {val_loss_accum.item():.4f}")
                    with open(self.__log_file, "a") as f:
                        f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                    if step > 0 and (step % 5000 == 0 or last_step):
                        # optionally write model checkpoints
                        checkpoint_path = os.path.join(
                            self.__log_dir, f"model_{step:05d}.pt"
                        )
                        checkpoint = {
                            "model": self.__raw_model.state_dict(),
                            "config": self.__raw_model.config,
                            "step": step,
                            "val_loss": val_loss_accum.item(),
                        }
                        # you might also want to add optimizer.state_dict() and
                        # rng seeds etc., if you wanted to more exactly resume training
                        torch.save(checkpoint, checkpoint_path)

            # once in a while generate from the model (except step 0, which is noise)
            if (step > 0 and step % 250 == 0) or last_step:
                self.__model.eval()
                num_return_sequences = 4
                max_length = 32
                tokens = self.__enc.encode("Hello, I'm a language model,")
                tokens = torch.tensor(tokens, dtype=torch.long)
                tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
                xgen = tokens.to(self.__device)
                sample_rng = torch.Generator(device=self.__device)
                sample_rng.manual_seed(42 + self.__ddp_rank)
                while xgen.size(1) < max_length:
                    # forward the model to get the logits
                    with torch.no_grad():
                        with torch.autocast(
                            device_type=self.__device_type, dtype=torch.bfloat16
                        ):
                            logits, loss = self.__model(xgen)  # (B, T, vocab_size)
                        # take the logits at the last position
                        logits = logits[:, -1, :]  # (B, vocab_size)
                        # get the probabilities
                        probs = F.softmax(logits, dim=-1)
                        # do top-k sampling of 50 (huggingface pipeline default)
                        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                        # select a token from the top-k probabilities
                        # note: multinomial does not demand the input to sum to 1
                        ix = torch.multinomial(
                            topk_probs, 1, generator=sample_rng
                        )  # (B, 1)
                        # gather the corresponding indices
                        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                        # append to the sequence
                        xgen = torch.cat((xgen, xcol), dim=1)
                # print the generated text
                for i in range(num_return_sequences):
                    tokens = xgen[i, :max_length].tolist()
                    decoded = self.__enc.decode(tokens)
                    print(f"rank {self.__ddp_rank} sample {i}: {decoded}")

            # do one step of the optimization
            self.__model.train()
            self.__optimizer.zero_grad()
            loss_accum = 0.0
            for micro_step in range(self.__grad_accum_steps):
                x, y = self.__train_loader.next_batch()
                x, y = x.to(self.__device), y.to(self.__device)
                # added after video, this field is also used by the forward pass.
                if self.__ddp:
                    self.__model.require_backward_grad_sync = (
                        micro_step == self.__grad_accum_steps - 1
                    )
                with torch.autocast(
                    device_type=self.__device_type, dtype=torch.bfloat16
                ):
                    logits, loss = self.__model(x, y)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN. Scale the loss here so it comes out right
                loss = loss / self.__grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()
            if self.__ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(self.__model.parameters(), 1.0)
            # determine and set the learning rate for this iteration
            lr = self.__get_lr(step)
            for param_group in self.__optimizer.param_groups:
                param_group["lr"] = lr
            self.__optimizer.step()
            if self.__device_type == "cuda":
                torch.cuda.synchronize()  # wait for the GPU to finish work
            t1 = time.time()
            dt = t1 - t0  # time difference in seconds
            tokens_processed = (
                self.__micro_batch_size
                * self.__sequence_length
                * self.__grad_accum_steps
                * self.__ddp_world_size
            )
            tokens_per_sec = tokens_processed / dt
            if self.__master_process:
                print(
                    f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
                )
                with open(self.__log_file, "a") as f:
                    f.write(f"{step} train {loss_accum.item():.6f}\n")

            if self.__ddp:
                destroy_process_group()
