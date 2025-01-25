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
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import matplotlib.pyplot as plt
from common_utils import setup_logger

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py


class BaseTrainer:
    base_checkpoint_path = "/tmp/lmsmall/checkpoints"

    def __init__(
        self,
        model_name: str,
        model: nn.Module,
        total_batch_size: int,
        B: int,  # micro batch size
        T: int,
        max_lr: float,
        min_lr: float,
        weight_decay: float,
        learning_rate: float,
        data_name: str,
        log_level: int,
    ):
        """
        Args:
            model_name: the name of the model
            model: the model to train
            total_batch_size: the total batch size in number of tokens
            B: the micro batch size
            T: the sequence length
            max_lr: the maximum learning rate
            min_lr: the minimum learning rate
            weight_decay: the weight decay
            learning_rate: the learning rate
            data_name: the name of the training and testing data
        """
        assert torch.cuda.is_available()
        logger = setup_logger('base_trainer', model_name, log_level)
        self.__logger = logger
        self.__initialize_ddp()
        self.__device_type = "cuda"
        torch.cuda.manual_seed(1337)
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
        self.__logger.debug(f"grad_accum_steps: {self.__grad_accum_steps}")
        if self.__master_process:
            self.__logger.info(f"total desired batch size: {total_batch_size}")
            self.__logger.info(
                f"=> calculated gradient accumulation steps: {self.__grad_accum_steps}"
            )
        self.__train_loader = DataLoaderLite(
            micro_batch_size=B,
            sequence_length=T,
            process_rank=self.__ddp_rank,
            num_processes=self.__ddp_world_size,
            master_process=self.__master_process,
            split=Split.TRAIN,
            data_name=data_name,
        )
        self.__test_loader = DataLoaderLite(
            micro_batch_size=B,
            sequence_length=T,
            process_rank=self.__ddp_rank,
            num_processes=self.__ddp_world_size,
            master_process=self.__master_process,
            split=Split.TEST,
            data_name=data_name,
        )

        torch.set_float32_matmul_precision("high")
        model.to(self.__device)
        self.__model_name = model_name
        self.__model = model
        if self.__ddp:
            self.__model = DDP(self.__model, device_ids=[self.__ddp_local_rank])
        self.__raw_model = (
            self.__model.module if self.__ddp else model
        )  # always contains the "raw" unwrapped model

        self.__max_lr = max_lr
        self.__min_lr = min_lr

        # optimize!
        self.__optimizer = self.__configure_optimizers(weight_decay, learning_rate)

        self.__prepare_checkpoint_file()
        
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.debug(f"Total number of parameters: {total_params}")

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
            self.__logger.debug(f"using DDP with rank {self.__ddp_rank}; ddp_local_rank {self.__ddp_local_rank}; world size {self.__ddp_world_size}")
        else:
            # vanilla, non-DDP run
            self.__ddp_rank = 0
            self.__ddp_local_rank = 0
            self.__ddp_world_size = 1
            self.__master_process = True
            self.__device = "cuda"
            self.__logger.info(f"using device: {self.__device}")

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
        if self.__master_process:
            self.__logger.info(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            self.__logger.info(
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

    def __get_lr(self, it, warmup_steps: int, max_steps: int):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return self.__max_lr * (it + 1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return self.__min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff starts at 1 and goes to 0
        return self.__min_lr + coeff * (self.__max_lr - self.__min_lr)

    def __prepare_checkpoint_file(self):
        # create the log directory we will write checkpoints to and log to
        os.makedirs(self.base_checkpoint_path, exist_ok=True)

    def __get_checkpoint_path(self) -> str:
        return BaseTrainer.get_checkpoint_path(self.__model_name) 
    
    @staticmethod    
    def get_checkpoint_path(model_name: str) -> str:
        return os.path.join(
            BaseTrainer.base_checkpoint_path, f"{model_name}_checkpoint.txt"
        )
    def train(self, resume_from_checkpoint: bool, warmup_steps: int, max_steps: int):
        checkpoint_path = self.__get_checkpoint_path()
        loss_per_step = {}
        start_step = 0
        if os.path.exists(checkpoint_path):
            if resume_from_checkpoint:
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                self.__logger.info(f"resuming from checkpoint {checkpoint}")
                self.__raw_model.load_state_dict(checkpoint["model_state_dict"])
                self.__optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                loss_per_step = checkpoint["loss_per_step"]
                start_step = max(loss_per_step.keys()) + 1
            else:
                # delete the checkpoint file and start training from beginning
                os.remove(checkpoint_path)
        checkpoint_steps = [i for i in range(0, max_steps, max_steps // 20)]
        if checkpoint_steps[-1] != max_steps - 1:
            checkpoint_steps.append(max_steps - 1)
        for step in range(start_step, max_steps):
            step_start_time = time.time()
            # do one step of the optimization
            self.__model.train()
            self.__optimizer.zero_grad()
            loss_accum = 0.0
            training_progress = (step + 1) / max_steps
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
                    logits, loss = self.__model(x, y, training_progress)
                    self.__logger.debug(f"loss shape: {loss.shape}, logits shape: {logits.shape}")
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN. Scale the loss here so it comes out right
                loss = loss / self.__grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()
            if self.__ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            loss_per_step[step] = loss_accum.item()
            # determine and set the learning rate for this iteration
            lr = self.__get_lr(step, warmup_steps, max_steps)
            for param_group in self.__optimizer.param_groups:
                param_group["lr"] = lr
            self.__optimizer.step()
            torch.cuda.synchronize()  # wait for the GPU to finish work
            self.__log_and_checkpoint(
                step, checkpoint_steps, step_start_time, loss_per_step, lr
            )

        if self.__ddp:
            destroy_process_group()
        
    def __log_and_checkpoint(
        self,
        step: int,
        checkpoint_steps: list[int],
        step_start_time: int,
        loss_per_step: dict[int, float],
        learning_rate: float,
    ):
        # once in a while checkpoint the model
        if (step in checkpoint_steps) and self.__master_process:
            # optionally write model checkpoints
            checkpoint_path = self.__get_checkpoint_path()
            checkpoint = {
                "model_state_dict": self.__raw_model.state_dict(),
                "optimizer_state_dict": self.__optimizer.state_dict(),
                "loss_per_step": loss_per_step,
            }
            if os.path.exists(checkpoint_path):
                # remove the old checkpoint file if it exists
                os.remove(checkpoint_path)
            torch.save(checkpoint, checkpoint_path)
        t1 = time.time()
        dt = t1 - step_start_time  # time difference in seconds
        tokens_processed = (
            self.__micro_batch_size
            * self.__sequence_length
            * self.__grad_accum_steps
            * self.__ddp_world_size
        )
        tokens_per_sec = tokens_processed / dt
        norm = torch.nn.utils.clip_grad_norm_(self.__model.parameters(), 1.0)
        if self.__master_process:
            self.__logger.info(f"step {step:5d} | loss: {loss_per_step[step]:.6f} | lr {learning_rate:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
    def train_and_test(
        self,
        resume_from_checkpoint: bool,
        training_warmup_steps: int,
        training_max_steps: int,
        testing_steps: int,
    ):
        self.train(resume_from_checkpoint, training_warmup_steps, training_max_steps)
        self.test(testing_steps)
        pass

    def test(self, max_steps: int):
        # calculate test loss
        self.__model.eval()
        with torch.no_grad():
            test_loss_accum = 0.0
            for _ in range(max_steps):
                test_start_time = time.time()
                x, y = self.__test_loader.next_batch()
                x, y = x.to(self.__device), y.to(self.__device)
                with torch.autocast(device_type=self.__device_type, dtype=torch.bfloat16):
                    logits, loss = self.__model(x, y)
                loss = loss / max_steps
                test_loss_accum += loss.detach()
                self.__logger.info(f"one step test time: {(time.time() - test_start_time)*1000:.2f}ms")
            if self.__ddp:
                dist.all_reduce(test_loss_accum, op=dist.ReduceOp.AVG)
        if self.__master_process:
            self.__logger.info(f"test loss: {test_loss_accum.item():.4f}")
