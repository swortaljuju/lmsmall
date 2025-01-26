import os
import torch
from torch.nn import functional as F
from baseline_gpt2 import model_name as gp2_model_name, GPTConfig, GPT
from conv_attention import model_name as conv_model_name, Config as ConvConfig, ConvAttention
from progressive_diverging import model_name as prog_model_name, Config as ProgConfig, ProgressiveDiverging
from base_trainer import BaseTrainer
import tiktoken
import time

TEST_PROMPTS = ["hello world"]
enc = tiktoken.get_encoding("gpt2")
MODELS = [(gp2_model_name, GPT(GPTConfig()) ), (conv_model_name, ConvAttention(ConvConfig())), (prog_model_name, ProgressiveDiverging(ProgConfig()))]
response = {}
space_token = enc.encode(" ")
max_length = 200
device = 'cuda'
device_type = 'cuda'

for model_name, model in MODELS:
    checkpoint_path = BaseTrainer.get_checkpoint_path(model_name)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print(f"Loaded {model_name}")
        response[model_name] = []
        for prompt in TEST_PROMPTS:
            tokens = enc.encode(prompt)
            if len(tokens) > model.get_initial_block_size():
                tokens = tokens[:model.get_initial_block_size()]
            if model_name == conv_model_name and len(tokens) < ConvConfig().initial_block_size:
                tokens += space_token * ((ConvConfig().initial_block_size - len(tokens)))
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 )
            start_time = time.time()
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen) # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :] # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            tokens = xgen[0, :max_length].tolist()
            decoded = enc.decode(tokens)
            response[model_name].append(decoded)
            print(f"elapsed time: {(time.time() - start_time) * 1000 :.2f} ms")

for idx, prompt in enumerate(TEST_PROMPTS):
    print(f"Prompt: {prompt}")
    for model_name in response.keys():
        print(f"    {model_name}: \n{response[model_name][idx]}\n")