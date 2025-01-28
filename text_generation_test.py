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
MODELS = [
    (gp2_model_name, GPT(GPTConfig()) ), 
    (conv_model_name, ConvAttention(ConvConfig())), 
    (prog_model_name, ProgressiveDiverging(ProgConfig())),
    ]
response = {}
space_token = enc.encode(" ")
max_length = 2048
device = 'cuda'
device_type = 'cuda'

for model_name, model in MODELS:
    checkpoint_path = BaseTrainer.get_checkpoint_path(model_name)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        print(f"Loaded {model_name}")
        response[model_name] = []
        for prompt in TEST_PROMPTS:
            start_time = time.time()
            response[model_name].append(model.generate_text(prompt, max_length, device, device_type))
            print(f"elapsed time: {(time.time() - start_time) * 1000 :.2f} ms")

for idx, prompt in enumerate(TEST_PROMPTS):
    print(f"Prompt: {prompt}")
    for model_name in response.keys():
        print(f"    {model_name}: \n{response[model_name][idx]}\n")