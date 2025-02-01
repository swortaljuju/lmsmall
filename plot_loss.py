import matplotlib.pyplot as plt
from baseline_gpt2 import model_name as gp2_model_name
from conv_attention import model_name as conv_model_name
from progressive_diverging import model_name as prog_model_name
from base_trainer import BaseTrainer
import os
import torch

for model_name, color in [(gp2_model_name, 'blue'), (conv_model_name, 'green'), (prog_model_name, 'red')]:
    checkpoint_path = BaseTrainer.get_checkpoint_path(model_name)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        loss_per_step = checkpoint["loss_per_step"]
        loss_array = sorted([[k, v] for k, v in loss_per_step.items()])
        plt.plot(*zip(*loss_array), label = model_name, color = color )

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Training Loss Curve")
plt.legend()
plt.show()