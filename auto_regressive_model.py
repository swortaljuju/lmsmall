import torch.nn as nn
import torch
from torch.nn import functional as F
import tiktoken

class AutoRegressiveModel(nn.Module):
    def __init__(self, context_window_size: int):
        super().__init__()
        self.enc = tiktoken.get_encoding("gpt2")
        self.__context_window_size = context_window_size

    def generate_text(self, prompt: str, max_length: int, device: str, device_type: str) -> str:
        self.eval()
        tokens = self.enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    model_input = xgen
                    if xgen.size(1) > self.__context_window_size:
                        model_input = xgen[:, -self.__context_window_size:]
                    logits, loss = self.forward(model_input, training_progress = 1) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 5), topk_indices is (5, 5)
                topk_probs, topk_indices = torch.topk(probs, 5, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        tokens = xgen[0, :max_length].tolist()
        return self.enc.decode(tokens)
    