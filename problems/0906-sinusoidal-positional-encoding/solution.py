import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    # TODO: return a (seq_len, d_model) tensor of sinusoidal positional encodings
    position = torch.arange(0, seq_len, 1).reshape(seq_len,1)
    div_term = -2*math.log(10000)*torch.arange(0, d_model, 2)/d_model
    div_term = torch.exp(div_term)
    broadcasted = position*div_term
    sinusoidal_embeddings = torch.zeros((seq_len,d_model))
    sinusoidal_embeddings[:,::2] = torch.sin(broadcasted)
    sinusoidal_embeddings[:,1::2] = torch.cos(broadcasted)
    return sinusoidal_embeddings
