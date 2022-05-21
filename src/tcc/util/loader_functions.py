
import pandas as pd
import torch

def json_load_fn(path):
    with open(path, 'r') as f:
        return pd.read_json(f, typ='series').to_frame().T.reset_index(drop=True)

def torch_tensor_load_fn(path, device=None):
    return torch.tensor(torch.load(path), device=device)

