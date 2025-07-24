import torch
from bpnetlite.bpnet import CountWrapper as BPNetCountWrapper


def load_bpnet(model_path: str, device: str = 'cpu') -> BPNetCountWrapper:
    """
    Load a BPNet model from the specified path.

    Parameters
    ----------
    model_path : str
        Path to the saved BPNet model.
    device : str, optional
        Device to load the model onto ('cpu', 'cuda', etc.). Default is 'cpu'.

    Returns
    -------
    BPNetCountWrapper: The loaded BPNet model in a wrapper that returns a single count value for each input.
    """
    
    model = BPNetCountWrapper(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    return model