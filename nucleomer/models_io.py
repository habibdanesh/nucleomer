import os
import numpy as np
import torch
from bpnetlite.bpnet import CountWrapper as BPNetCountWrapper


def load_bpnet(model_path, device='cpu'):
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
    
    return BPNetCountWrapper(torch.load(model_path, map_location=device, weights_only=False))


class ProCapNet(torch.nn.Module):
    """
    Adapted from https://github.com/kundajelab/ProCapNet.
    """

    def __init__(self, n_filters=512, n_layers=8, n_outputs=2):
        super(ProCapNet, self).__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.trimming = 557
        self.deconv_kernel_size = 75

        self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
        self.rconvs = torch.nn.ModuleList([
            torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i,
                dilation=2**i) for i in range(1, self.n_layers+1)
        ])
        self.fconv = torch.nn.Conv1d(n_filters, n_outputs,
                                    kernel_size=self.deconv_kernel_size)

        self.relus = torch.nn.ModuleList([torch.nn.ReLU() for _ in range(0, self.n_layers+1)])
        self.linear = torch.nn.Linear(n_filters, 1)

    def forward(self, X):
        start, end = self.trimming, X.shape[2] - self.trimming

        X = self.relus[0](self.iconv(X))
        for i in range(self.n_layers):
            X_conv = self.relus[i+1](self.rconvs[i](X))
            X = torch.add(X, X_conv)

        X = X[:, :, start - self.deconv_kernel_size//2 : end + self.deconv_kernel_size//2]

        X = torch.mean(X, axis=2)
        y_counts = self.linear(X).reshape(X.shape[0], 1)

        return y_counts
    

def load_procapnet(model_path, fold_num=0, device='cpu'):
    """
    Load a ProCapNet model from the specified path.
    Adapted from https://github.com/kundajelab/ProCapNet.

    Parameters
    ----------
    model_path : str
        Path to the saved model directory.

    fold_num : int, optional
        Zero-based index of the model (0 to 6 inclusive). Default is 0.
    
    device : str, optional
        Device to load the model onto ('cpu', 'cuda', etc.). Default is 'cpu'.

    Returns
    -------
    ProCapNet: The loaded model in a wrapper that returns a single count value for each input.
    """

    model = ProCapNet()

    fold_dir = f"{model_path}/fold_{fold_num}"
    torch_file = f"{fold_dir}/{os.listdir(fold_dir)[0]}"
    assert torch_file.endswith(f".procapnet_model.fold{fold_num}.state_dict.torch")
    
    model.load_state_dict(torch.load(torch_file, map_location=device, weights_only=True))
    
    return model