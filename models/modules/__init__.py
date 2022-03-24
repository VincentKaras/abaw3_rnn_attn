
import torch.nn as nn

def get_activation(name: str):
    """
    Helper function which returns an activation layer based on a string.
    Supports ReLU, LeakyReLU, GELU and SELU
    """

    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "leakyrelu":
        return nn.LeakyReLU()
    elif name.lower() == "gelu":  
        return nn.GELU()
    elif name.lower() == "selu":
        return nn.SELU()
    else:
        raise ValueError("Activation {} not available".format(name))
    
