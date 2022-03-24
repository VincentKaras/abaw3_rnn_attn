"""
Stores some convenience functions for accessing fusion layers
"""

from end2you.models.multimodal.fusion.concat import ConcatFusion
from end2you.models.multimodal.fusion.attention import AttentionFusion

import torch.nn as nn


def get_fusion_layer(method: str, *args, **kwargs) ->nn.Module:
    
    """
    Modified version of the E2Y wrapper class that provides fusion layers
    """

    fusion_dict = {
        "concat": ConcatFusion,
        "attention": AttentionFusion,
    }
    
    assert method in fusion_dict.keys()
    
    fusion_layer = fusion_dict[method](*args, **kwargs)

    return fusion_layer