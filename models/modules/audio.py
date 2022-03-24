import torch
import torch.nn as nn
import numpy as np

from end2you.models.audio.base import Base
from end2you.models.audio import AudioModel

from typing import List
from models import get_activation


class AudioConvEncoder(nn.Module):
    """
    Encoder acting on raw audio. Mimics the End2You AudioModel interface but opens the parameter selection.
    """
    def __init__(self, input_size: int, out_channels: List[int], conv_kernel_size: List[int],
                 conv_stride: List[int], conv_padding: List[int], pool_kernel_size: List[int], pool_stride: List[int], pool_padding: List[int], 
                 activ_fn=nn.ELU(), global_avg_pool=False):
        """
        Speech emotion recognition model initialiser
        :param input_size: int Size of the input to the model
        :param out_channels: List[int] [N] output channels of the Conv Layers
        :param conv_kernel_size: List[int] [N] kernel size of the Conv layers
        :param conv_stride: List[int] [N] stride of the conv layers
        :param pool_kernel_size: List[int] kernel size of the pooling layers
        :param pool_stride: List[int] kernel size of the pooling layers
        :param global_avg_pool: Bool, compute the mean over the final dimension of the output. If not set, output is flattened instead (default).
        """

        super(AudioConvEncoder, self).__init__()
        
        self.global_avg_pool = global_avg_pool
        
        self.model, self.num_features = self._build_audio_model(input_size,
                                                                out_channels=out_channels,
                                                                conv_kernel_size=conv_kernel_size,
                                                                conv_stride=conv_stride,
                                                                conv_padding=conv_padding,
                                                                pool_kernel_size=pool_kernel_size,
                                                                pool_stride=pool_stride,
                                                                pool_padding=pool_padding,
                                                                activ_fn=activ_fn)
        
        

    def _build_audio_model(self, input_size: int, out_channels: List[int], conv_kernel_size: List[int],
                           conv_stride: List[int], conv_padding: List[int], pool_kernel_size: List[int], 
                           pool_stride: List[int], pool_padding: List[int],
                           activ_fn):
        """
        Build the convolutional
        :param input_size:
        :param arch: str, the architecture
        :return: A tuple of audio model and number of output features
        """

        in_channels = [1] + out_channels[:-1]
        #padding = ((np.array(conv_kernel_size) - 1) // 2).tolist()
        num_layers = len(in_channels)
        # set up the args for the BaseModel
        conv_args = {
            f"layer{i}": {
                "in_channels": in_channels[i],
                "out_channels": out_channels[i],
                "kernel_size": conv_kernel_size[i],
                "stride": conv_stride[i],
                "padding": conv_padding[i]

            } for i in range(num_layers)
        }

        pool_args = {
            f"layer{i}": {
                "kernel_size": pool_kernel_size[i],
                "stride": pool_stride[i],
                "padding": pool_padding[i]
            } for i in range(num_layers)
        }

        audio_model = Base(conv_layers_args=conv_args, maxpool_layers_args=pool_args, normalize=True, activ_fn=activ_fn)
        
        # output features
        if self.global_avg_pool:
            num_out_features = out_channels[-1] # number of features is determined by the channels out of the final conv op.
        else:
            # compute the number of features at the output by calculating the reduction of the input size and multiplying by number of channels of the final conv op
            conv_red_size = Base._num_out_features(input_size=input_size, conv_args=conv_args, mp_args=pool_args)
            num_out_features = conv_red_size * out_channels[-1]

        return audio_model, num_out_features

    def forward(self, x):
        """
        Forward through the audio encoder. Either flatten (default) or Average Pool the last dimension.
        Input: [N, 1, L]
        Output: [N, Cout]
        """
    
        audio_embedding = self.model(x)
        
        if len(audio_embedding.size()) > 2:
            
            if self.global_avg_pool:
                audio_embedding = torch.mean(audio_embedding, dim=-1)
            else: 
                n = audio_embedding.size(0)
                audio_embedding = audio_embedding.reshape(n, -1)
        
        return audio_embedding
    
    
def get_audio_network(input_size:int, name:str, global_pool=False, pretrained=False) -> AudioConvEncoder:
    """
    Factory helper which constructs one of several predefined audio encoder modules.
    Also loads pre-trained weights if desired and available.
    """
    network_options = { 
        "emo16": {
            "out_channels": [40, 40],
            "conv_kernel_size": [20, 40],
            "conv_stride": [1, 1],
            "conv_padding": [9, 19],
            "pool_kernel_size": [2, 10],
            "pool_stride": [2, 10],
            "pool_padding": [1, 4]
            
        },
        "emo18": {
            "out_channels": [16, 32, 64],
            "conv_kernel_size": [8, 6, 6],
            "conv_padding":  ((np.array([8, 6, 6]) - 1) // 2).tolist(),
            "conv_stride": [1, 1, 1],
            "pool_kernel_size": [10, 8, 8],
            "pool_stride": [10, 8, 8],
            "pool_padding": [0, 0, 0]   # not given so should be default value
            
        },
        "zhao19": {
            "out_channels": [64, 64, 128, 128],
            "conv_kernel_size": [3, 3, 3, 3],
            "conv_stride": [1, 1, 1, 1],
            "conv_padding": ((np.array([3, 3, 3, 3]) - 1) // 2).tolist(),
            "pool_kernel_size": [4, 4, 4, 4],
            "pool_stride": [4, 4, 4, 4],
            "pool_padding": [0, 0, 0, 0] # default value is 0. Make this optional later?
            
        },
    }
    
    if name not in network_options.keys():
        raise NotImplementedError("Network {} not available".format(name))
    
    opts = network_options[name]
    
    encoder = AudioConvEncoder(input_size=input_size, 
                            out_channels=opts["out_channels"], 
                            conv_kernel_size=opts["conv_kernel_size"],
                            conv_stride=opts["conv_stride"],
                            conv_padding=opts["conv_padding"],
                            pool_kernel_size=opts["pool_kernel_size"],
                            pool_stride=opts["pool_stride"],
                            pool_padding=opts["pool_padding"],
                            global_avg_pool=global_pool
                            )
    
    # load pretrained weights
    if pretrained:
        weights_paths = {
            "emo16": None,
            "emo18": None,
            "zhao19": "/data/eihw-gpu5/karasvin/models/pretrained/audio/RECOLA/16000Hz/zhao19_extractor.pth.tar"
        }
        
        weight_file = weights_paths[name]
        if weight_file is not None:
            print("Loading weights for pretrained E2Y audio model ...")
            weights = torch.load(weight_file)
            encoder.load_state_dict(weights)
            
    return encoder


