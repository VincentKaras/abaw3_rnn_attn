"""
Convenience module that wraps a module to execute a sequence of inputs in parallel, then restores the sequence.
"""

import torch.nn as nn
import torch


class TimeDistributed(nn.Module):
    """
    A helper class which wraps a module and executes it on every timestep.
    """

    def __init__(self, model:nn.Module, batch_first=True) -> None:
        super(TimeDistributed, self).__init__()
        
        self.batch_first = batch_first
        self.model = model
        
    def forward(self, input:torch.Tensor):
        
        assert len(input.size()) > 2 , "Input to the Time Distributed Model needs to have at least 3 dimensions, has {}".format(len(input.size()))  # [BS, T, ...]
        sample_shape = input.size()[2:]
        
        if self.batch_first:    # [BS, T, ...]
            batch_size = input.size(0)
            num_steps = input.size(1)
        else:   # [T, BS, ...]
            batch_size = input.size(1)
            num_steps = input.size(0)
            
        # combine batch and time dims [BS * T, ...]
        squashed_input = input.reshape(-1, *sample_shape)   # [BS * T, ...]
        #fwd pass
        output = self.model(squashed_input) # [BS * T, ...]
        # restore dimensions
        if self.batch_first:
            output = output.reshape(batch_size, -1, *output.size()[1:])
        else:
            output = output.reshape(-1, batch_size, *output.size()[1:])
            
        return output
