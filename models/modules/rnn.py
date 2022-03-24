import torch.nn as nn
import torch


class RNNModel(nn.Module):
    
    def __init__(self, num_layers, d_input, d_hidden, dropout=0.0, bidirectional=False, batch_first=True) -> None:
        super().__init__()
        
        self.rnn = torch.nn.LSTM(input_size=d_input, 
                                 hidden_size=d_hidden,
                                 num_layers=num_layers,
                                 bidirectional=bidirectional,
                                 batch_first=batch_first,
                                 dropout=dropout)
        
        if bidirectional:
            self.num_features = d_hidden * 2
        else:
            self.num_features = d_hidden
        
    def forward(self, x):
        
        self.rnn.flatten_parameters()
        
        return self.rnn(x)  # output and cell states
        