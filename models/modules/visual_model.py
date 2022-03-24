import torch
import torch.nn as nn
from models import get_activation

from models.modules.visual import VisualConvEncoder
from models.modules.timedistributed import TimeDistributed
from models.modules.attention import SelfAttentionModelV2
from models.modules.rnn import RNNModel
from typing import Tuple


class VisualNetwork(nn.Module):
    
    def __init__(self, visual_input_size:int, visual_encoder:str, d_embedding:int, n_layers:int, s2s_model:str, dropout=0.3, activation="selu", output_names=["VA"], pretrained=True, batch_first=True) -> None:
        super().__init__()
        
        self.d_embedding = d_embedding
        self.dropout = dropout
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.pretrained = pretrained
        
        self.s2s_model = s2s_model
        self.activation = activation    # string 
        
        self.output_names = output_names
        
        self.encoder = TimeDistributed(VisualConvEncoder(visual_input_size, name=visual_encoder, pretrained=pretrained, batch_first=self.batch_first))
        
        self.s2s = None # set by subclasses
        self.linear_continuous = None #nn.Linear(d_embedding, 2)
        self.linear_categorical = None
        
        
    def _get_seq2seq_model() -> Tuple[nn.Module, int]:
        return None, 0
    
    
    def forward(self, x):
        
        audio, visual = x   # unpack
        
        emb = self.encoder(visual)
        
        if "rnn" in self.s2s_model:
            
            emb, _ = self.s2s(emb)
            out = self.linear_continuous(emb)
            out_cat = self.linear_categorical(emb)
        else:
            
            x = self.s2s(emb)
            x = get_activation(self.activation)(x)
            out = self.linear_continuous(x)
            out_cat = self.linear_categorical(x)
            
        return {self.output_names[0]: out,
                "discretized": out_cat}
    
     
    def freeze_extractors(self):
        """
        Freeze pretrained feature extractor
        """
        
        if self.pretrained:
            self.encoder.requires_grad_(False)
            
            
    def unfreeze_extractors(self):
        
        self.encoder.requires_grad_(True)
    

class VisualAttentionNetwork(VisualNetwork):
    """
    Subclass with attention
    """
    
    def __init__(self, visual_input_size: int, visual_encoder: str, d_embedding: int, d_ff:int, n_heads:int, n_layers: int, s2s_model: str, dropout=0.3, activation="selu", output_names=["VA"], pretrained=True, batch_first=True) -> None:
        super().__init__(visual_input_size=visual_input_size, 
                         visual_encoder=visual_encoder, 
                         d_embedding=d_embedding, 
                         n_layers=n_layers, 
                         s2s_model=s2s_model, 
                         dropout=dropout, 
                         activation=activation, 
                         output_names=output_names, 
                         pretrained=pretrained, 
                         batch_first=batch_first)
        
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # rest of the network        
        self.s2s, num_out_features = self._get_seq2seq()
        
        self.linear_continuous = nn.Linear(num_out_features, 2)
        self.linear_categorical = nn.Linear(num_out_features, 24)
        
    def _get_seq2seq(self) -> nn.Module:
        
        model = SelfAttentionModelV2(n_inputs=self.encoder.model.num_features, n_heads=self.n_heads, d_embedding=self.d_embedding, d_feedforward=self.d_ff, n_layers=self.n_layers,
                                    dropout=self.dropout, activation=self.activation, batch_first=self.batch_first)
        
        return model, model.num_features
        
        
class VisualRecurrentNetwork(VisualNetwork):
    def __init__(self, visual_input_size: int, visual_encoder: str, d_embedding: int, n_layers: int, s2s_model: str, dropout=0.3, activation="selu", output_names=["VA"], pretrained=True, batch_first=True, bidirectional=True) -> None:
        super().__init__(visual_input_size, 
                         visual_encoder, 
                         d_embedding, 
                         n_layers, 
                         s2s_model, 
                         dropout, 
                         activation, 
                         output_names, 
                         pretrained, 
                         batch_first)
        self.bidirectional = bidirectional
    
        self.s2s, num_out_features = self._get_seq2seq()
        
        self.linear_continuous = nn.Linear(num_out_features, 2)
        self.linear_categorical = nn.Linear(num_out_features, 24)
        
    def _get_seq2seq(self) -> Tuple[nn.Module, int]:
        
        model = RNNModel(num_layers=self.n_layers, d_input=self.encoder.model.num_features, d_hidden=self.d_embedding, dropout=self.dropout,
                         bidirectional=self.bidirectional, batch_first=self.batch_first)
        
        return model, model.num_features