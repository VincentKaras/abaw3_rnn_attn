import torch
import torch.nn as nn
from models import get_activation
from models.modules.audio import get_audio_network
from models.modules.attention import SelfAttentionModelV2
from models.modules.timedistributed import TimeDistributed
from models.modules.rnn import RNNModel
from typing import Tuple



class AudioNetwork(nn.Module):
    """
    Superclass for Unimodal audio models consisting of an encoder, a s2s model, (an activation) and a linear output layer
    """
    
    def __init__(self, audio_input_size, audio_encoder:str, d_embedding:int, n_layers:int, s2s_model:str, dropout=0.3, activation="selu", output_names=["VA"], pretrained=True, batch_first=True) -> None:
        super().__init__()
        
        self.encoder = TimeDistributed(get_audio_network(input_size=audio_input_size, name=audio_encoder, pretrained=pretrained, global_pool=True))
        
        
        self.d_embedding = d_embedding
        self.dropout = dropout
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.pretrained = pretrained
        self.output_names = output_names
        
        self.s2s_model = s2s_model
        self.activation = activation    # string 
        
        self.s2s = None # set by subclasses
        #self.linear = nn.Linear(d_embedding, 2)
        self.linear_continuous = None #nn.Linear(d_embedding, 2)
        self.linear_categorical = None
        
    def _get_seq2seq(self) -> Tuple[nn.Module, int]:
        return None, 0
        
    def forward(self, x):
        
        audio, visual = x   # unpack
        
        emb = self.encoder(audio)
        
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
    
    
class AudioAttentionNetwork(AudioNetwork):
    """
    Audio model with self attention 
    """
    
    def __init__(self, audio_input_size, audio_encoder: str, d_embedding: int, d_ff: int, n_heads: int, n_layers: int, s2s_model: str, dropout=0.3, activation="selu", output_names=["VA"], pretrained=True, batch_first=True) -> None:
        super().__init__(audio_input_size=audio_input_size, 
                         audio_encoder=audio_encoder, 
                         d_embedding=d_embedding, 
                         n_layers=n_layers, 
                         s2s_model=s2s_model, 
                         activation=activation, 
                         dropout=dropout,
                         output_names=output_names,  
                         pretrained=pretrained, 
                         batch_first=batch_first)
        
        self.d_ff = d_ff
        self.n_heads = n_heads
        
        self.s2s, num_out_features = self._get_seq2seq()
        
        self.linear_continuous = nn.Linear(num_out_features, 2)
        self.linear_categorical = nn.Linear(num_out_features, 24)
    
    def _get_seq2seq(self):
        
        model = SelfAttentionModelV2(n_inputs=self.encoder.model.num_features, n_heads=self.n_heads, d_embedding=self.d_embedding, d_feedforward=self.d_ff, n_layers=self.n_layers,
                                    dropout=self.dropout, activation=self.activation, batch_first=self.batch_first)
        
        return model, model.num_features
        
        
class AudioRecurrentNetwork(AudioNetwork):
    """
    Audio model with recurrent 
    """
    def __init__(self, audio_input_size, audio_encoder: str, d_embedding: int, n_layers: int, s2s_model: str, dropout=0.3, activation="selu", output_names=["VA"], bidirectional=True, pretrained=True, batch_first=True) -> None:
        super().__init__(audio_input_size, audio_encoder, d_embedding, n_layers, s2s_model, dropout, activation, output_names, pretrained, batch_first)
        
        self.bidirectional = bidirectional
        
        self.s2s, num_out_features = self._get_seq2seq()
        
        self.linear_continuous = nn.Linear(num_out_features, 2)
        self.linear_categorical = nn.Linear(num_out_features, 24)
        
    def _get_seq2seq(self) -> Tuple[nn.Module, int]:
        
        model = RNNModel(num_layers=self.n_layers, d_input=self.encoder.model.num_features, d_hidden=self.d_embedding, dropout=self.dropout,
                         bidirectional=self.bidirectional, batch_first=self.batch_first)
        
        return model, model.num_features