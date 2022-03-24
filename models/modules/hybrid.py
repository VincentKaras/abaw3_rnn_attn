"""
Hybrid architectures that include multimodal and unimodal branches

"""

import nntplib
import torch
import torch.nn as nn

from models import get_activation
from models.modules.audio import get_audio_network
from models.modules.visual import VisualConvEncoder
from models.modules.timedistributed import TimeDistributed
from models.modules.rnn import RNNModel
from models.modules.attention import MulTV2, SelfAttentionModelV2, Embedding, SelfAttentionStack, CrossModalAttentionV2
from models.modules.fusion import get_fusion_layer


class AuxNetwork(nn.Module):
    """Abstract class"""
    
    def __init__(self, audio_input_size:int, visual_input_size:int, audio_backbone:str, visual_backbone:str, temporal_model:str, 
                 visual_pretrained=True, audio_pretrained=False, audio_pool=True, batch_first=True, 
                 d_embedding=128, num_layers=4, num_bins:int=1, num_outs:int=2, dropout=0.3, activation="gelu", output_names=["VA", "CAT"]) -> None:
        super().__init__()
        
        self.audio_input_size = audio_input_size
        self.visual_input_size = visual_input_size
        self.audio_backbone = audio_backbone
        self.visual_backbone = visual_backbone
        self.audio_pretrained = audio_pretrained
        self.visual_pretrained = visual_pretrained
        self.audio_pool = audio_pool
        self.temporal_model = temporal_model
        self.batch_first = batch_first
        self.d_embedding = d_embedding
        self.num_layers = num_layers
        self.num_outputs = num_bins * num_outs
        self.dropout = dropout
        self.activation = activation
        self.output_names = output_names
        
        self.audio_net = None
        self.visual_net = None
        self.audiovisual_net = None
        
        # feature extractors
        self.visual_encoder = TimeDistributed(VisualConvEncoder(visual_input_size, name=visual_backbone, pretrained=visual_pretrained))
        self.audio_encoder = TimeDistributed(get_audio_network(audio_input_size, name=audio_backbone, global_pool=audio_pool))
        
        self.audio_embedder = None
        self.visual_embedder = None
        
        
    def forward(self, x):
        
        audio, visual = x
        
        # pass through feature extractors
        fa = self.audio_encoder(audio)
        fv = self.visual_encoder(visual)
        
        # pass through embedding layers
        emb_a = self.audio_embedder(fa)
        emb_v = self.visual_embedder(fv)
        
        
        # call each sub module
        audio_out = self.audio_net(emb_a)
        visual_out = self.visual_net(emb_v)
        audiovisual_out = self.audiovisual_net([emb_a, emb_v])
        
        # combine into a dict
        out = {"VA_audio": audio_out["VA"],
               "VA_visual": visual_out["VA"],
               "VA_audiovisual": audiovisual_out["VA"],
               "CAT_audio": audio_out["discretized"],
               "CAT_visual": visual_out["discretized"],
               "CAT_audiovisual": audiovisual_out["discretized"]
        }
        
        return out
    
    
    def freeze_extractors(self):
        
        if self.audio_pretrained:
            self.audio_encoder.requires_grad_(False)
            
        if self.visual_encoder:
            self.visual_encoder.requires_grad_(False)
            
    def unfreeze_extractors(self):
        
        self.audio_encoder.requires_grad_(True)
        self.visual_encoder.requires_grad_(True)
            
    
        

class AuxAttentionNetwork(AuxNetwork):
    """
    Subclass with attention stacks
    """
    def __init__(self, audio_input_size: int, visual_input_size: int, audio_backbone: str, visual_backbone: str, temporal_model: str, visual_pretrained=True, audio_pretrained=True, audio_pool=True, batch_first=True, d_embedding=128, d_feedforward=128, num_layers=4, num_layers_audio=4, num_layers_visual=4, num_layers_cv=2, num_layers_ca=2, num_heads=8, num_bins: int = 1, num_outs: int = 2, dropout=0.3, activation="gelu", output_names=["VA", "CAT"]) -> None:
        super().__init__(audio_input_size=audio_input_size, 
                         visual_input_size=visual_input_size, 
                         audio_backbone=audio_backbone, 
                         visual_backbone=visual_backbone, 
                         temporal_model=temporal_model, 
                         visual_pretrained=visual_pretrained, 
                         audio_pretrained=audio_pretrained, 
                         audio_pool=audio_pool, 
                         batch_first=batch_first, 
                         d_embedding=d_embedding, 
                         num_layers=num_layers, 
                         num_bins=num_bins, 
                         num_outs=num_outs, 
                         dropout=dropout, 
                         activation=activation, 
                         output_names=output_names)
        
        self.d_feedforward = d_feedforward
        self.num_heads = num_heads
        self.num_layers_visual = num_layers_visual
        self.num_layers_audio = num_layers_audio
        self.num_layers_cv = num_layers_cv
        self.num_layers_ca = num_layers_ca
        
        
        self.audio_embedder = Embedding(self.audio_encoder.model.num_features, self.d_embedding)
        self.visual_embedder = Embedding(self.visual_encoder.model.num_features, self.d_embedding)
        
        # audio network is Selfattn -> FC , so is visual
        
        self.audio_selfattn = SelfAttentionStack(n_layers=self.num_layers_audio, d_embedding=self.d_embedding, d_feedforward=self.d_feedforward,
                                                 n_heads=self.num_heads, dropout=dropout, activation=self.activation,
                                                 norm_first=False, batch_first=batch_first)
        self.visual_selfattn = SelfAttentionStack(n_layers=self.num_layers_visual, d_embedding=self.d_embedding, d_feedforward=self.d_feedforward,
                                                  n_heads=self.num_heads, dropout=self.dropout, activation=self.activation, 
                                                  norm_first=False, batch_first=batch_first)
        
        # audiovisual network is 2x Selfattn -> Crossmodal + Concat + Linear 
        self.cross_audio = CrossModalAttentionV2(n_heads=self.num_heads, d_embedding=self.d_embedding, d_ff=self.d_feedforward,
                                                 n_layers=self.num_layers_ca, dropout=dropout, activation=self.activation,
                                                 batch_first=batch_first)
        self.cross_visual = CrossModalAttentionV2(n_heads=self.num_heads, d_embedding=self.d_embedding, d_ff=self.d_feedforward,
                                                 n_layers=self.num_layers_ca, dropout=dropout, activation=self.activation,
                                                 batch_first=batch_first)
        # concatenation layer of audiovisual embeddings
        self.concat = get_fusion_layer("concat", [self.d_embedding, self.d_embedding])
        
        # subnetworks
        #self.audio_net = nn.Sequential(self.audio_selfattn,
        #                               get_activation(self.activation),
        #                               nn.Linear(self.d_embedding, self.num_outputs))
        #self.visual_net = None
        
        self.act_audio = get_activation(self.activation)
        self.act_visual = get_activation(self.activation)
        self.act_audiovisual = get_activation(self.activation)
        
        # output heads
        self.regression_audio = nn.Linear(self.d_embedding, self.num_outputs)
        self.classification_audio = nn.Linear(self.d_embedding, 24)
        
        self.regression_visual = nn.Linear(self.d_embedding, self.num_outputs)
        self.classification_visual = nn.Linear(self.d_embedding, 24)
        
        self.regression_audiovisual = nn.Linear(2 * self.d_embedding, self.num_outputs)
        self.classification_audiovisual = nn.Linear(2 * self.d_embedding, 24)
        
        
    def forward(self, x):
        
        audio, visual = x
        
        # pass through feature extractors
        fa = self.audio_encoder(audio)
        fv = self.visual_encoder(visual)
        
        # pass through embedding layers
        emb_a = self.audio_embedder(fa)
        emb_v = self.visual_embedder(fv)
        
        # pass through selfattn
        sa_a = self.audio_selfattn(emb_a)
        sa_v = self.visual_selfattn(emb_v)
        
        # branch out
        
        # audio visual
        vtoa,_ = self.cross_audio(sa_v, sa_a)
        atov, _ = self.cross_visual(sa_a, sa_v)
        
        # audio 
        audio_out_reg = self.regression_audio(self.act_audio(sa_a))
        audio_out_cat = self.classification_audio(self.act_audio(sa_a))
        
        # visual
        visual_out_reg = self.regression_visual(self.act_visual(sa_v))
        visual_out_cat = self.classification_visual(self.act_visual(sa_v))
      
        
        audiovisual = self.concat([vtoa, atov])
        audiovisual = self.act_audiovisual(audiovisual)
        
        audiovisual_out_reg = self.regression_audiovisual(audiovisual)
        audiovisual_out_cat = self.classification_audiovisual(audiovisual)
        
        
        return {
            "VA_audio": audio_out_reg,
            "VA_visual":visual_out_reg,
            "VA_audiovisual": audiovisual_out_reg,
            "CAT_audio": audio_out_cat,
            "CAT_visual":visual_out_cat,
            "CAT_audiovisual": audiovisual_out_cat
        }
        
        #return audio_out_reg, audio_out_cat, visual_out_reg, visual_out_cat, audiovisual_out_reg, audiovisual_out_cat
        
        
class AuxRecurrentNetwork(AuxNetwork):
    """
    With recurrent layers
    """
    def __init__(self, audio_input_size: int, visual_input_size: int, audio_backbone: str, visual_backbone: str, temporal_model: str, visual_pretrained=True, audio_pretrained=False, audio_pool=True, batch_first=True, d_embedding=128, num_layers=4, num_bins: int = 1, num_outs: int = 2, dropout=0.3, activation="gelu", output_names=["VA", "CAT"], bidirectional=True, num_layers_visual=1, num_layers_audio=1) -> None:
        super().__init__(audio_input_size, 
                         visual_input_size, 
                         audio_backbone, 
                         visual_backbone, 
                         temporal_model, 
                         visual_pretrained, 
                         audio_pretrained, 
                         audio_pool, 
                         batch_first, 
                         d_embedding, 
                         num_layers, 
                         num_bins, 
                         num_outs, 
                         dropout, 
                         activation, 
                         output_names)
        self.bidirectional = bidirectional
        
        self.num_layers_visual = num_layers_visual
        self.num_layers_audio = num_layers_audio
        
        self.audio_embedder = Embedding(self.audio_encoder.model.num_features, self.d_embedding, pos_encode=False)
        self.visual_embedder = Embedding(self.visual_encoder.model.num_features, self.d_embedding, pos_encode=False)
        
        
        self.concat = get_fusion_layer("concat", [self.d_embedding, self.d_embedding])
        
        # RNNS
        self.rnn_audio = RNNModel(d_input=self.d_embedding, num_layers=self.num_layers_audio, d_hidden=self.d_embedding, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn_visual = RNNModel(d_input=self.d_embedding, num_layers=self.num_layers_visual, d_hidden=self.d_embedding, dropout=self.dropout, bidirectional=self.bidirectional)
        self.rnn_audiovisual = RNNModel(d_input=self.d_embedding * 2, num_layers=self.num_layers, d_hidden=self.d_embedding, dropout=self.dropout, bidirectional=self.bidirectional)
        
        
        self.act_audio = get_activation(self.activation)
        self.act_visual = get_activation(self.activation)
        self.act_audiovisual = get_activation(self.activation)
        
        # output heads
        self.regression_audio = nn.Linear(self.rnn_audio.num_features, self.num_outputs)
        self.classification_audio = nn.Linear(self.rnn_audio.num_features, 24)
        
        self.regression_visual = nn.Linear(self.rnn_visual.num_features, self.num_outputs)
        self.classification_visual = nn.Linear(self.rnn_visual.num_features, 24)
        
        self.regression_audiovisual = nn.Linear(self.rnn_audiovisual.num_features, self.num_outputs)
        self.classification_audiovisual = nn.Linear(self.rnn_audiovisual.num_features, 24)
        
    def forward(self, x):
        
        audio, visual = x
        
        f_a = self.audio_encoder(audio)
        f_v = self.visual_encoder(visual)
        
        emb_a = self.audio_embedder(f_a)
        emb_v = self.visual_embedder(f_v)
        
        x_a, h_a = self.rnn_audio(emb_a)
        x_v, h_v = self.rnn_visual(emb_v)
        
        x_av = self.concat([emb_a, emb_v])
        x_av, h_av = self.rnn_audiovisual(x_av)
        
        # audio 
        audio_out_reg = self.regression_audio(self.act_audio(x_a))
        audio_out_cat = self.classification_audio(self.act_audio(x_a))
        
        # visual
        visual_out_reg = self.regression_visual(self.act_visual(x_v))
        visual_out_cat = self.classification_visual(self.act_visual(x_v))
      
        # audiovisual
        audiovisual = self.act_audiovisual(x_av)
        audiovisual_out_reg = self.regression_audiovisual(audiovisual)
        audiovisual_out_cat = self.classification_audiovisual(audiovisual)
        
        
        return {
            "VA_audio": audio_out_reg,
            "VA_visual":visual_out_reg,
            "VA_audiovisual": audiovisual_out_reg,
            "CAT_audio": audio_out_cat,
            "CAT_visual":visual_out_cat,
            "CAT_audiovisual": audiovisual_out_cat
        } 