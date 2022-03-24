"""
Holds classes and methods for attention-based models
"""

import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.modules import get_activation

from models.modules.fusion import get_fusion_layer


class PositionalEncoding(nn.Module):
    """
    Module that performs fixed sine-cosine position encoding
    """
    def __init__(self, d_model, max_len=5000, batch_first=True):
        super(PositionalEncoding, self).__init__()
        
        self.batch_first = batch_first
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)    # add another dimension
        if not batch_first:
            pe = pe.transpose(0, 1) # make the sequence dimension the 0 dim.
            
        self.register_buffer("pe", pe)  # not a model parameter, but is saved

    def forward(self, x):
        """
        Forward pass of positional Encoding
        :param x: Sequence passed to PE [sequence_length, batch_size, embedding_dim] or [batch_size, sequence_length embedding_dim]
        :return: output [sequence_length, batch_size, embedding_dim] or [batch_size, sequence_length embedding_dim]
        """
        if not self.batch_first:
            return x + self.pe[:x.size(0), :]
        
        else:
            return x + self.pe[:, :x.size(1), :]


class SelfAttentionModel(nn.Module):
    """
    Combines a linear encoder, and a self-attention stack
    """

    def __init__(self, n_inputs, n_heads, d_embedding, d_feedforward, n_layers, dropout=0.3, activation="gelu", batch_first=True, mask=None):
        """
        Args:
            n_inputs Dimension of the input features 
            n_heads Number of attention heads
            n_hidden Dimension of the embeddings in the attention and feedforward blocks
            n_layers Number of attention blocks
            dropout Dropout rate applied inside the attention stack
            batch_first Whether input has the batch as first dimension. Default True
            mask Tensor [seq_len, seq_len] masks timesteps in the sequence when computing attention. Defaults to None
        """
        
        super(SelfAttentionModel, self).__init__()

        self.model_type = "Transformer"
        self.batch_first = batch_first
        self.d_embedding = d_embedding
        self.d_ff = d_feedforward
        self.num_features = self.d_embedding
        
        # register mask into buffer
        self.register_buffer("src_mask", mask)
        
        self.linear = nn.Linear(n_inputs, d_embedding)
        if isinstance(activation, str):
            self.act = get_activation(activation)
        else:
            self.act = activation
        
        self.pos_encoder = PositionalEncoding(d_embedding, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        enc_layers = nn.TransformerEncoderLayer(d_model=d_embedding, nhead=n_heads, dim_feedforward=d_feedforward, dropout=dropout, batch_first=batch_first, activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=enc_layers, num_layers=n_layers)

        self.init_weights() # does nothing atm

    def init_weights(self):
        pass

    def forward(self, src:torch.Tensor):
        """
        Inputs: [BS, T, N] or [T, BS, N] depending on self.batch_first
        """

        # bring input to dimension of attention model
        x = self.act(self.linear(src))

        #x = x * math.sqrt(self.n_hidden)
        
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        embedding = self.transformer_encoder(x, self.src_mask)
        #output = self.decoder(embedding)

        return embedding
    
    @staticmethod
    def _gen_square_subsequent_mask(seq_len) -> torch.Tensor:
        """
        Creates a mask that hides future time steps. Can be passed to the Self Attention Module as argument.
        :param seq_len: Length of the sequence
        :return: A tensor [seq_len, seq_len] with a triangle structure that contains 0.0 where entries are not to be masked and a large negative number where they are
        """

        mask = torch.tril(torch.ones(size=(seq_len, seq_len), device=torch.cuda.current_device()))
        mask = mask.float().masked_fill(mask == 0, float(-1.0e-9)).masked_fill(mask == 1, float(0.0))
        return mask
    

class SelfattentionLayer(nn.Module):
    """
    Reimplementation of TransformerEncoderLayer that supports selu
    """
    def __init__(self, d_embedding:int, d_feedforward: int, n_heads: int, dropout=0.3, activation="selu", norm_first=False, batch_first=True) -> None:
        super().__init__()
        
        # set dropout appropriately
        if activation == "selu":
            self.drop_mha = nn.AlphaDropout(dropout)
            self.drop_ffn = nn.AlphaDropout(dropout)
        else:
            self.drop_mha = nn.Dropout(dropout)
            self.drop_ffn = nn.Dropout(dropout)
        
        if isinstance(activation, str):
            self.act = get_activation(activation)
        else:
            self.act = activation  
            
        self.norm_first = norm_first
        self.layernorm_mha = nn.LayerNorm(d_embedding)
        self.layernorm_ffn = nn.LayerNorm(d_embedding)
        
        self.mha = nn.MultiheadAttention(embed_dim=d_embedding, num_heads=n_heads, dropout=dropout, batch_first=batch_first)
        self.ffn = PointWiseFeedForward(d_embedding=d_embedding, d_ff=d_feedforward, activation=activation, dropout=dropout)      
  
    def _mha(self, x:torch.Tensor, attn_mask=None, key_padding_mask=None):
        
        x, attn_weights = self.mha(x, x, x, attn_mask, key_padding_mask)
        x = self.drop_mha(x)
        return x
    
    def _ffn(self, x:torch.Tensor):
        
        x = self.ffn(x)
        x = self.drop_ffn(x)
        
        return x
  
    def forward(self, src:torch.Tensor, src_mask=None):
        
        x = src
        if self.norm_first:
            x = x + self._mha(self.layernorm_mha(x), src_mask)
            x = x + self._ffn(self.layernorm_ffn(x))
        else:
            x = self.layernorm_mha(x + self._mha(x, src_mask))
            x = self.layernorm_ffn(x + self._ffn(x))
        
        return x  
    

class SelfAttentionStack(nn.Module):
    """
    A stack of Selfattention layers. Like Transformer Encoder but supports more activations
    """
    def __init__(self, n_layers:int, d_embedding:int, d_feedforward: int, n_heads: int, dropout=0.3, activation="selu", norm_first=False, batch_first=True) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList([SelfattentionLayer(d_embedding=d_embedding, d_feedforward=d_feedforward, 
                                                        n_heads=n_heads, dropout=dropout, activation=activation, 
                                                        norm_first=norm_first, batch_first=batch_first) for _ in range(n_layers)])
        
    def forward(self, x:torch.Tensor, src_mask=None):
        
        for i, l in enumerate(self.layers):
            x = l(x, src_mask)
            
        return x


class SelfAttentionModelV2(nn.Module):
    """
    Combines a conv1d encoder, and a self-attention stack.
    Gives a wider variety of options for activation functions including selu
    """

    def __init__(self, n_inputs, n_heads, d_embedding, d_feedforward, n_layers, dropout=0.3, activation="gelu", encode_pos=True, batch_first=True, mask=None):
        """
        Args:
            n_inputs Dimension of the input features 
            n_heads Number of attention heads
            n_hidden Dimension of the embeddings in the attention and feedforward blocks
            n_layers Number of attention blocks
            dropout Dropout rate applied inside the attention stack
            batch_first Whether input has the batch as first dimension. Default True
            mask Tensor [seq_len, seq_len] masks timesteps in the sequence when computing attention. Defaults to None
        """
        
        super(SelfAttentionModelV2, self).__init__()

        self.model_type = "Transformer"
        self.batch_first = batch_first
        self.d_embedding = d_embedding
        self.d_ff = d_feedforward
        self.num_features = self.d_embedding
        
        self.encode_pos = encode_pos
        
        # register mask into buffer
        self.register_buffer("src_mask", mask)
        
        if activation == "selu":
            self.dropout = nn.AlphaDropout(dropout)
        else:
            self.dropout = nn.Dropout(dropout)
        
        self.conv = nn.Conv1d(n_inputs, d_embedding, kernel_size=3, padding="same")
        
        if isinstance(activation, str):
            self.act = get_activation(activation)
        else:
            self.act = activation
        
        self.pos_encoder = PositionalEncoding(d_embedding, batch_first=True)
        
        self.encoder = SelfAttentionStack(n_layers=n_layers, d_embedding=d_embedding, d_feedforward=d_feedforward, n_heads=n_heads, dropout=dropout, activation=activation, batch_first=batch_first)
        # Transformer Encoder
        #enc_layers = nn.TransformerEncoderLayer(d_model=d_embedding, nhead=n_heads, dim_feedforward=d_feedforward, dropout=dropout, batch_first=batch_first, activation=activation)
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer=enc_layers, num_layers=n_layers)

        self.init_weights() # does nothing atm

    def init_weights(self):
        pass

    def forward(self, src:torch.Tensor):
        """
        Inputs: [BS, T, N] or [T, BS, N] depending on self.batch_first
        """

        # bring input to dimension of attention model
        x = self.conv(torch.permute(src, (0, 2, 1)))
        x = torch.permute(x, (0, 2, 1))
        x = self.act(x)

        #x = x * math.sqrt(self.n_hidden)
        
        #positional encoding 
        if self.encode_pos:
            x = self.pos_encoder(x)
            
        # dropout
        x = self.dropout(x)
        
        embedding = self.encoder(x, self.src_mask)
        #output = self.decoder(embedding)

        return embedding
    
    @staticmethod
    def _gen_square_subsequent_mask(seq_len) -> torch.Tensor:
        """
        Creates a mask that hides future time steps. Can be passed to the Self Attention Module as argument.
        :param seq_len: Length of the sequence
        :return: A tensor [seq_len, seq_len] with a triangle structure that contains 0.0 where entries are not to be masked and a large negative number where they are
        """

        mask = torch.tril(torch.ones(size=(seq_len, seq_len), device=torch.cuda.current_device()))
        mask = mask.float().masked_fill(mask == 0, float(-1.0e-9)).masked_fill(mask == 1, float(0.0))
        return mask
    

    

class PointWiseFeedForward(nn.Module):
    """
    Returns a Transformer feedforward network, consists of a stack of 2 Linear layers with intermediate activation and dropout.
    """
    def __init__(self, d_embedding, d_ff, activation=nn.GELU(), dropout=0.3) -> None:
        super(PointWiseFeedForward, self).__init__()
        
        if activation == "selu":
            self.drop = nn.AlphaDropout(dropout)
        else:
            self.drop = nn.Dropout(dropout)
        
        if isinstance(activation, str):
            self.activation = self._get_activation(activation)
        else: 
            self.activation = activation
        
        self.ffn = nn.Sequential(nn.Linear(d_embedding, d_ff),  # (batch_size, seq_len, dff)
                                self.activation,
                                self.drop,
                                nn.Linear(d_ff, d_embedding),  # (batch_size, seq_len, dmodel)
                                )
                                   
    def forward(self, x):
        
        return self.ffn(x)  
    
    def _get_activation(self, name: str):
        
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
    

class CrossModalAttentionLayer(nn.Module):
    """
    Class which performs crossmodal attention between a source and a target modality. Has a single layer with attention and feedforward network
    """
    
    def __init__(self, n_heads, d_embedding, d_ff, dropout=0.5, activation="gelu", batch_first=True) -> None:
        super(CrossModalAttentionLayer, self).__init__()
        
        self.d_embedding = d_embedding
        self.d_ff = d_ff
        self.n_heads = n_heads
        
        self.dropout_mha = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        if isinstance(activation, str):
            self.act = get_activation(activation)
        else: 
            self.act = activation
        
        
        self.mha = nn.MultiheadAttention(embed_dim=self.d_embedding, num_heads=n_heads, dropout=dropout, batch_first=batch_first)
        self.ffn = PointWiseFeedForward(d_embedding=self.d_embedding, d_ff=d_ff)
        
        self.layernorm_mha = nn.LayerNorm(d_embedding)
        self.layernorm_ffn = nn.LayerNorm(d_embedding)
        
    
    def forward(self, source, target):
        
        # mha block
        attn_out, attn_weights = self.mha(target, source, source)   # optional mask
        x = self.dropout_mha(attn_out)
        out_1 = self.layernorm_mha(x + target)
        
        # ffn block
        ffn_out = self.ffn(out_1)
        ffn_out = self.act(ffn_out)
        ffn_out = self.dropout_ffn(ffn_out)
        out_2 = self.layernorm_ffn(ffn_out + out_1)
        
        return out_2, attn_weights
    

class CrossModalAttentionLayerV2(nn.Module):
    """
    Re-implementation with different block order.
    Respects the selu function 
    """
    def __init__(self, n_heads, d_embedding, d_ff, dropout=0.5, activation="gelu", batch_first=True) -> None:
        super(CrossModalAttentionLayerV2, self).__init__()
            
        self.d_embedding = d_embedding
        self.d_ff = d_ff
        self.n_heads = n_heads
        
        if activation == "selu":
            self.dropout_mha = nn.AlphaDropout(dropout)
            self.dropout_ffn = nn.AlphaDropout(dropout)
        else:
            self.dropout_mha = nn.Dropout(dropout)
            self.dropout_ffn = nn.Dropout(dropout)
        
        if isinstance(activation, str):
            self.act = get_activation(activation)
        else: 
            self.act = activation
            
        self.mha = nn.MultiheadAttention(embed_dim=self.d_embedding, num_heads=n_heads, dropout=dropout, batch_first=batch_first)
        self.ffn = PointWiseFeedForward(d_embedding=self.d_embedding, d_ff=d_ff)
        
        self.layernorm_src = nn.LayerNorm(d_embedding)
        self.layernorm_tgt = nn.LayerNorm(d_embedding)
        self.layernorm_ffn = nn.LayerNorm(d_embedding)
        
    def forward(self, source, target, mask=None):
        
        # mha block
        src = self.layernorm_src(source)
        tgt = self.layernorm_tgt(target)
        attn_out, attn_weights = self.mha(tgt, src, src)
        x = self.dropout_mha(attn_out)
        out_1 = x + attn_out
        
        # ffn block
        ffn_in = self.layernorm_ffn(out_1)
        ffn_out = self.ffn(ffn_in)
        ffn_out = self.act(ffn_out)
        ffn_out = self.dropout_ffn(ffn_out)
        out_2 = out_1 + ffn_out
        
        return out_2, attn_weights
            

class GatedCrossModalAttentionLayer(CrossModalAttentionLayerV2):
    def __init__(self, n_heads, d_embedding, d_ff, dropout=0.5, activation="gelu", batch_first=True) -> None:
        super().__init__(n_heads, d_embedding, d_ff, dropout, activation, batch_first)
        
        self.alpha = nn.parameter.Parameter(torch.Tensor([0.5]))
        
    def forward(self, source, target, mask=None):
         # mha block
        src = self.layernorm_src(source)
        tgt = self.layernorm_tgt(target)
        attn_out, attn_weights = self.mha(tgt, src, src)
        x = self.dropout_mha(attn_out)
        out_1 = self.alpha * x +  (1 - self.alpha) * attn_out # weighed sum -perhaps this helps?
        
        # ffn block
        ffn_in = self.layernorm_ffn(out_1)
        ffn_out = self.ffn(ffn_in)
        ffn_out = self.act(ffn_out)
        ffn_out = self.dropout_ffn(ffn_out)
        out_2 = out_1 + ffn_out
        
        return out_2, attn_weights
    

class GatedCrossModalAttention(nn.Module):
    """
    Follows the definition of Tsai et al. to stack multiple cross modal attention layers
    """
    def __init__(self, n_heads, d_embedding, d_ff, n_layers:int, dropout=0.5, activation="gelu", batch_first=True) -> None:
        super().__init__()
        
        self.d_embedding = d_embedding
        self.d_ff = d_ff
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([GatedCrossModalAttentionLayer(n_heads=n_heads, d_embedding=d_embedding, d_ff=d_ff, dropout=dropout, activation=activation, batch_first=True) for _ in range(n_layers)])
        
    def forward(self, source, target):
        
        src_0 = source
       
        attn_weights = {}
        x = target
       
        for i, l in enumerate(self.layers):
           x, w = l(source=src_0, target=target)
           attn_weights["crossmodal_layer{}_block".format(i + 1)] = w  
    
        return x, w             

        
class CrossModalAttention(nn.Module):
    """
    Stack of Cross Modal Layers to perform crossmodal attention
    """
    
    def __init__(self, n_heads, d_embedding, d_ff, n_layers, dropout=0.5, activation="gelu", batch_first=True) -> None:
        super(CrossModalAttention, self).__init__()
        
        self.model_type = "Transformer"
        # linear mappings
        #self.linear_src = nn.Linear(n_inputs_src, n_hidden)
        #self.relu_src = nn.ReLU(n_hidden)
        #self.linear_target = nn.Linear(n_inputs_target, n_hidden)
        #self.relu_target = nn.ReLU(n_hidden)
        
        # cross modal 
        #self.audio_to_video = None
        #self.video_to_audio = None
        
        self.d_embedding = d_embedding
        self.d_ff = d_ff
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([CrossModalAttentionLayer(d_embedding=d_embedding, d_ff=d_ff, n_heads=n_heads, dropout=dropout, activation=activation, batch_first=batch_first) for _ in range(n_layers)])
        #self.cm = nn.S
        
        
    def forward(self, source, target):
        """
        returns the output tensor and a dictionary of attn weights
        """
        
        attn_weights = {}
        
        x = target
        
        for i in range(self.n_layers):
            x, block = self.layers[i](source, x)
        
            attn_weights["crossmodal_layer{}_block".format(i + 1)] = block
            
        return x, attn_weights
    
    
class CrossModalAttentionV2(nn.Module):
    """
    Follows the definition of Tsai et al. to stack multiple cross modal attention layers
    """
    def __init__(self, n_heads, d_embedding, d_ff, n_layers:int, dropout=0.5, activation="gelu", batch_first=True) -> None:
        super().__init__()
        
        self.d_embedding = d_embedding
        self.d_ff = d_ff
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([CrossModalAttentionLayerV2(n_heads=n_heads, d_embedding=d_embedding, d_ff=d_ff, dropout=dropout, activation=activation, batch_first=True) for _ in range(n_layers)])
        
    def forward(self, source, target):
        
        src_0 = source
       
        attn_weights = {}
        x = target
       
        for i, l in enumerate(self.layers):
           x, w = l(source=src_0, target=target)
           attn_weights["crossmodal_layer{}_block".format(i + 1)] = w  
    
        return x, w 
    

class MulT(nn.Module): 
    """
    Implements the Multimodal Transformer for Unaligned Sequences (Tsai et al., 2019)
    """
    
    def __init__(self, input_size_audio: int, input_size_visual: int, n_layers: int,
                 n_heads: int, d_embedding: int, d_ff: int, dropout = 0.5, activation="gelu", batch_first=True, n_cv_layers: int=None, n_ca_layers: int=None) -> None:
        super().__init__()
        
        self.input_size_audio = input_size_audio
        self.input_size_visual = input_size_visual
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_embedding = d_embedding
        self.d_ff = d_ff
        
        self.num_features = d_embedding
        
        # split the dimension equally over the two modalities
        self.d_embedding_audio = d_embedding // 2
        self.d_embedding_visual = d_embedding // 2
        #self.n_layers_audio = n_layers // 2
        #self.n_layers_visual = n_layers // 2
        self.n_layers_audio = n_ca_layers if n_ca_layers is not None else n_layers // 2
        self.n_layers_visual = n_cv_layers if n_cv_layers is not None else n_layers // 2
        
        
        # linear embedding
        #self.linear_audio = nn.Linear(input_size_audio, self.d_embedding_audio)
        #self.linear_visual = nn.Linear(input_size_visual, self.d_embedding_visual)
        
        # 1D conv on input sequence
        self.conv_audio = nn.Conv1d(self.input_size_audio, self.d_embedding_audio, kernel_size=3, padding="same")
        self.conv_visual = nn.Conv1d(self.input_size_visual, self.d_embedding_visual, kernel_size=3, padding="same")
        
        if isinstance(activation, str):
            self.act_audio = get_activation(activation)
            self.act_visual = get_activation(activation)
        
        #self.act_audio = nn.LeakyReLU()
        #self.act_visual = nn.LeakyReLU()
        
        # positional encoding
        self.pe_audio = PositionalEncoding(self.d_embedding_audio, batch_first=batch_first)
        self.pe_visual = PositionalEncoding(self.d_embedding_visual, batch_first=batch_first)
        
        # crossmodal 
        self.cross_audio_visual = CrossModalAttention(n_heads=self.n_heads,n_layers=self.n_layers_audio, d_embedding=self.d_embedding_audio, d_ff=self.d_ff, dropout=dropout, batch_first=batch_first)
        self.cross_visual_audio = CrossModalAttention(n_heads=self.n_heads,n_layers=self.n_layers_visual, d_embedding=self.d_embedding_visual, d_ff=self.d_ff, dropout=dropout, batch_first=batch_first)
        
        # concatenation
        self.concat = get_fusion_layer("concat", [self.d_embedding_audio, self.d_embedding_visual])
        
        # self attention
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_embedding, 
                                                   nhead=self.n_heads, 
                                                   dim_feedforward=self.d_ff, 
                                                   dropout=dropout, 
                                                   activation=F.gelu, 
                                                   batch_first=batch_first)
        self.attn = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.n_layers)
        
        
    def forward(self, x):
        
        # unpack
        x_audio, x_visual = x   # [N, T, D]
        
        # mapping and PE
        #x_audio = self.act_audio(self.linear_audio(x_audio))
        #x_visual = self.act_visual(self.linear_visual(x_visual))
        
        #x_audio = torch.permute(x_audio, 0, 2, 1)    # [N, D, T]
        #x_visual = torch.permute(0, 2, 1)
        
        # permute for temporal convolution
        x_audio = self.conv_audio(torch.permute(x_audio, (0, 2, 1)))
        x_visual = self.conv_visual(torch.permute(x_visual, (0, 2, 1)))
        
        # permute back
        x_audio = torch.permute(x_audio, (0, 2, 1))
        x_visual = torch.permute(x_visual, (0, 2, 1))
        
        #assert x_audio.size(-1) == self.pe_audio.d_model
        #assert x_visual.size(-1) == self.pe_visual.d_model
        
        x_audio = self.pe_audio(x_audio)
        x_visual = self.pe_visual(x_visual)
        
      
        
        # cross modal 
        atov, _ = self.cross_audio_visual(x_audio, x_visual)
        vtoa, _ = self.cross_visual_audio(x_visual, x_audio)
        
        # fusion
        multimodal = self.concat([atov, vtoa])
        
        # self attention
        embed = self.attn(multimodal)
        
        return embed
    

class MulTV2(nn.Module): 
    """
    Implements the Multimodal Transformer for Unaligned Sequences (Tsai et al., 2019)
    """
    
    def __init__(self, input_size_audio: int, input_size_visual: int, n_layers: int,
                 n_heads: int, d_embedding: int, d_ff: int, dropout = 0.5, activation="gelu", batch_first=True, n_cv_layers: int=None, n_ca_layers: int=None) -> None:
        super().__init__()
        
        self.input_size_audio = input_size_audio
        self.input_size_visual = input_size_visual
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_embedding = d_embedding
        self.d_ff = d_ff
        
        self.num_features = d_embedding
        
        # split the dimension equally over the two modalities
        self.d_embedding_audio = d_embedding // 2
        self.d_embedding_visual = d_embedding // 2
        #self.n_layers_audio = n_layers // 2
        #self.n_layers_visual = n_layers // 2
        self.n_layers_audio = n_ca_layers if n_ca_layers is not None else n_layers // 2
        self.n_layers_visual = n_cv_layers if n_cv_layers is not None else n_layers // 2
        
        # 1D conv on input sequence
        self.conv_audio = nn.Conv1d(self.input_size_audio, self.d_embedding_audio, kernel_size=3, padding="same")
        self.conv_visual = nn.Conv1d(self.input_size_visual, self.d_embedding_visual, kernel_size=3, padding="same")
        
        if isinstance(activation, str):
            self.act_audio = get_activation(activation)
            self.act_visual = get_activation(activation)
        
        #self.act_audio = nn.LeakyReLU()
        #self.act_visual = nn.LeakyReLU()
        
        # positional encoding
        self.pe_audio = PositionalEncoding(self.d_embedding_audio, batch_first=batch_first)
        self.pe_visual = PositionalEncoding(self.d_embedding_visual, batch_first=batch_first)
        
        # crossmodal 
        self.cross_audio_visual = CrossModalAttentionV2(n_heads=self.n_heads,n_layers=self.n_layers_audio, d_embedding=self.d_embedding_audio, d_ff=self.d_ff, dropout=dropout, batch_first=batch_first)
        self.cross_visual_audio = CrossModalAttentionV2(n_heads=self.n_heads,n_layers=self.n_layers_visual, d_embedding=self.d_embedding_visual, d_ff=self.d_ff, dropout=dropout, batch_first=batch_first)
        
        # concatenation
        self.concat = get_fusion_layer("concat", [self.d_embedding_audio, self.d_embedding_visual])
        
        # self attention
        """
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_embedding, 
                                                   nhead=self.n_heads, 
                                                   dim_feedforward=self.d_ff, 
                                                   dropout=dropout, 
                                                   activation=F.gelu, 
                                                   batch_first=batch_first)
        self.attn = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.n_layers)
        """
        
        self.selfattn = SelfAttentionStack(n_layers=n_layers, n_heads=n_heads, d_embedding=d_embedding, d_feedforward=d_ff, 
                                           dropout=dropout, activation=activation, batch_first=batch_first, norm_first=False)
        
    def forward(self, x):
        
        # unpack
        x_audio, x_visual = x   # [N, T, D]
        
        # mapping and PE
        #x_audio = self.act_audio(self.linear_audio(x_audio))
        #x_visual = self.act_visual(self.linear_visual(x_visual))
        
        #x_audio = torch.permute(x_audio, 0, 2, 1)    # [N, D, T]
        #x_visual = torch.permute(0, 2, 1)
        
        # permute for temporal convolution
        x_audio = self.conv_audio(torch.permute(x_audio, (0, 2, 1)))
        x_visual = self.conv_visual(torch.permute(x_visual, (0, 2, 1)))
        
        # permute back
        x_audio = torch.permute(x_audio, (0, 2, 1))
        x_visual = torch.permute(x_visual, (0, 2, 1))
        
        #assert x_audio.size(-1) == self.pe_audio.d_model
        #assert x_visual.size(-1) == self.pe_visual.d_model
        
        x_audio = self.pe_audio(x_audio)
        x_visual = self.pe_visual(x_visual)
        
        # cross modal 
        atov, _ = self.cross_audio_visual(x_audio, x_visual)
        vtoa, _ = self.cross_visual_audio(x_visual, x_audio)
        
        # fusion
        multimodal = self.concat([atov, vtoa])
        
        # self attention
        embed = self.selfattn(multimodal)
        
        return embed
    
    
class GatedMulTV2(nn.Module): 
    """
    Implements the Multimodal Transformer for Unaligned Sequences (Tsai et al., 2019)
    """
    
    def __init__(self, input_size_audio: int, input_size_visual: int, n_layers: int,
                 n_heads: int, d_embedding: int, d_ff: int, dropout = 0.5, activation="gelu", batch_first=True, n_cv_layers: int=None, n_ca_layers: int=None) -> None:
        super().__init__()
        
        self.input_size_audio = input_size_audio
        self.input_size_visual = input_size_visual
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_embedding = d_embedding
        self.d_ff = d_ff
        
        self.num_features = d_embedding
        
        # split the dimension equally over the two modalities
        self.d_embedding_audio = d_embedding // 2
        self.d_embedding_visual = d_embedding // 2
        #self.n_layers_audio = n_layers // 2
        #self.n_layers_visual = n_layers // 2
        self.n_layers_audio = n_ca_layers if n_ca_layers is not None else n_layers // 2
        self.n_layers_visual = n_cv_layers if n_cv_layers is not None else n_layers // 2
        
        # 1D conv on input sequence
        self.conv_audio = nn.Conv1d(self.input_size_audio, self.d_embedding_audio, kernel_size=3, padding="same")
        self.conv_visual = nn.Conv1d(self.input_size_visual, self.d_embedding_visual, kernel_size=3, padding="same")
        
        if isinstance(activation, str):
            self.act_audio = get_activation(activation)
            self.act_visual = get_activation(activation)
        
        #self.act_audio = nn.LeakyReLU()
        #self.act_visual = nn.LeakyReLU()
        
        # positional encoding
        self.pe_audio = PositionalEncoding(self.d_embedding_audio, batch_first=batch_first)
        self.pe_visual = PositionalEncoding(self.d_embedding_visual, batch_first=batch_first)
        
        # crossmodal 
        self.cross_audio_visual = GatedCrossModalAttention(n_heads=self.n_heads,n_layers=self.n_layers_audio, d_embedding=self.d_embedding_audio, d_ff=self.d_ff, dropout=dropout, batch_first=batch_first)
        self.cross_visual_audio = GatedCrossModalAttention(n_heads=self.n_heads,n_layers=self.n_layers_visual, d_embedding=self.d_embedding_visual, d_ff=self.d_ff, dropout=dropout, batch_first=batch_first)
        
        # concatenation
        self.concat = get_fusion_layer("concat", [self.d_embedding_audio, self.d_embedding_visual])
        
        # self attention
        """
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_embedding, 
                                                   nhead=self.n_heads, 
                                                   dim_feedforward=self.d_ff, 
                                                   dropout=dropout, 
                                                   activation=F.gelu, 
                                                   batch_first=batch_first)
        self.attn = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.n_layers)
        """
        
        self.selfattn = SelfAttentionStack(n_layers=n_layers, n_heads=n_heads, d_embedding=d_embedding, d_feedforward=d_ff, 
                                           dropout=dropout, activation=activation, batch_first=batch_first, norm_first=False)
        
    def forward(self, x):
        
        # unpack
        x_audio, x_visual = x   # [N, T, D]
        
        # mapping and PE
        #x_audio = self.act_audio(self.linear_audio(x_audio))
        #x_visual = self.act_visual(self.linear_visual(x_visual))
        
        #x_audio = torch.permute(x_audio, 0, 2, 1)    # [N, D, T]
        #x_visual = torch.permute(0, 2, 1)
        
        # permute for temporal convolution
        x_audio = self.conv_audio(torch.permute(x_audio, (0, 2, 1)))
        x_visual = self.conv_visual(torch.permute(x_visual, (0, 2, 1)))
        
        # permute back
        x_audio = torch.permute(x_audio, (0, 2, 1))
        x_visual = torch.permute(x_visual, (0, 2, 1))
        
        #assert x_audio.size(-1) == self.pe_audio.d_model
        #assert x_visual.size(-1) == self.pe_visual.d_model
        
        x_audio = self.pe_audio(x_audio)
        x_visual = self.pe_visual(x_visual)
        
        # cross modal 
        atov, _ = self.cross_audio_visual(x_audio, x_visual)
        vtoa, _ = self.cross_visual_audio(x_visual, x_audio)
        
        # fusion
        multimodal = self.concat([atov, vtoa])
        
        # self attention
        embed = self.selfattn(multimodal)
        
        return embed
    
    
class Embedding(nn.Module):
    
    def __init__(self, d_input:int, d_embedding:int, pos_encode=True) -> None:
        super().__init__()
    
        self.conv = nn.Conv1d(d_input, d_embedding, padding="same", kernel_size=3)
        if pos_encode:
            self.pe = PositionalEncoding(d_embedding)
        else: 
            self.pe = None
        
    def forward(self, x):
        # conv over the sequence
        x = self.conv(torch.permute(x, (0, 2, 1)))
        x = torch.permute(x, (0, 2, 1))
        
        # optional position embedding
        if self.pe is not None:
            x = self.pe(x)
            
        return x