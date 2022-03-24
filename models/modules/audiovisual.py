from re import S
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import math
from models.modules.rnn import RNNModel

from models.modules.visual import VisualConvEncoder
from models.modules.audio import AudioConvEncoder, get_audio_network
from models.modules.timedistributed import TimeDistributed
from models.modules.attention import MulTV2, SelfAttentionModel, MulT, GatedMulTV2
from models.modules.fusion import get_fusion_layer

from typing import List

from models import MODEL_PATHS
from models import get_activation



"""
Defines various modules that are combined into models

Vincent Karas, 01/2022
"""


class JointFeatureEncoder(nn.Module):
    pass

class LinearDecoder(nn.Module):
    """
    Linear Decoder containing Dropout and Linear layer
    """
    def __init__(self, n_inputs, n_outputs, dropout=0.3):
        super(LinearDecoder, self).__init__()
        self.model_type = "Linear"
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        x = self.dropout(x)
        output = self.decoder(x)

        return output




        
class AudioVisualNetwork(nn.Module):
    """
    Combines an audio and visual encoder, a fusion layer and an output for valence and arousal classification.
    This is an abstract base class, the fusion layer and outputs should be set by its subclasses.
    """
    def __init__(self, audio_input_size:int, visual_input_size:int, audio_backbone:str, visual_backbone:str, temporal_model:str, 
                 visual_pretrained=True, audio_pretrained=False, audio_pool=True, batch_first=True, 
                 d_embedding=128, num_layers=4, num_bins:int=1, num_outs:int=2, dropout=0.3, activation="gelu", output_names=["VA", "CAT"]) -> None:
        super().__init__()
        
        # names of the outputs. Model should return this many tensors
        self.output_names = output_names
        
        self.temporal_model = temporal_model
        
        self.num_outputs = num_bins * num_outs # make this an attribute so it can be accessed
        
        self.d_embedding = d_embedding
        self.num_layers = num_layers
        
        self.dropout = dropout
        self.batch_first = batch_first
        
        
        #############################
        ######## BACKBONES ##########
        #############################
        
        self.audio_network = TimeDistributed(get_audio_network(input_size=audio_input_size, name=audio_backbone, global_pool=audio_pool, pretrained=audio_pretrained))
        self.num_audio_features = self.audio_network.model.num_features
        self.audio_pretrained = audio_pretrained
        
        
        self.visual_network = TimeDistributed(VisualConvEncoder(input_size=visual_input_size, name=visual_backbone, pretrained=visual_pretrained, batch_first=batch_first))
        self.num_visual_features = self.visual_network.model.num_features
        self.visual_pretrained = visual_pretrained
          
        # activation 
        if isinstance(activation, str):
            self.activation = get_activation(activation)
        else: 
            self.activation = activation
        
        
        # initialise the seq2seq model
        # TODO refactor this into a method outside for more options and flex
        """
        self.seq2seq_model, num_out_features = self._get_seq2seq_model(    # args are now handled through attributes - allows for overloading
        
            name=temporal_model, 
            num_layers=num_layers, 
            d_embedding=d_embedding, 
            dropout=dropout, 
            batch_first=batch_first, 
            activation=activation)
        """
        # placeholders, to be overwritten by the subclass
        self.seq2seq_model = None
        self.linear_continuous = None
        self.linear_categorical = None
        
        
        # make the linear output layer
        
        #self.linear = nn.Linear(num_out_features, self.num_outputs)
        
        
    def _get_seq2seq_model(self, d_embedding=128, num_layers=4, num_heads=4, dropout=0.3, activation="gelu", batch_first=True) -> tuple[nn.Module, int]:
        """
        Provides a seq2seq model.
        Receives the outputs of the feature extractors and combines them into a joint feature vector
        Returns: A tuple of a nn Module and its output dimension (int)
        """
        
        name = self.temporal_model

        # concatenation of features followed by self-attention
        if name == "transformer":
            
            print("Creating the concatenation + self attention model block")
            
            # initialise the fusion layer - is just concat for now
            fusion_layer = get_fusion_layer("concat", num_feats_modality=[self.num_audio_features, self.num_visual_features])
            # get the number of features that will be put into the seq2seq model
            seq2seq_input_features = fusion_layer.num_features
            
            assert sum([self.audio_network.model.num_features, self.visual_network.model.num_features]) == fusion_layer.num_features
            #assert multimodal.size(-1) == self.fusion_layer.num_features, "Feature dimension of multimodal embedding does not match that expected by the fusion layer: {}, {}".format(multimodal.size(), self.fusion_layer.num_features)
        
    
            # standard args, might expand later
            n_heads = num_heads
            n_hidden = d_embedding
            n_layers = num_layers
            
            selfattnmodel = SelfAttentionModel(n_inputs=seq2seq_input_features, n_heads=n_heads, d_embedding=n_hidden, n_layers=n_layers)

            model = nn.Sequential(fusion_layer,
                                  selfattnmodel,
                                  )

            return model, selfattnmodel.num_features
        
        
        # MulT - symmetric crossmodal attention followed by concatenation and self attention
        if name == "mult":
            
            print("Creating the multimodal Transformer")
            
            mult = MulT(input_size_audio=self.audio_network.model.num_features,
                        input_size_visual=self.visual_network.model.num_features,
                        n_layers=num_layers,
                        n_heads=num_heads,
                        d_embedding=d_embedding,
                        d_ff=d_embedding,
                        dropout=dropout,
                        batch_first=True)
            # add a LeakyReLU as activation function to the output
            #model = nn.Sequential(mult, 
            #                      torch.nn.LeakyReLU())
            
            #return model, mult.num_features
            
            return mult, mult.num_features
        
    
        if name == "rnn":
            
            print("Creating the concatenation + rnn model block")
            
            # initialise the fusion layer - is just concat for now
            fusion_layer = get_fusion_layer("concat", num_feats_modality=[self.num_audio_features, self.num_visual_features])
            assert sum([self.audio_network.model.num_features, self.visual_network.model.num_features]) == fusion_layer.num_features
            # get the number of features 
            fusion_features = fusion_layer.num_features
            # a linear layer to avoid high-dim rnn model
            linear = nn.Linear(fusion_features, d_embedding)
            
            if isinstance(activation, str):
                act =  get_activation(activation)
            else:
                act = activation
         
            rnnmodel = RNNModel(d_input=d_embedding, d_hidden=d_embedding, dropout=dropout, num_layers=num_layers, batch_first=batch_first)
            
            model =  nn.Sequential(
                                   fusion_layer,
                                   linear,
                                   act, 
                                   rnnmodel
                                   )
            
            return model, rnnmodel.num_features
        
        elif name == "birnn":
            # construct bidirectional rnn
            
            print("Creating the concatenation + bidirectional rnn model block")
            
            # initialise the fusion layer - is just concat for now
            fusion_layer = get_fusion_layer("concat", num_feats_modality=[self.num_audio_features, self.num_visual_features])
            assert sum([self.audio_network.model.num_features, self.visual_network.model.num_features]) == fusion_layer.num_features
            # get the number of features 
            fusion_features = fusion_layer.num_features
            # a linear layer to avoid high-dim rnn model
            linear = nn.Linear(fusion_features, d_embedding)
            
            if isinstance(activation, str):
                act =  get_activation(activation)
            else:
                act = activation
         
            rnnmodel = RNNModel(d_input=d_embedding, d_hidden=d_embedding, dropout=dropout, num_layers=num_layers, bidirectional=True, batch_first=batch_first)
            
            model =  nn.Sequential(
                                   fusion_layer,
                                   linear,
                                   act, 
                                   rnnmodel
                                   )
            
            return model, rnnmodel.num_features
        
        
        if name == "linear":
            
            print("concat + linear layer")
            
            fusion_layer = get_fusion_layer("concat", num_feats_modality=[self.num_audio_features, self.num_visual_features])
            
            linear = torch.nn.Linear(fusion_layer.num_features, d_embedding)
            
            if isinstance(activation, str):
                act =  get_activation(activation)
            else:
                act = activation
            
            model = nn.Sequential(fusion_layer, 
                                  act,    # assuming the inputs did not get a nonlinearity before concatenation!
                                  linear,
                                  )
            
            return model, d_embedding


    def forward(self, model_input:list):
        """
        Forward pass of the fusion model
        
        :param model_input (list): List with audio and visual input tensor.
            visual_input (BS x T x C x H x W)
            audio_input (BS x S x 1 x T)
            
        returns: Model output with dimens
        """
        
        assert type(model_input) in [list, tuple]
        
        # unpack
        audio_input, visual_input = model_input
        
        # audio input should already be good due to torchaudio.load inserting channel dim - do another reshape if not.
        assert len(audio_input.size()) == 4, "Audio input does not have 4 dimensions but {}".format(audio_input.size())
        
        # what about visual input?
        assert len(visual_input.size()) == 5, "Visual input does not have 5 dimensions but {}".format(visual_input.size())
        
        # audio and visual network should contain TimeDistributed layers that automatically resize the audio, visual inputs
        audio_embedding = self.audio_network(audio_input)
        
        # pool last dim auf audio embedding
        #if len(audio_embedding.size()) == 4:
        #    batch_size, seq_length =  audio_embedding.size()[:2]
        #    audio_embedding = audio_embedding.sum(-1)
        
        visual_embedding = self.visual_network(visual_input)
        
        assert len(audio_embedding.size()) == 3, "Audio embedding does not have 3 dimensions but {} with shape {}".format(len(audio_embedding.size()), audio_embedding.size())  
        assert len(visual_embedding.size()) == 3, "Visual embedding does not have 3 dimensions but {} with shape {}".format(len(visual_embedding.size()), visual_embedding.size())          
        
        # fuse the embeddings
        #multimodal = self.fusion_layer([audio_embedding, visual_embedding])
        
        #assert sum([self.audio_network.model.num_features, self.visual_network.model.num_features]) == self.fusion_layer.num_features
        #assert multimodal.size(-1) == self.fusion_layer.num_features, "Feature dimension of multimodal embedding does not match that expected by the fusion layer: {}, {}".format(multimodal.size(), self.fusion_layer.num_features)
        
        
        if "rnn" in self.temporal_model:
            emb, _ = self.seq2seq_model([audio_embedding, visual_embedding])    # remove cell states
            
            output = self.linear_continuous(emb)   # no additional activation needed
            
            discrete_output = self.linear_categorical(emb)
            
        else: 
            emb = self.seq2seq_model([audio_embedding, visual_embedding])
            
            emb = self.activation(emb)
        
            output = self.linear_continuous(emb)
            
            discrete_output = self.linear_categorical(emb)
        
        # return a dictionary
        out = {
            self.output_names[0]: output,
            "discretized": discrete_output
            }
        
        return out
    
    
    def get_num_outputs(self) -> int:
        """
        Returns the number of outputs in the last model layer for valence-arousal prediction.
        """
        return self.num_outputs
    
    def get_num_categorical_outputs(self) -> int:
        """
        Return the number of categories the valence arousal space is discretized into
        """
        return 24
    
    def freeze_extractors(self):
        """
        Sets requires_grad of all parameters of the feature extractors to False.
        Only applies if the feature networks are pretrained.
        """
        
        if self.visual_pretrained:
            print("Freezing visual feature extractor.")
            self.visual_network.requires_grad_(False) 
        
        if self.audio_pretrained:
            print("Freezing audio feature extractor.")
            self.audio_network.requires_grad_(False)
            
            
    def unfreeze_extractors(self):
        """
        Sets requires_grad of all parameters of the feature extractors to True.
        """
        
        self.audio_network.requires_grad_(True)
        self.visual_network.requires_grad_(True)
        
        
######
class AudioVisualAttentionNetwork(AudioVisualNetwork):
    """
    Subclass for attention-based models
    """
    
    def __init__(self, audio_input_size: int, visual_input_size: int, audio_backbone: str, visual_backbone: str, 
                 temporal_model: str, visual_pretrained=True, audio_pretrained=False, audio_pool=True, 
                 batch_first=True, 
                 d_embedding=128, d_feedforward=256, num_layers=4, num_crossaudio_layers=4, num_crossvisual_layers=4, num_heads=4, 
                 num_bins: int = 1, num_outs: int = 2, dropout=0.3, activation="gelu", output_names=["VA"]) -> None:
        super().__init__(audio_input_size, visual_input_size, audio_backbone, visual_backbone, 
                         temporal_model, visual_pretrained, audio_pretrained, audio_pool, 
                         batch_first, d_embedding, num_layers, 
                         num_bins, num_outs, dropout, activation, output_names)
        # additional arguments
        self.num_crossaudio_layers = num_crossaudio_layers
        self.num_crossvisual_layers = num_crossvisual_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        
        self.seq2seq_model, num_out_features = self._get_seq2seq_model()
        
        self.linear_continuous = nn.Linear(num_out_features, self.get_num_outputs())
        self.linear_categorical = nn.Linear(num_out_features, self.get_num_categorical_outputs())
         
         
    def _get_seq2seq_model(self) -> tuple[nn.Module, int]:
        """
        Provides an attention model
        """
        
        if self.temporal_model == "selfattn":
            
            print("Creating the concatenation + self attention model block")
            
            # initialise the fusion layer - is just concat for now
            fusion_layer = get_fusion_layer("concat", num_feats_modality=[self.num_audio_features, self.num_visual_features])
            # get the number of features that will be put into the seq2seq model
            seq2seq_input_features = fusion_layer.num_features
            
            assert sum([self.audio_network.model.num_features, self.visual_network.model.num_features]) == fusion_layer.num_features
            
            model = SelfAttentionModel(
                n_inputs=seq2seq_input_features,
                n_heads=self.num_heads,
                n_layers=self.num_layers,
                d_embedding=self.d_embedding,
                d_feedforward=self.d_feedforward,
                dropout=self.dropout,
                activation=self.activation,
                batch_first=self.batch_first
            )
            
            stack =  nn.Sequential(
                fusion_layer,
                model
            )
            
            return stack, model.num_features
        
        elif self.temporal_model == "mult":
            
            print("Creating the MulT architecture")
            
            model = MulT(
                input_size_audio=self.audio_network.model.num_features,
                input_size_visual=self.visual_network.model.num_features,
                n_heads=self.num_heads,
                dropout=self.dropout,
                n_layers=self.num_layers,
                n_ca_layers=self.num_crossaudio_layers,
                n_cv_layers=self.num_crossvisual_layers,
                d_embedding=self.d_embedding,
                d_ff=self.d_feedforward,
                batch_first=self.batch_first,
                activation=self.activation
            )
            
        elif self.temporal_model == "multv2":
            
            print("Creating the MulT architecture V2")
            
            model = MulTV2(
                input_size_audio=self.audio_network.model.num_features,
                input_size_visual=self.visual_network.model.num_features,
                n_heads=self.num_heads,
                dropout=self.dropout,
                n_layers=self.num_layers,
                n_ca_layers=self.num_crossaudio_layers,
                n_cv_layers=self.num_crossvisual_layers,
                d_embedding=self.d_embedding,
                d_ff=self.d_feedforward,
                batch_first=self.batch_first,
                activation=self.activation
            )
        
        elif self.temporal_model == "gatedmultv2":
            
            print("Creating the  Gated MulT architecture V2")
            
            model = GatedMulTV2(
                input_size_audio=self.audio_network.model.num_features,
                input_size_visual=self.visual_network.model.num_features,
                n_heads=self.num_heads,
                dropout=self.dropout,
                n_layers=self.num_layers,
                n_ca_layers=self.num_crossaudio_layers,
                n_cv_layers=self.num_crossvisual_layers,
                d_embedding=self.d_embedding,
                d_ff=self.d_feedforward,
                batch_first=self.batch_first,
                activation=self.activation
            )
        
        else: 
            raise NotImplementedError("model {} not implemented".format(self.temporal_model))
            
        return model, model.num_features
    
    #def forward():
    #    pass


class AudioVisualRecurrentModel(AudioVisualNetwork):
    """
    Subclass for recurrent models
    """
    
    def __init__(self, audio_input_size: int, visual_input_size: int, audio_backbone: str, visual_backbone: str, 
                 temporal_model: str, visual_pretrained=True, audio_pretrained=False, audio_pool=True, 
                 batch_first=True, d_embedding=128, num_layers=4, 
                 num_bins: int = 1, num_outs: int = 2, dropout=0.3, activation="gelu", output_names=["VA"], bidirectional=False) -> None:
        super().__init__(audio_input_size, visual_input_size, audio_backbone, visual_backbone, 
                         temporal_model, visual_pretrained, audio_pretrained, audio_pool, 
                         batch_first, d_embedding, num_layers, num_bins, num_outs, dropout, activation, output_names)
        
        self.bidirectional=bidirectional
        
        self.seq2seq_model, num_out_features = self._get_seq2seq_model()
        
        self.linear_continuous = nn.Linear(num_out_features, self.get_num_outputs())
        self.linear_categorical = nn.Linear(num_out_features, self.get_num_categorical_outputs())
        
    def _get_seq2seq_model(self) -> tuple[nn.Module, int]:
        """
        Provides a recurrent model
        """
        
        if self.temporal_model == "rnn":
            
            print("Creating the concatenation + rnn model block")
            
            # initialise the fusion layer - is just concat for now
            fusion_layer = get_fusion_layer("concat", num_feats_modality=[self.num_audio_features, self.num_visual_features])
            # get the number of features that will be put into the seq2seq model
            seq2seq_input_features = fusion_layer.num_features
            
            assert sum([self.audio_network.model.num_features, self.visual_network.model.num_features]) == seq2seq_input_features
            
            # a linear layer to avoid high-dim rnn model
            linear = nn.Linear(seq2seq_input_features, self.d_embedding)
            
            if isinstance(self.activation, str):
                act =  get_activation(self.activation)
            else:
                act = self.activation
            
            model = RNNModel(
                num_layers=self.num_layers,
                d_input=self.d_embedding,
                d_hidden=self.d_embedding,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                batch_first=True)
            
            return nn.Sequential(fusion_layer, 
                                 linear,
                                 act,
                                 model), model.num_features
            
        if self.temporal_model == "birnn":
            
            print("Creating the concatenation + bidirectional rnn model block")
            
            # initialise the fusion layer - is just concat for now
            fusion_layer = get_fusion_layer("concat", num_feats_modality=[self.num_audio_features, self.num_visual_features])
            # get the number of features that will be put into the seq2seq model
            seq2seq_input_features = fusion_layer.num_features
            
            assert sum([self.audio_network.model.num_features, self.visual_network.model.num_features]) == fusion_layer.num_features
            
             # a linear layer to avoid high-dim rnn model
            linear = nn.Linear(seq2seq_input_features, self.d_embedding)
            
            if isinstance(self.activation, str):
                act =  get_activation(self.activation)
            else:
                act = self.activation
            
            model = RNNModel(
                num_layers=self.num_layers,
                d_input=self.d_embedding,
                d_hidden=self.d_embedding,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                batch_first=True)
            
            return nn.Sequential(fusion_layer,
                                 linear, 
                                 act, 
                                 model), model.num_features
            
        
    #def forward(self):
    #    pass
    
class AudioVisualLinearNetwork(AudioVisualNetwork):
    
    def __init__(self, audio_input_size: int, visual_input_size: int, audio_backbone: str, visual_backbone: str, temporal_model: str, visual_pretrained=True, audio_pretrained=False, audio_pool=True, batch_first=True, d_embedding=128, num_layers=4, num_bins: int = 1, num_outs: int = 2, dropout=0.3, activation="gelu", output_names=["VA"]) -> None:
        super().__init__(audio_input_size, visual_input_size, audio_backbone, visual_backbone, temporal_model, visual_pretrained, audio_pretrained, audio_pool, batch_first, d_embedding, num_layers, num_bins, num_outs, dropout, activation, output_names)
        
        self.seq2seq_model, num_out_features = self._get_seq2seq_model()
        
        self.linear_continuous = nn.Linear(num_out_features, self.get_num_outputs())
        self.linear_categorical = nn.Linear(num_out_features, self.get_num_categorical_outputs())
        
    # TODO fill this
    def _get_seq2seq_model(self) -> tuple[nn.Module, int]:
        """
        Provides a simple linear model
        """
        return None, 0
    
        
        