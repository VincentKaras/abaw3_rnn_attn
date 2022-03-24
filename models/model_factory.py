import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from end2you.utils import Params

import pytorch_model_summary

from models.modules.audiovisual import AudioVisualAttentionNetwork, AudioVisualRecurrentModel, AudioVisualLinearNetwork
from models.modules.audio_model import AudioAttentionNetwork, AudioRecurrentNetwork
from models.modules.visual_model import VisualAttentionNetwork, VisualRecurrentNetwork
from models.modules.hybrid import AuxAttentionNetwork, AuxRecurrentNetwork


"""
Provides the definitions of supported models. Might add more / custom model support later

Vincent Karas, 01/2022
"""

class ModelFactory():
    """
    A helper class which returns the desired model contained inside a wrapper
    """
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(args:Params):
        """
        Static Factory method which creates the requested model
        :param args: The params object provided by the parser (top level)
        :return: a model wrapper object which contains a PyTorch model and some extra helpful stuff
        """
        
        
        #if isinstance(args.task, list):
        #    output_names = args.task
        #    tasks = args.task
        #else:
        #    output_names = [args.task]
        #    tasks = [args.task]
        output_names = ["VA"]
        tasks = ["VA"]
            
        # get the input sizes - this is process dependent due to nested params objects
        if args.process == "train":
            audio_input_size=int(args.train.audio_sr * args.train.audio_window_size)
            visual_input_size = args.train.image_size
            seq_length = args.train.seq_length
        else:
            audio_input_size=int(args.test.audio_sr * args.test.audio_window_size)
            visual_input_size = args.test.image_size
            seq_length = args.test.seq_length
            

        # create a model
        if args.model.model_name == "concatselfattn":
            model = AudioVisualAttentionNetwork(audio_input_size=audio_input_size,
                                       visual_input_size=visual_input_size,
                                       audio_backbone=args.model.audio_backbone,
                                       visual_backbone=args.model.visual_backbone,
                                       temporal_model="selfattn",
                                       batch_first=True,
                                       visual_pretrained=args.model.visual_pretrained,
                                       audio_pretrained=args.model.audio_pretrained,
                                       audio_pool=True,
                                       num_bins = args.model.num_bins,
                                       num_outs = args.model.num_outs,
                                       output_names=output_names,
                                       dropout=args.model.dropout,
                                       num_layers=args.model.num_layers,
                                       num_heads=args.model.num_heads,
                                       d_embedding=args.model.d_embedding,
                                       d_feedforward=args.model.d_feedforward)
        elif args.model.model_name == "mult":   # multimodal transformer
            model = AudioVisualAttentionNetwork(audio_input_size=audio_input_size,
                                       visual_input_size=visual_input_size,
                                       audio_backbone=args.model.audio_backbone,
                                       visual_backbone=args.model.visual_backbone, 
                                       visual_pretrained=args.model.visual_pretrained,
                                       audio_pretrained=args.model.audio_pretrained,
                                       audio_pool=True,
                                       temporal_model="mult",
                                       batch_first=True,
                                       num_bins = args.model.num_bins,
                                       num_outs = args.model.num_outs,
                                       output_names=output_names,
                                       dropout=args.model.dropout,
                                       num_layers=args.model.num_layers,
                                       num_crossaudio_layers=args.model.num_crossaudio_layers,
                                       num_crossvisual_layers=args.model.num_crossvisual_layers,
                                       num_heads=args.model.num_heads,
                                       d_embedding=args.model.d_embedding,
                                       d_feedforward=args.model.d_feedforward)
        
        elif args.model.model_name == "multv2":   # multimodal transformer
            model = AudioVisualAttentionNetwork(audio_input_size=audio_input_size,
                                       visual_input_size=visual_input_size,
                                       audio_backbone=args.model.audio_backbone,
                                       visual_backbone=args.model.visual_backbone, 
                                       visual_pretrained=args.model.visual_pretrained,
                                       audio_pretrained=args.model.audio_pretrained,
                                       audio_pool=True,
                                       temporal_model="multv2",
                                       batch_first=True,
                                       num_bins = args.model.num_bins,
                                       num_outs = args.model.num_outs,
                                       output_names=output_names,
                                       dropout=args.model.dropout,
                                       num_layers=args.model.num_layers,
                                       num_crossaudio_layers=args.model.num_crossaudio_layers,
                                       num_crossvisual_layers=args.model.num_crossvisual_layers,
                                       num_heads=args.model.num_heads,
                                       d_embedding=args.model.d_embedding,
                                       d_feedforward=args.model.d_feedforward)
            
        elif args.model.model_name == "gatedmultv2":   # multimodal transformer with gated cma
            model = AudioVisualAttentionNetwork(audio_input_size=audio_input_size,
                                       visual_input_size=visual_input_size,
                                       audio_backbone=args.model.audio_backbone,
                                       visual_backbone=args.model.visual_backbone, 
                                       visual_pretrained=args.model.visual_pretrained,
                                       audio_pretrained=args.model.audio_pretrained,
                                       audio_pool=True,
                                       temporal_model="gatedmultv2",
                                       batch_first=True,
                                       num_bins = args.model.num_bins,
                                       num_outs = args.model.num_outs,
                                       output_names=output_names,
                                       dropout=args.model.dropout,
                                       num_layers=args.model.num_layers,
                                       num_crossaudio_layers=args.model.num_crossaudio_layers,
                                       num_crossvisual_layers=args.model.num_crossvisual_layers,
                                       num_heads=args.model.num_heads,
                                       d_embedding=args.model.d_embedding,
                                       d_feedforward=args.model.d_feedforward)
        
        
        elif args.model.model_name == "concatrnn":
            model = AudioVisualRecurrentModel(audio_input_size=audio_input_size,
                                       visual_input_size=visual_input_size,
                                       audio_backbone=args.model.audio_backbone,
                                       visual_backbone=args.model.visual_backbone,
                                       temporal_model="rnn",
                                       batch_first=True,
                                       visual_pretrained=args.model.visual_pretrained,
                                       audio_pretrained=args.model.audio_pretrained,
                                       audio_pool=True,
                                       num_bins = args.model.num_bins,
                                       num_outs = args.model.num_outs,
                                       output_names=output_names,
                                       dropout=args.model.dropout,
                                       num_layers=args.model.num_layers,
                                       d_embedding=args.model.d_embedding)
            
        elif args.model.model_name == "concatbirnn":
            model = AudioVisualRecurrentModel(audio_input_size=audio_input_size,
                                       visual_input_size=visual_input_size,
                                       audio_backbone=args.model.audio_backbone,
                                       visual_backbone=args.model.visual_backbone,
                                       temporal_model="birnn",
                                       batch_first=True,
                                       visual_pretrained=args.model.visual_pretrained,
                                       audio_pretrained=args.model.audio_pretrained,
                                       audio_pool=True,
                                       num_bins = args.model.num_bins,
                                       num_outs = args.model.num_outs,
                                       output_names=output_names,
                                       dropout=args.model.dropout,
                                       num_layers=args.model.num_layers,
                                       d_embedding=args.model.d_embedding,
                                       bidirectional=args.model.bidirectional,)
            
        elif args.model.model_name == "concatlinear":
            model = AudioVisualLinearNetwork(audio_input_size=audio_input_size,
                                       visual_input_size=visual_input_size,
                                       audio_backbone=args.model.audio_backbone,
                                       visual_backbone=args.model.visual_backbone,
                                       temporal_model="linear",
                                       batch_first=True,
                                       visual_pretrained=args.model.visual_pretrained,
                                       audio_pretrained=args.model.audio_pretrained,
                                       audio_pool=True,
                                       num_bins = args.model.num_bins,
                                       num_outs = args.model.num_outs,
                                       output_names=output_names,
                                       dropout=args.model.dropout,
                                       num_layers=args.model.num_layers,
                                       d_embedding=args.model.d_embedding)
            
        elif args.model.model_name == "audioselfattn":
            model = AudioAttentionNetwork(audio_input_size=audio_input_size,
                                          audio_encoder=args.model.audio_backbone,
                                          s2s_model="selfattn",
                                          batch_first=True,
                                          pretrained=args.model.audio_pretrained,
                                          output_names=output_names,
                                          d_embedding=args.model.d_embedding,
                                          d_ff=args.model.d_feedforward,
                                          n_layers=args.model.num_layers,
                                          n_heads=args.model.num_heads)
        elif args.model.model_name == "audiornn":
            model = AudioRecurrentNetwork(audio_input_size=audio_input_size,
                                          audio_encoder=args.model.audio_backbone,
                                          s2s_model="rnn",
                                          batch_first=True,
                                          pretrained=args.model.audio_pretrained,
                                          output_names=output_names,
                                          d_embedding=args.model.d_embedding,
                                       dropout=args.model.dropout,
                                       activation=args.model.activation,
                                       n_layers=args.model.num_layers,
                                       bidirectional=args.model.bidirectional,
                                       )
            
        
        elif args.model.model_name == "visualselfattn":
            model = VisualAttentionNetwork(visual_input_size=visual_input_size,
                                           visual_encoder=args.model.visual_backbone,
                                           s2s_model="selfattn",
                                           batch_first=True,
                                           pretrained=args.model.visual_pretrained,
                                           output_names=output_names,
                                           d_embedding=args.model.d_embedding,
                                           d_ff=args.model.d_feedforward,
                                           n_layers=args.model.num_layers,
                                           n_heads=args.model.num_heads)
            
        elif args.model.model_name == "visualrnn":
            model = VisualRecurrentNetwork(visual_input_size=visual_input_size,
                                           visual_encoder=args.model.visual_backbone,
                                           s2s_model="rnn",
                                           batch_first=True,
                                           pretrained=args.model.visual_pretrained,
                                           output_names=output_names,
                                           d_embedding=args.model.d_embedding,
                                            dropout=args.model.dropout,
                                        activation=args.model.activation,
                                       n_layers=args.model.num_layers,
                                       bidirectional=args.model.bidirectional,)
            
        # aux
            
        elif args.model.model_name == "auxattn":
            model = AuxAttentionNetwork(audio_input_size=audio_input_size,
                                        audio_backbone=args.model.audio_backbone,
                                        audio_pool=True,
                                        audio_pretrained=args.model.audio_pretrained,
                                        visual_input_size=visual_input_size,
                                        visual_backbone=args.model.visual_backbone,
                                        visual_pretrained=args.model.visual_pretrained,
                                        temporal_model="selfattn",
                                        num_bins=args.model.num_bins,
                                        num_outs=args.model.num_outs,
                                        d_embedding=args.model.d_embedding,
                                        d_feedforward=args.model.d_feedforward,
                                        num_heads=args.model.num_heads,
                                        num_layers=args.model.num_layers,
                                        num_layers_audio=args.model.num_layers_audio,
                                        num_layers_visual=args.model.num_layers_visual,
                                        num_layers_ca=args.model.num_crossaudio_layers,
                                        num_layers_cv=args.model.num_crossvisual_layers,
                                        dropout=args.model.dropout,
                                        activation=args.model.activation,
                                        )
        elif args.model.model_name == "auxrnn":
            model = AuxRecurrentNetwork(audio_input_size=audio_input_size,
                                        audio_backbone=args.model.audio_backbone,
                                        audio_pool=True,
                                        audio_pretrained=args.model.audio_pretrained,
                                        visual_input_size=visual_input_size,
                                        visual_backbone=args.model.visual_backbone,
                                        visual_pretrained=args.model.visual_pretrained,
                                        temporal_model="rnn",
                                        d_embedding=args.model.d_embedding, 
                                       dropout=args.model.dropout,
                                       activation=args.model.activation,
                                       num_layers=args.model.num_layers,
                                        num_layers_audio=args.model.num_layers_audio,
                                        num_layers_visual=args.model.num_layers_visual,
                                       bidirectional=args.model.bidirectional,)    
        
            
        elif args.model.model_name == "hybrid":
            
            # TODO
            raise NotImplementedError
            
        
        else:
            raise NotImplementedError("Model {} not implemented yet".format(args.model.model_name))


        # summarise model
        pytorch_model_summary.summary(model, 
                                      [torch.zeros(1, seq_length, 1, audio_input_size),
                                      torch.zeros(1, seq_length, 3, visual_input_size, visual_input_size)], 
                                      #show_hierarchical=True, 
                                      show_input=False,
                                      print_summary=True)
        

        # move model to GPU
        if args.cuda:
            print("*" * 5)
            print("Moving model to GPU ...")
            model = model.cuda()

        if args.process == "train":     # add the train args into the wrapper
            return ModelWrapper(model=model, 
                                name=args.name,
                                ckpt_dir=args.checkpoints_dir,
                                lr=args.train.lr,
                                lr_policy=args.train.lr_policy,
                                opt=args.train.optimizer,
                                weight_decay=args.train.weight_decay,
                                is_train=args.train.is_training,
                                cuda = args.cuda,
                                num_bins=args.model.num_bins,
                                num_outputs=args.model.num_outs,
                                output_names=output_names,
                                tasks=tasks)
        else:   #test
            return ModelWrapper(model=model, 
                            name=args.name,
                            ckpt_dir=args.checkpoints_dir,
                            is_train=args.test.is_training,
                            cuda = args.cuda,
                            num_bins=args.model.num_bins,
                            num_outputs=args.model.num_outs,
                            output_names=output_names,
                            tasks=tasks)



class ModelWrapper():
    """
    A helper class which wraps the model and handles things formatting data for forward pass, formatting the outputs, saving, loading, ...
    """
    def __init__(self, model:nn.Module, name:str, ckpt_dir, is_train:bool, cuda:bool, 
                 num_outputs:int, num_bins=1, tasks=["VA"], output_names=["VA"], lr=0.001, lr_policy="step", opt="adam", weight_decay=0.0, *args, **kwargs):
        
        self._model = model
        self._name = name
        self._tasks = tasks
        self.ckpt_dir = ckpt_dir
        self.lr = lr
        self.lr_policy = lr_policy
        self._opt = opt
        self._weight_decay = weight_decay
        self.is_train = is_train
        self.cuda = cuda
        self.num_bins = num_bins
        self.num_outputs = num_outputs
        self.output_names = output_names


    def set_train(self):
        self.is_train = True
        self._model.train()


    def set_eval(self):
        self.is_train = False
        self._model.eval()
        
        
    def freeze_extractors(self):
        """
        Freezes the feature extraction networks of the model
        """
        self._model.freeze_extractors()
    
    
    def unfreeze_extractors(self):
        """
        Unfreezes the feature extraction networks of the model
        """
        self._model.unfreeze_extractors()
        

    def get_model(self):
        """
        Returns the internal model object
        """
        return self._model
    
    
    def model_to_cuda(self):
        if self.cuda:
            print("Moving model to GPU ...")
            self._model = self._model.cuda()
            

    def forward(self, input, return_estimates=False):
        """
        Helper function - just passes the input to the internal model for now and returns the results.
        If format_estimates is set, it will reduce the predictions in the last dimension and return an additional dict
        """
        prediction_dict = {}
        
        if isinstance(input, dict):
            input = [input["audio"], input["image"]]
        
        predictions = self._model(input)
        
        
        #if len(self.output_names) == 1:
        #    prediction_dict[self.output_names[0]] = predictions
        #else:
        #    for i, on in enumerate(self.output_names):
        #        prediction_dict[on] = predictions[i]
            
        if return_estimates:
            estimates = self._get_estimates(output=predictions)
            return predictions, estimates
        
        else: 
            return predictions
    
    def _get_estimates(self, output:dict):
        
        estimates = {}
        
        for task in output.keys():
            if task == "AU":
                o = (torch.sigmoid(output["AU"].cpu()) > 0.5).type(torch.LongTensor)
                estimates["AU"] = o.numpy()
            elif task == "EXPR":
                o = torch.softmax(output["EXPR"].cpu(), dim=-1).argmax(-1).type(torch.LongTensor)
                estimates["EXPR"] = o.numpy()
            elif task == "VA":
                if self.num_bins > 1:   # only necessary if number of bins is greater than 1
                    v = torch.softmax(output["VA"][:,:,:self.num_bins].cpu(), dim=-1).numpy()
                    a = torch.softmax(output["VA"][:,:,self.num_bins:].cpu(), dim=-1).numpy()
                    bins = np.linspace(-1, 1, num=self.num_bins)
                    v = (bins * v).sum(-1)  # [BS, T]
                    a = (bins * a).sum(-1)  # [BS, T]
                    estimates["VA"] = np.stack([v, a], axis=-1) #[BS, T, 2]
                else:
                    estimates["VA"] = output["VA"].detach().cpu().numpy()
        
        return estimates
    
    def save(self, label):
        """
        saves the network
        """
        save_filename = "net_epoch_{}_{}".format(label, self.name)
        save_path = Path(self.ckpt_dir / save_filename)
        save_dict = {"state_dict": self._model.state_dict(), "epoch": label}
        
        torch.save(save_dict, save_path)
        print("Saved model to {}".format(save_path))
        





