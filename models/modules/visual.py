import torch
import torch.nn as nn
from torchvision import transforms

from vggface2_pytorch.models.senet import SENet, senet50
from vggface2_pytorch.models.resnet import ResNet, resnet50
from vggface2_pytorch import utils as vggface2_utils

from facenet_pytorch import InceptionResnetV1

from models.modules.mobilefacenet import MobileFaceNet

from models import MODEL_PATHS


class VisualConvEncoder(nn.Module):
    """
    Wrapper class for encoder acting on an image sequence
    """
    def __init__(self, input_size:int, name:str, pretrained=True, batch_first=True, channel_order="RGB") -> None:
        super(VisualConvEncoder, self).__init__()
        
        self.name = name
        self.input_size = input_size
        self.pretrained = pretrained
        self.batch_first = batch_first
        self.model, self.num_features = self._create_model()
        
        self.channel_order = channel_order
        
        oxford_vggface2_models = ["resnet50_ft", 
                      "resnet50_scratch",
                      "senet50_ft",
                      "senet50_scratch"]
        
        facenet_pytorch_models = ["inceptionresnetv1"]
        
        mobilefacenet_pytorch_models = ["mobilefacenet"]
        
        if pretrained:
            if self.name in oxford_vggface2_models:
                # bgr order
                self.normalize = transforms.Normalize(mean=[0.357, 0.406, 0.512], std=[1.0, 1.0, 1.0])
            
            elif self.name in facenet_pytorch_models:
                # facenet-pytorch works directly on PIL images / uint8 numpy arrays - applies (x - 127.5) / 128.0
                # based on fixed_image_standardization function, adapted to tensors [0, 1.0]
                self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                
            elif self.name in mobilefacenet_pytorch_models:
                self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])    
            
            else: # standard pytorch models normalisation    
                self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]) 
        
    def _create_model(self) -> tuple[nn.Module, int]:
        """
        Helper which creates a CNN, initializes it, and wraps it in a TimeDistributed module
        """
        
        feature_sizes = {
            "resnet50_ft": 2048,
            "resnet50_scratch": 2048,
            "senet50_ft": 2048,
            "senet50_scratch": 2048,
            "inceptionresnetv1": 512,    # facenet pytorch
            "mobilefacenet": 512
        }
        
        # VGGFace2 models
        self.names = ["resnet50_ft", 
                      "resnet50_scratch",
                      "senet50_ft",
                      "senet50_scratch",
                      "inceptionresnetv1",
                      "mobilefacenet"]
        
        if self.name not in self.names:
            raise ValueError("Model {} not available!".format(self.name))
        
        # setup the model architecture
        if (self.name == "resnet50_ft") | (self.name == "resnet50_scratch"):
            cnn = resnet50(num_classes=8631, include_top=False)
        elif (self.name == "senet50_ft") | (self.name == "senet50_scratch"): 
            cnn = senet50(num_classes=8631, include_top=False)
        elif self.name == "inceptionresnetv1":
            if self.pretrained:
                print("Loading pre-trained weights for facenet vggface2 model ...")
                cnn = InceptionResnetV1(pretrained="vggface2")
            else:
                cnn = InceptionResnetV1(pretrained=None)
                
        elif self.name == "mobilefacenet":
            cnn = mobile_facenet(pretrained=self.pretrained)
        else:
            cnn = None
        
        # load the pre-trained weights for the oxford vggface2 net
        if self.pretrained and self.name in MODEL_PATHS.vggface2_models:
            print("Loading pre-trained weights for oxford vggface2 model ...")
            weights_dir = MODEL_PATHS.pretrained_dir
            # VGGFace models
            weights_file = str(weights_dir / "VGGFace2" / (self.name + "_weight.pkl"))       
            # needs specific method 
            vggface2_utils.load_state_dict(cnn, weights_file)
                
            # other models
            
        # wrap with TimeDistributed 
        # wrapped_model = TimeDistributed(model=cnn, batch_first=self.batch_first)  
        
        # get number of features
        num_features = feature_sizes[self.name] 
        
        # return wrapped_model, num_features
        return cnn, num_features
        
       
    def forward(self, input):
        """
        Input: Batch of N H x W images of shape [N, 3, H, W]
        Output: [N, Cout]
        Does a pooling of the last output
        """
        
        # apply normalisation if the encoder is pre-trained. Inspired by End2You
        if self.pretrained and self.normalize is not None:
            input = self.normalize(input)
        
        visual_embedding = self.model(input) 
        
        if len(visual_embedding.size()) > 2:
            visual_embedding = torch.mean(visual_embedding, dim=[2, 3]) # assuming [N, Cout, 1, 1] due to lack of global avg pooling
          
        return visual_embedding
    
    
def mobile_facenet(pretrained=True):
    
    model = MobileFaceNet([112, 112], 136)
    
    if pretrained:
        checkpoint = torch.load(str(MODEL_PATHS.mobilefacenet_pretrained))
        model.load_state_dict(checkpoint["state_dict"])
        
    # remove the output layer from the MobileFaceNet
    model.remove_output_layer()
    
    return model