from torchvision.transforms import Resize, ColorJitter, RandomHorizontalFlip, ToTensor, Normalize, Compose
from torchaudio.transforms import AmplitudeToDB
import torch
import torch.nn.functional as F
import math
import numpy as np
"""
Helper file with transforms for audio and images
"""


def get_audio_transforms():

    #return Compose([
    #    AmplitudeToDB("power", 80),
    #    Normalize(mean=[-14.8],
    #              std=[19.895])
    #])
    
    return AmplitudeToDB("power", 80)


def train_transforms(img_shape, mean=None, std=None):

    cj = 0.2    # color jitter 

    transforms = [
        ToTensor(), # now done in getitem 
        Resize(img_shape),
        ColorJitter(brightness=cj, contrast=cj, saturation=cj),
        RandomHorizontalFlip(),
       
    ]
    
    if (mean is not None) and (std is not None):
        transforms.append(Normalize(mean=mean, std=std))

    return Compose(transforms)


def audio_train_transforms(p_swap=0.1, p_flip=0.1, snr=100):
    """
    Sets parameters for audio transforms. Transforms are done in getitem directly
    """
    
    return {
        "swap": p_swap,
        "snr": snr,
        "flip": p_flip,
    }


def test_transforms(img_shape, mean=None, std=None):

    transforms = [
        ToTensor(),    # now  done in getitem
        Resize(img_shape),
    ]

    if (mean is not None) and (std is not None):
        transforms.append(Normalize(mean=mean, std=std))

    return Compose(transforms=transforms)
    
    
    
    
def get_audiovisual_transforms(img_shape, train=True):
    
    if train:
        
        visual_transform = train_transforms(img_shape=img_shape)
    else:
        visual_transform = test_transforms(img_shape)
        
    return {
        "audio": get_audio_transforms(),
        "visual": visual_transform
    }
    


def discretize_labels(labels:torch.Tensor):
    """
    turns the valence -arousal labels into categories matching slices of the circumplex
    Args: labels Tensor of shape [BS, T, 2]
    Returns: cats Tensor of shape [BS, T, ]
    """
    #device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    
    num_categories = 24
    pi = 3.1416 # about Pi
    
    angle_boundaries = torch.linspace(-pi, pi, 9, device=labels.get_device() if labels.is_cuda else "cpu")
    #angle_boundaries = angle_boundaries.cuda(labels.get_device()) if labels.is_cuda else angle_boundaries   # create 8 equal segments (45 deg)
    radial_boundaries = torch.linspace(0, math.sqrt(2), 4, device=labels.get_device() if labels.is_cuda else "cpu")
    #radial_boundaries =  radial_boundaries.cuda(labels.get_device()) if labels.is_cuda else radial_boundaries
    
    valence = labels[..., 0]
    arousal = labels[..., 1]
    # compute radial component
    #radius = torch.linalg.norm(labels, dim=-1)
    radius = torch.sqrt(valence.pow(2) + arousal.pow(2))
    angle = torch.atan2(arousal, valence)   # valence is x coordinate, arousal is y coordinate
    
    # bucketize
    qradius = torch.bucketize(radius, radial_boundaries) - 1    # subtract 1 because we want to start at 0
    qangle = torch.bucketize(angle, angle_boundaries) - 1
    
    # combine 
    cats = qangle + qradius * 8 # should give [0, 23]
    
    # replace any entries > 
    cats = torch.clamp(cats, 0, num_categories -1)
    
    # one-hot encoding
    #hot = F.one_hot(cats, num_classes=num_categories)
    
    return cats


def discretize_labels_np(data: np.ndarray):
    
    num_categories = 24
    pi = 3.1416
    
    angle_boundaries = np.linspace(-pi, pi, 9)
    radial_boundaries = np.linspace(0, math.sqrt(2), 4)
    
    valence = data[..., 0]
    arousal = data[..., 1]
    
    radius = np.sqrt(valence ** 2 + arousal ** 2)
    angle = np.arctan2(arousal, valence)
    
    qradius = np.digitize(radius, radial_boundaries) - 1
    qangle = np.digitize(angle, angle_boundaries) - 1 
    
    cats = qangle + 8 * qradius
    
    return cats
    
    