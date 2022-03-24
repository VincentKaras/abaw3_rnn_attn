"""
Get Dataloaders for the Affwild Dataset

"""


from torch.utils.data import DataLoader, Subset, Dataset

from end2you.utils import Params

from data.dataset_VA import VA_Dataset
from data.dataset_AU import AU_Dataset
from data.dataset_EXPR import EXPR_Dataset

from data.test_dataset import VA_TestDataset

from utils.transforms import get_audio_transforms, train_transforms, test_transforms, audio_train_transforms


def pad_collate(batch):
    """
    Helper function which pads all tensors to the same sequence length.
    Should not be required since all tensors are created with matching lengths and checked with assert statements.
    """


def get_dataloader(params:Params, train_mode:str=None, task:str="VA", persist=False, tiny=False, visual_aug=True, audio_aug=False) -> DataLoader:
    """
    Factory Method which returns a Dataloader.
    At the moment, only capable of returning a VA dataset loader. Might extend later.
    :param params A Params object containing the configuration for the dataset. This should be properly configured to contain all necessary args 
    :param train_mode A string of either ["Train", "Validation", "Test"]. 
    :param task A string selecting the annotation type to load. Should always be VA.
    :param tiny Bool If True, return a loader on a subset of the data that contains a single batch. Default False.
    Should not be necessary if specific file is given for each partition? But is needed atm because of dataset class structure
    """

    #if train_mode is "train":
    
    print("Setting up {} transforms".format(train_mode.lower()))
    if (train_mode == "Train"):
        if visual_aug:
            print("Image transforms with augmentation...")
            image_transforms = train_transforms(params.image_size)
        else:
            print("Image augmentation disabled!")
            image_transforms = test_transforms(params.image_size)
        
        if audio_aug:    
            print("Audio transforms with augmentation ...")
            audio_transforms = audio_train_transforms()
        else:
            print("audio transforms disabled!")
            audio_transforms = None
        
        # here we can use stride to dilate sequences
        fps = params.fps
        
    else:
        print("Only converting image to Tensor and Resizing")
        image_transforms = test_transforms(params.image_size)
        print("No transforms applied to audio")
        audio_transforms = None
        
        # lock fps since we dont want to stride on the validation / test set
        fps = 30

    if "VA" in task:
        
        dataset=VA_Dataset(dataset_file=params.dataset_file, 
                               seq_len=params.seq_length, 
                               fps=fps, 
                               image_size=params.image_size,
                               audio_sr=params.audio_sr, 
                               audio_window_size=params.audio_window_size,
                               train_mode=train_mode,
                               channel_order=params.image_channel_order,
                               transforms=image_transforms,
                               audio_transforms=audio_transforms,
                               tiny=tiny)
        
        if tiny:
            print("Creating a subset of the dataset that can hold a single batch")
            dataset = Subset(dataset, range(0, params.batch_size))
            shuffle = False # disable shuffle so we always get the same batch
        else:
            shuffle = params.is_training    # True for train, False for validation
            print("Shuffling data" if params.is_training else "Not shuffling data") 

        return DataLoader(
            dataset=dataset, 
            batch_size=params.batch_size, 
            shuffle=shuffle,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
            persistent_workers=persist
            )

    elif "EXPR" in task:

        return DataLoader(
            dataset=EXPR_Dataset(dataset_file=params.dataset_file, 
                                 seq_len=params.seq_length, 
                                 fps=fps, 
                                 audio_sr=params.audio_sr, 
                                 audio_window_size=params.audio_window_size,
                                 train_mode=train_mode, 
                                 channel_order=params.image_channel_order,
                                 transforms=image_transforms,
                                 tiny=tiny), 
            batch_size=params.batch_size, 
            shuffle=params.is_training,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,)
    
    elif "AU" in task:

        return DataLoader(
            dataset=AU_Dataset(dataset_file=params.dataset_file, 
                               seq_len=params.seq_length, 
                               fps=fps, 
                               audio_sr=params.audio_sr, 
                               audio_window_size=params.audio_window_size,
                               train_mode=train_mode,
                               channel_order=params.image_channel_order,
                               transforms=image_transforms,
                               tiny=tiny
                ), 
            batch_size=params.batch_size, 
            shuffle=params.is_training,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,)

    else:
        raise NotImplementedError("Task not implemented")
    

def get_test_dataloader(params:Params, train_mode:str=None, task:str="VA", persist=False, video=None, visual_aug=True, audio_aug=False) -> DataLoader:
    """
    Creates a dataloader for the test set
    """
    
    print("Setting up transforms ...")
    
    print("Only converting image to Tensor and Resizing")
    image_transforms = test_transforms(params.image_size)
    print("No transforms applied to audio")
    audio_transforms = None
        
     # lock fps since we dont want to stride on the validation / test set
    fps = 30

    if "VA" in task:
        dataset = VA_TestDataset(dataset_file=params.dataset_file, 
                               seq_len=params.seq_length, 
                               fps=fps, 
                               image_size=params.image_size,
                               audio_sr=params.audio_sr, 
                               audio_window_size=params.audio_window_size,
                               train_mode=train_mode,
                               channel_order=params.image_channel_order,
                               transforms=image_transforms,
                               audio_transforms=audio_transforms,
                               video=video)
        
        return DataLoader(dataset, 
                           batch_size=1, 
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
            drop_last=False)
        
    else:
        
        print("Not implemented")
        return None
    


def get_dataset(params:Params, train_mode:str=None, task:str="VA", visual_aug=True, audio_aug=False):
    """
    Returns a dataset object
    """
    
    print("Setting up {} transforms".format(train_mode.lower()))
    if (train_mode == "Train"):
        if visual_aug:
            print("Image transforms with augmentation...")
            image_transforms = train_transforms(params.image_size)
        else:
            print("Image augmentation disabled!")
            image_transforms = test_transforms(params.image_size)
        
        if audio_aug:    
            print("Audio transforms with augmentation ...")
            audio_transforms = audio_train_transforms()
        else:
            print("audio transforms disabled!")
            audio_transforms = None
        
        # here we can use stride to dilate sequences
        fps = params.fps
        
    else:
        print("Only converting image to Tensor and Resizing")
        image_transforms = test_transforms(params.image_size)
        print("No transforms applied to audio")
        audio_transforms = None
        
        # lock fps since we dont want to stride on the validation / test set
        fps = 30

    if "VA" in task:
        
        dataset=VA_Dataset(dataset_file=params.dataset_file, 
                               seq_len=params.seq_length, 
                               fps=fps, 
                               image_size=params.image_size,
                               audio_sr=params.audio_sr, 
                               audio_window_size=params.audio_window_size,
                               train_mode=train_mode,
                               channel_order=params.image_channel_order,
                               transforms=image_transforms,
                               audio_transforms=audio_transforms
                            )
            
        return dataset

    elif "EXPR" in task:

        
        return EXPR_Dataset(dataset_file=params.dataset_file, 
                                 seq_len=params.seq_length, 
                                 fps=fps, 
                                 audio_sr=params.audio_sr, 
                                 audio_window_size=params.audio_window_size,
                                 train_mode=train_mode, 
                                 channel_order=params.image_channel_order,
                                 transforms=image_transforms,
                                )
    
    elif "AU" in task:

        return AU_Dataset(dataset_file=params.dataset_file, 
                               seq_len=params.seq_length, 
                               fps=fps, 
                               audio_sr=params.audio_sr, 
                               audio_window_size=params.audio_window_size,
                               train_mode=train_mode,
                               channel_order=params.image_channel_order,
                               transforms=image_transforms,
                      
                )

    else:
        raise NotImplementedError("Task not implemented")
    