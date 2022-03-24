from pathlib import Path

import torch
from PIL import Image
import cv2

import numpy as np
import pandas as pd

from data import PATHS
from data.dataset import BaseDataset, original_name

import torchaudio
import torchvision

#import random

"""
Implements custom dataset for Affwild2 with Valence-Arousal labels

Based on ABAW-NISL 2021
"""

class VA_Dataset(BaseDataset):
    """
    Constructs a dataset for the VA Challenge.
    Each sample contains at dictionary with tensors sequences of images and audio snippets as features and a numpy array of valence and arousal as labels
    """

    def __init__(self, dataset_file: Path, seq_len, fps, image_size, audio_window_size, audio_sr, train_mode="Train", channel_order="RGB",
                 transforms=None, audio_transforms=None, tiny=False):
        super(VA_Dataset, self).__init__(train_mode=train_mode, transforms=transforms, audio_transforms=audio_transforms, seq_len=seq_len, fps=fps, image_size=image_size, tiny=tiny)
        # overwrite the name
        self.audio_sr = audio_sr
        self.audio_window_size = int(audio_window_size * audio_sr)  # important - turn this into an integer for indexing. Also make sure model input size is consistent
        
        self.channel_order = channel_order
        
        self._name = "VA_dataset"

        # parse the sequences
        self._parse_dataset_paths(dataset_file=dataset_file, task="VA")
        
        print("Finished parsing dataset metadata. Created dataset of size {}".format(self._size))
        

    def _get_all_labels(self):
        if self._data is not None:
            return self._data["label"]
        

    def __getitem__(self, item):
        assert (item < self._size)

        images = []
        #labels = []
        #img_paths = []
        #frames_ids = []
        #video_names = []
        
        #images = torch.zeros(self.seq_len, 3, 112, 112)
        #images = torch.zeros(self.seq_len, 3, self.image_size, self.image_size)
        #df = self.sample_seqs[item]
        start = item * self.seq_len
        end = (item + 1) * self.seq_len
        df = self.sample_seqs.iloc[start:end]
        
        assert isinstance(df, pd.DataFrame), "Sequence metadata should be a Pandas Dataframe, is a {}".format(type(df))
        
        s = 0 # running index from 0
        for i, row in df.iterrows():
            # Image
            img_path = Path(row["path"])  # get the path to the image
            if not img_path.exists():
                img_path = self.face_dir / img_path # set face directory to preset variable - should be the norm if metadata does not save the cropped_aligned path 
                assert img_path.exists()
                
            image = cv2.imread(str(img_path))   # loads BGR by default
            if self.channel_order == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                assert image.shape == (112, 112, 3)
                
            # transformation
            if self.transform is not None:
                image = self.transform(image)
            #else:
            #    image =  torchvision.transforms.ToTensor()(image)
                
            #images[s, ...] = torchvision.transforms.ToTensor()(image)    # image is tensor [C x H x W]
            #images[s, ...] = image
            #s = s + 1
            images.append(image)
            
            #image = Image.open(str(img_path)).convert("RGB") # column major [W x H x C]
            
            #if self.channel_order == "BGR": #flip channels if necessary
            #    image = np.array(image) # convert to numpy array row major [H x W x C]
            #image = image[:,:,::-1].copy() # need to copy bc negative strides on channels axis is not supported by torch
                
            #if self.transform is not None:    
            #    image = self.transform(image) # image is now Tensor [C x H x W]
                
            # get the valence and arousal entries for this timestamp    
            #label = row[PATHS.tasks["VA"]].values.astype(np.float32)    
            #frame_id = row["frames_ids"]
            # append to lists
            #images.append(image)
            #labels.append(label)
            #img_paths.append(str(img_path))
            # frames_ids.append(frame_id)
            # video_names.append(row["video"])
            
        # images 
        #if self.transform is not None:
        #    images = self.transform(images)
        
        #############
        # labels
        labels = df[PATHS.tasks["VA"]].values.astype(np.float32)
        
        #############
        # frame ids
        frames_ids = df["frames_ids"].values.astype(np.int32)
            
        #############
        # video names

        # make sure that sequence is sampled from one video - this is only necessary bc of this code structure with video name as a df row.
        #assert len(np.unique(video_names)) == 1, "Expected a single video per sequence, got {}".format(len(np.unique(video_names)))
        assert df.video.nunique() == 1, "Expected a single video per sequence, got {}".format(df.video.nunique())
        #video_name = np.unique(video_names)[0]
        #video_name = video_names[0]
        video_name = df["video"].values[0]
        # deal with videos that have multiple persons by removing the suffix
        audio_name = original_name(video_name)
        # assemble path to audio file
        audio_file = self.audio_dir / (audio_name + ".wav")
        assert audio_file.exists(), "Audio file {} does not exist!".format(audio_file)

        # compute the audio window
        video_fps = np.unique(df["fps"].values)[0]
        # possibly need to replace single fps parameter with the dataframe info from individual files if not consistent in the entire dataset
        if not video_fps == self.fps: 
            "Requested fps {} does to match that of video {}: {}".format(self.fps, video_name, video_fps)
            
            
        frame_offset = int(max(0, frames_ids[0] * self.audio_sr / video_fps - self.audio_window_size // 2))
        num_audio_frames = int(frames_ids[-1] * self.audio_sr / video_fps + self.audio_window_size // 2) - frame_offset
        if num_audio_frames < self.audio_window_size:
            num_audio_frames = self.audio_window_size
        # read audio
        out, sr = torchaudio.load(str(audio_file), num_frames=num_audio_frames, frame_offset=frame_offset,
                                  normalize=True)
        # check for stereo files
        metadata = torchaudio.info(str(audio_file))
        if metadata.num_channels == 2:
            out = out[0]
        else:
            out = torch.squeeze(out)    # Tensor now has shape [N]
            
        assert self.audio_sr == sr , "audio sample rate needs to be {}".format(self.audio_sr)

        # get audio frames
        audio_frames = []
        audio_length = []

        for video_frame_id in frames_ids:
            
            # audio transform replace random frame
            if self.audio_transforms is not None:
                if np.random.random_sample() < self.audio_transforms["swap"]:
                    center = np.random.randint(0, num_audio_frames) # pick a random audio frame
                else:
                    center = (2 * video_frame_id + 1) / video_fps / 2
                    center = max(0, center * self.audio_sr - frame_offset) # need to subtract the offset because we cut off the audio file  during loading
            else:
                # compute the center of this audio frame
                center = (2 * video_frame_id + 1) / video_fps / 2
                center = max(0, center * self.audio_sr - frame_offset) # need to subtract the offset because we cut off the audio file  during loading
                    
            # compute start and end of frame
            start = int(max(0, center - self.audio_window_size // 2))
            end = int(center + self.audio_window_size // 2)

            # NISL code does some weird stuff here. Should have no effect?
            if end > out.size(-1):  # check if frame is out of bounds, in that case, left shift until it fits
                audio_frame = out[-self.audio_window_size:]
            else:
                audio_frame = out[start:end]
                
            # check if length fits, if not, 0 pad from left
            if audio_frame.size(0) < self.audio_window_size:
                missing = self.audio_window_size - audio_frame.size(0)
                audio_frame = torch.cat([torch.zeros(missing), audio_frame])
            elif audio_frame.size(0) > self.audio_window_size:
                audio_frame = audio_frame[:self.audio_window_size]    
            
            # final check 
            assert audio_frame.size(0) == self.audio_window_size, "Extracted audio frame is the wrong size. Expected {}, got {}".format(self.audio_window_size, audio_frame.size(0))
            
            # audio transform
            if self.audio_transforms is not None:
                # flip frame
                if np.random.random_sample() < self.audio_transforms["flip"]:
                    audio_frame = torch.flip(audio_frame, dims=[0])
                
                # add gaussian noise
                noise = 1 / self.audio_transforms["snr"] * torch.rand(self.audio_window_size)
                audio_frame += noise
            
            # expand the channel dimension again
            audio_frame = torch.unsqueeze(audio_frame, dim=0) # [1, N]
            
            audio_frames.append(audio_frame)
            audio_length.append(audio_frame.size(-1))
        
        assert len(audio_frames) == len(images), "Number of audio frames does not match number of images: {}, {}".format(len(audio_frames), len(images))

        # pack data
        sample = {
            "image": torch.stack(images, dim=0),
            #"image": images,
            "audio": torch.stack(audio_frames, dim=0),
            "audio_length": np.array(audio_length),
            #"label": np.array(labels),
            "label": labels,
            #"path": img_paths,
            "index": item,
            #"frames_ids": np.array(frames_ids),
            "frames_ids": frames_ids,
            "video": video_name,
            "fps": video_fps
        }

        return sample





