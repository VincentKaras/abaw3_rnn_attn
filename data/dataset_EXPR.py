from pathlib import Path

import torch
from PIL import Image

import numpy as np
import pandas as pd

from data import PATHS
from data.dataset import BaseDataset, original_name

import torchaudio
import torchvision

"""
Implements custom dataset for Affwild2 with Valence-Arousal labels

Based on ABAW-NISL 2021
"""

class EXPR_Dataset(BaseDataset):
    """
    Constructs a dataset for the EXPR Challenge.
    Each sample contains at dictionary with tensors sequences of images and audio snippets as features and a numpy array of valence and arousal as labels
    """

    def __init__(self, dataset_file: Path, seq_len, fps, audio_window_size, audio_sr, train_mode="Train", channel_order="RGB",
                 transforms=None):
        super(EXPR_Dataset, self).__init__(train_mode=train_mode, transforms=transforms, seq_len=seq_len, fps=fps)
        # overwrite the name
        self.audio_sr = audio_sr
        self.audio_window_size = audio_window_size
        self._name = "EXPR_dataset"
        
        self.channel_order = channel_order

        # parse the sequences
        self._parse_dataset_paths(dataset_file=dataset_file, task="EXPR")


    def _get_all_labels(self):
        if self._data is not None:
            return self._data["label"]

    def __getitem__(self, item):
        assert (item < self._size)

        images = []
        labels = []
        img_paths = []
        frames_ids = []
        video_names = []

        df = self.sample_seqs[item]
        for i, row in df.iterrows():
            img_path = Path(row["path"])  # get the path to the image
            if not img_path.exists():
                img_path = PATHS.face_dir / img_path.name # replace face directory by preset variable
                assert img_path.exists()
            image = Image.open(str(img_path)).convert("RGB") # column major [W x H x C]
            image = np.array(image) # convert to numpy array row major [H x W x C]
            if self.channel_order == "BGR": #flip channels if necessary
                image = image[:,:,::-1]
            if self.transform is not None:  
                image = self.transform(image) # image is now Tensor [C x H x W]
            label = row["label"].values.astype(np.float32)    # expression column is simply named label since it is an int, not a one-hot vector
            frame_id = row["frames_ids"]
            # append to lists
            images.append(image)
            labels.append(label)
            img_paths.append(img_path)
            frames_ids.append(frame_id)
            video_names.append(row["video"])

        # make sure that sequence is sampled from one video - this is only necessary bc of this code structure with video name as a df row.
        assert len(np.unique(video_names)) == 1
        video_name = np.unique(video_names)[0]
        # deal with videos that have multiple persons by removing the suffix
        audio_name = original_name(video_name)
        # assemble path to audio file
        audio_file = PATHS.audio_dir / (audio_name + ".wav")
        assert audio_file.exists(), "Audio file {} does not exist!".format(audio_file)

        # compute the audio window
        video_fps = np.unique(df["fps"].values)[0]
        # possibly need to replace single fps parameter with the dataframe info from individual files if not consistent in the entire dataset
        assert video_fps == self.fps, "Requested fps {} does to match that of video {}: {}".format(self.fps, video_name,
                                                                                                   video_fps)
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
        assert self.audio_sr == sr , "audio sample rate needs to be {}".format(self.audio_sr)

        # get audio frames
        audio_frames = []
        audio_length = []

        for video_frame_id in frames_ids:
            # compute the center of this audio frame
            center = (2 * video_frame_id + 1) / video_fps / 2
            center = max(0, center * self.audio_sr - frame_offset) # need to subtract the offset because we cut off the audio file  during loading
            # compute start and end of frame
            start = int(max(0, center - self.audio_window_size // 2))
            end = int(center + self.audio_window_size // 2)

            # NISL code does some weird stuff here. Should have no effect?
            if end > out.size(-1):  # check if frame is out of bounds, in that case, left shift until it fits
                audio_frame = out[:-self.audio_window_size]
            else:
                audio_frame = out[start:end]
            audio_frames.append(audio_frame)
            audio_length.append(audio_frame.size(0))
            assert len(audio_frames) == len(images)

            # pack data
            sample = {
                "image": torch.stack(images, dim=0),
                "audio": torch.stack(audio_frames, dim=0),
                "audio_length": audio_length,
                "label": np.array(labels),
                "path": img_paths,
                "index": item,
                "frames_ids": frames_ids,
                "video": video_name,
                "fps": video_fps
            }

            return sample





