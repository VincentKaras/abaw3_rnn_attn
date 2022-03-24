import pandas as pd
from PIL import Image
from pathlib import Path
from copy import copy
import numpy as np
from tqdm import tqdm
import pickle
import os
from torch.utils.data import Dataset
from typing import List

from data import PATHS, PATHS_GPU6, PATHS_GPU7

"""
Provides a basic dataset implementation for Affwild2 dataset

Based on the ABAW NISL 2021 code

Vincent Karas 23/11/2021
"""


class BaseDataset(Dataset):

    def __init__(self, train_mode="Train", seq_len=8, fps=30, image_size=112, transforms=None, audio_transforms=None, tiny=False):

        super(BaseDataset, self).__init__()
        self._name = "BaseDataset"
        self._train_mode = train_mode
        #self._transforms = transforms

        self._data = None
        self._ids = None
        self._size = 0

        self.seq_len = seq_len
        self.fps = fps
        self.image_size=image_size
        self.sample_seqs = None # type: pd.DataFrame

        self._VALID_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "ppm", "bmp"]

        self._set_transforms(transforms)
        self.audio_transforms = audio_transforms
        
        # limits the size of the dataset for debugging
        self.tiny = tiny
        
        # data paths - defaults to node 5
        self.face_dir = PATHS.face_dir
        self.audio_dir = PATHS.audio_dir
        
               

    def __len__(self):
        return self._size

    def __getitem__(self, item):
        """
        Abstract method of the dataset class, needs to be implemented by the subclass
        :param item:
        :return:
        """
        pass

    def _set_transforms(self, transforms):
        """
        Method for subclasses to set transforms on the dataset
        :return:
        """
        if transforms is not None:
            self.transform = transforms
        else: 
            self.transform = lambda x: x

    def get_transforms(self):
        return self.transform

    def _is_image_file(self, filename: Path) -> bool:
        return filename.suffix.lower() in self._VALID_IMAGE_EXTENSIONS

    def _is_csv_file(self, filename: Path) -> bool:
        return str(filename.suffix) == "csv"

    def _read_annotation_file(self, file_path, task="VA"):
        """
        Wrapper which loads the annotation file and selects the entry for the desired task and partition
        :param file_path: Path to the pickle dump file that holds the dataset info
        :param task: The task to load (should always be VA)
        :return: A dictionary whose keys are video names without file extensions
        """
        
        # change data paths if necessary
        #if "eihw-gpu7" in str(file_path):
        node_name = os.getenv("SLURMD_NODENAME", "eihw-gpu5")
        print("running on node: " + node_name)
        if node_name == "eihw-gpu5":
            self.face_dir = PATHS.face_dir
            self.audio_dir = PATHS.audio_dir
        elif node_name == "eihw-gpu6":
            self.face_dir = PATHS_GPU6.face_dir
            self.audio_dir = PATHS_GPU6.audio_dir
        elif node_name == "eihw-gpu7":   # node 7
            self.face_dir = PATHS_GPU7.face_dir
            self.audio_dir = PATHS_GPU7.audio_dir

        with open(file_path, "rb") as f:

            if task == "VA":
                challenge = "VA_Estimation_Challenge"
            elif task == "EXPR":
                challenge = "EXPR_Classification_Challenge"
            elif task == "AU":
                challenge = "AU_Detection_Challenge"
            else:
                raise ValueError("Task {} not available".format(task))

            data = pickle.load(f)
            data = data[challenge]

            # choose the partition to load
            if self._train_mode == "Train":
                data = data["Train_Set"]
            elif self._train_mode == "Validation":
                data = data["Validation_Set"]
            else:
                raise ValueError("Can only load Train or Validation partitions")

            return data

    def _parse_dataset_paths(self, dataset_file, task):
        """
        Parses the annotation file and loads a list of sequence dataframes, as well as setting size and indices of the dataset
        :param dataset_file: The annotation file
        :param task: The task to load (either VA, EXPR or AU)
        :return: A dictionary corresponding to the selected task and _train_mode
        """

        self._data = self._read_annotation_file(dataset_file, task)
        stride = 30 // self.fps # stride pulls every nth frame from the data
        
        # if we stride > 1 to get dilated sequences, we can stagger the sequence start to still cover all frames in the dataset. If not, we get N/D frames, where D is the dilation
        stagger_sequences = True
        if stagger_sequences:
            offsets = list(range(stride))[::1]  # create a set starting at every second frame  -> 1/2 data
        else: 
            offsets = [0] # create a set starting at 0th frame -> 1/stride data
            
        dfs = []    # type: List[pd.DataFrame]
        
        if not self.tiny:
            videos = self._data.keys()
        else:
            # pick a random video
            videos = list(self._data.keys())
            videos = [videos[np.random.randint(0, len(videos))]] # pick a random video
            print("Randomly selected video: {}".format(videos))

        # iterate over the video metadata and parse the frames into sequences
        for video in tqdm(videos, total=len(videos)):
            data = self._data[video]
            assert isinstance(data, pd.DataFrame)
            data["video"] = video   # add a column containing the video name (should not have extension)
            data["video"] = data["video"].astype("string") # otherwise will be of type object

            for offset in offsets:
                sampled_frames = copy(data.iloc[offset:].iloc[::stride]).reset_index()  # pull a sub-sample from the dataframe
                for i in range(len(sampled_frames) // self.seq_len):
                    start, end = i * self.seq_len, (i+1) * self.seq_len
                    if end >= len(data):    # check that index is not out of range
                        new_df = sampled_frames.iloc[start:-1]
                    else:
                        new_df = sampled_frames.iloc[start:end]
                    if not len(new_df) == self.seq_len:
                        assert len(new_df) < self.seq_len
                        # pad sequences that are too short by replicating the last frame
                        missing_rows = self.seq_len - len(new_df)
                        new_df = new_df.append([new_df.iloc[-1]] * missing_rows)

                    # debugging assertion statements
                    assert isinstance(new_df, pd.DataFrame), "Video {} sequence {} did not yield a Pandas Dataframe: {}".format(video, i, type(new_df))

                    dfs.append(new_df)
            
            del data
                    
        # count the number of items in the list of samples
        self._ids = np.arange(len(dfs))
        self._size = len(self._ids)
        # concatenate the dataframes into one big df to avoid the memory issue - need to change the access then in __getitem__
        self.sample_seqs = pd.concat(dfs, axis=0)   # type: pd.DataFrame
        self.sample_seqs.reset_index(inplace=True, drop=True) # change the index to 0 - N * seq_length
        
        del dfs
       


def original_name(video_name:str):
    """
    Removes the suffix that indicates it is a 2-subject video to get the original name of the file.
    Requires that the file format has already been stripped from the end of the file name
    """
    if video_name.endswith("_left"):
        return video_name[:-5]
    elif video_name.endswith("_right"):
        return video_name[:-6]
    else:
        return video_name






