import torch
from torch.utils.data import Dataset

from data.dataset import BaseDataset
from data import PATHS, PATHS_GPU6, PATHS_GPU7
import pickle
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
import os
import math
from tqdm import tqdm
from copy import copy
import cv2
import torchaudio
from utils.file_io import original_name



class TestBaseDataset(BaseDataset):
    """
    Subclass of the Base dataset class, specifically for testing (no labels)
    
    """
    def __init__(self,  video, train_mode="Test", seq_len=16, fps=30, image_size=112, transforms=None, audio_transforms=None, ):
        super().__init__(train_mode, seq_len, fps, image_size, transforms, audio_transforms)
        
        #data for 1 video
        self.video = video
        
        
    def _read_annotation_file(self, file_path, task="VA"):
        """
        Wrapper which loads the annotation file and selects the entry for the desired task and partition
        :param file_path: Path to the pickle dump file that holds the dataset info
        :param task: The task to load (should always be VA)
        :param video: The video to load (we process the test set video by video)
        :return: A dictionary whose keys are video names without file extensions
        """
        
        # change data paths if necessary
        node_name = os.getenv("SLURMD_NODENAME", "eihw-gpu5")
        print("running on node: " + node_name)
        if node_name == "eihw-gpu5":
            self.face_dir = PATHS.face_dir
            self.audio_dir = PATHS.audio_dir
        elif node_name == "eihw-gpu6":
            self.face_dir = PATHS_GPU6.face_dir
            self.audio_dir = PATHS_GPU6.audio_dir
        else:   # node 7
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
            if self._train_mode == "Test":
                data = data["Test_Set"]
            else:
                raise ValueError("Can only load Test partitions")

            return data
        
    def _parse_dataset_paths(self, dataset_file, task):
        """
        Parses the annotation file and loads a list of sequence dataframes, as well as setting size and indices of the dataset
        :param dataset_file: The annotation file
        :param task: The task to load (either VA, EXPR or AU)
        :return: A dictionary corresponding to the selected task and _train_mode
        """

        self._data = self._read_annotation_file(dataset_file, task)
        #stride = 30 // self.fps # stride pulls every nth frame from the data
        stride = 1 # always 1
        
        
        # if we stride > 1 to get dilated sequences, we can stagger the sequence start to still cover all frames in the dataset. If not, we get N/D frames, where D is the dilation
        #stagger_sequences = True
        #if stagger_sequences:
        #    offsets = list(range(stride))[::2]  # create a set starting at every second frame  -> 1/2 data
        #else: 
        #    offsets = [0] # create a set starting at 0th frame -> 1/stride data
        offsets = [0] # always start at 0 
            
        dfs = []    # type: List[pd.DataFrame]
        
        if not self.video:
            videos = self._data.keys()
        else:
            # pick a random video
            #videos = list(self._data.keys())
            #videos = [videos[np.random.randint(0, len(videos))]] # pick a random video
            #print("Randomly selected video: {}".format(videos))
            videos = [self.video]    # only load this video

        # iterate over the video metadata and parse the frames into sequences
        for video in tqdm(videos, total=len(videos)):
            data = self._data[video]
            assert isinstance(data, pd.DataFrame)
            data["video"] = video   # add a column containing the video name (should not have extension)
            data["video"] = data["video"].astype("string") # otherwise will be of type object

            for offset in offsets:
                #sampled_frames = copy(data.iloc[offset:].iloc[::stride]).reset_index()  # pull a sub-sample from the dataframe
                sampled_frames = copy(data.iloc[offset:]).reset_index()
                frame_count = sampled_frames["frame_count"][0]
                assert len(sampled_frames) == frame_count, "The sampled frames should have as many elements {} as there are video frames {}".format(len(sampled_frames), frame_count)
                for i in range(math.ceil(len(sampled_frames) / self.seq_len)):    # round up here to make sure we catch all frames
                    start, end = i * self.seq_len, (i+1) * self.seq_len
                    if end >= len(data):    # check that index is not out of range
                        new_df = sampled_frames.iloc[start:]    # until the last element is reached
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
        
        
class VA_TestDataset(TestBaseDataset):
    """
    VA Dataset but only for testing (no labels)


    Each sample contains at dictionary with tensors sequences of images and audio snippets as features and a integers of frames ids, fps and frame count info instead of labels
    """

    def __init__(self, dataset_file: Path, seq_len, fps, image_size, audio_window_size, audio_sr, train_mode="Test", channel_order="RGB",
                 transforms=None, audio_transforms=None, video=None):
        super(VA_TestDataset, self).__init__(train_mode=train_mode, transforms=transforms, audio_transforms=audio_transforms, seq_len=seq_len, fps=fps, image_size=image_size, video=video)
        # overwrite the name
        self.audio_sr = audio_sr
        self.audio_window_size = int(audio_window_size * audio_sr)  # important - turn this into an integer for indexing. Also make sure model input size is consistent
        
        self.channel_order = channel_order
        
        self._name = "VA_TestDataset"

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
        #labels = df[PATHS.tasks["VA"]].values.astype(np.float32)
        
        #############
        # frame ids
        frames_ids = df["frames_ids"].values.astype(np.int32)
        
        #############
        # frame count
        frame_count = df["frame_count"].values.astype(np.int32)
            
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
            #"label": labels,
            #"path": img_paths,
            "index": item,
            #"frames_ids": np.array(frames_ids),
            "frames_ids": frames_ids,
            "frame_count": frame_count,
            "video": video_name,
            "fps": video_fps
        }

        return sample