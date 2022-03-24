import os
import pickle

import numpy as np
import pandas as pd

import argparse
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from typing import List
import cv2

from copy import deepcopy

from utils.file_io import get_fps, original_name

"""
Creates annotation files for the Affwild2 dataset.

An annotation file holds information on Tasks and Partitions in Dictionaries.
For each video file, the ids of the frames, the paths to those images and the labels are stored. 
Also, the name of the video file is stored. 
The corresponding audio information will be inferred by the dataset class . 
This is done from the name of the video, which matches an audio file and the ids of the frames, which give info on the time.

Based on ABAW NISL 2021 https://github.com/wtomin/UncertainEmotion.git

Vincent Karas, 26/11/2021
"""

parser = argparse.ArgumentParser(description="Create and save an annotation file for Affwild2")
parser.add_argument("--vis", action="store_true", help="Visualise the label distribution")
parser.add_argument("--annotation_dir", type=str,
                    default="/data/eihw-gpu5/karasvin/databases/ABAW3_Affwild2/annotations",
                    help="Path to the annotation folders of the challenge")
parser.add_argument("--output_dir", type=str, default="/data/eihw-gpu5/karasvin/data_preprocessing/ABAW3_Affwild2/annotations", 
                    help="Folder where the output file will be saved")
parser.add_argument("--output_file", type=str, default="annotations.pkl", help="Name of the output file")
parser.add_argument("--video_dir", type=str, default="/data/eihw-gpu5/karasvin/databases/ABAW3_Affwild2/videos",
                    help="Path to the videos")
parser.add_argument("--face_dir", type=str,
                     default="/data/eihw-gpu5/karasvin/databases/ABAW3_Affwild2/cropped_aligned",
                    help="Path to the extracted faces")
parser.add_argument("--task", type=str, default="all", choices=["VA", "EXPR", "AU"], help="Task to process. Default is all")

args = parser.parse_args()

# reader functions for the tasks


def read_AU(file:str):
    with open(file, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
        lines = [x.strip() for x in lines]
        lines = [x.split(",") for x in lines] # split into individual AUs
        lines = [[float(y) for y in x ] for x in lines]
        return np.array(lines, dtype=np.int8)


def read_EXPR(file:str):
    with open(file, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
        lines = [x.strip() for x in lines]
        lines = [int(x) for x in lines]
        return np.array(lines, dtype=np.int8)


def read_VA(file:str, precision=np.float16):
    """
    reads valence and arousal annotations from a file. Expects the first line to be header.
    Args: file (str) File that contains comma separated lines of valence and arousal scores. 
    """
    with open(file, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
        lines = [x.strip() for x in lines]
        lines = [x.split(",") for x in lines]  # split into individual AUs
        lines = [[float(y) for y in x] for x in lines]
        return np.array(lines, dtype=precision)


def plot_pie(AU_list, pos_freq, neg_freq):
    ploting_labels = [x+'+ {0:.2f}'.format(y) for x, y in zip(AU_list, pos_freq)] + [x+'- {0:.2f}'.format(y) for x, y in zip(AU_list, neg_freq)]
    cmap = matplotlib.cm.get_cmap('coolwarm')
    colors = [cmap(x) for x in pos_freq] + [cmap(x) for x in neg_freq]
    fracs = np.ones(len(AU_list)*2)
    plt.pie(fracs, labels=ploting_labels, autopct=None, shadow=False, colors=colors,startangle =78.75)
    plt.title("AUs distribution")
    plt.show()


def frames_to_labels(label_array:np.ndarray, frames:List[Path], invalid_value):
    """
    Key method which maps labels to frames.
    :param label_array: A numpy array (NxL) of labels
    :param frames: A list of >=N frames
    :param invalid_value: Invalid annotation value. Frames and labels with this value are discarded
    :return: A tuple of label array, frames and frames ids
    """
    N = label_array.shape[0]
    label_array = label_array.reshape((N, -1))

    assert len(label_array) >= len(frames), "There need to be at least as many labels as frames"
    assert len(frames) > 0, "No frames were given!"
    print(len(label_array), len(frames))

    frames_ids = [int(frame.stem) - 1 for frame in frames]  # frame ids start at 0, so subtract 1

    def pad_frames_ids():
        """
        pad the frames ids to have the same length as the label array using the last valid frame id
        This assumes labels always start at frame id 0
        :return:
        """
        padded_frames_ids = []
        last = 0
        for i in range(len(label_array)):
            if i in frames_ids:
                padded_frames_ids.append(i)
                last = i
            else:
                # need to insert the last frame
                if i == 0: # if we are at the start, we cant go back - pick the first frame
                    last = frames_ids[0]
                padded_frames_ids.append(last)

        assert len(label_array) == len(padded_frames_ids)

        return padded_frames_ids

    # pad the frames if necessary
    #if len(frames) < len(label_array):
    #    frames_ids = pad_frames_ids()
    #assert len(frames_ids) == len(label_array), "number of labels and frames dont match: {} {}".format(len(label_array), len(frames_ids))

    # print(len(frames_ids))
    # print(max(frames_ids))

    # find rows with one or more invalid labels to drop
    to_drop = np.any(label_array == invalid_value, axis=1)  # Boolean array (N,) that is true where entries are invalid
    print("{} rows are invalid".format(np.count_nonzero(to_drop)))
    # filter the label array
    #filtered_label_array = label_array[np.logical_not(to_drop)]
    #print("Leaving {} rows".format(len(filtered_label_array)))
    # filter the frames ids - unfortunately we cant simply use indexing from the label array because frames list may be longer (or shorter)

    drop_ids = [i for i in range(N) if to_drop[i]]
    keep_ids = [i for i in range(N) if not to_drop[i]]
    print("Keeping {} rows".format(len(keep_ids)))

    #filtered_label_array = label_array[keep_ids, :]
    #filtered_frames_ids = [i for i in frames_ids if i in keep_ids]
    #filtered_frames_ids = list(np.array(frames_ids)[keep_ids])
    # print("{} frame ids remaining".format(len(frames_ids)))

    # drop any invalid frames
    filtered_frames_ids = [i for i in frames_ids if i not in drop_ids]
    # get the indexes of the label array that correspond to valid frames
    indexes = [True if i in filtered_frames_ids else False for i in range(len(label_array))]
    filtered_label_array = label_array[indexes]

    # do another check, then return
    assert len(filtered_label_array) == len(filtered_frames_ids), \
        "Number of valid labels {} does not match number of frames {}".format(len(filtered_label_array),
                                                                              len(filtered_frames_ids))
    # frames ids and frames correspond to each other, so we can filter after dropping
    # filtered_frames = list(np.array(frames)[frames_ids])
    prefix = frames[0].parent
    filtered_frames = [prefix / "{0:05d}.jpg".format(i + 1) for i in filtered_frames_ids]
    for ff in filtered_frames:
        assert ff.exists(), "Path {} does not exist".format(str(ff))
    return filtered_label_array, filtered_frames, filtered_frames_ids


def main():
    # args and checks
    annotation_dir = Path(args.annotation_dir)
    assert annotation_dir.exists(), "Annotations dir {} not found".format(annotation_dir)
    face_dir = Path(args.face_dir)
    assert face_dir.exists(), "Face dir not found"
    video_dir = Path(args.video_dir)
    assert video_dir.exists(), "Video dir not found"
    output_dir = (Path(args.output_dir))
    
    task = args.task
    if task == "all":
        tasks = ["VA_Estimation_Challenge",  "EXPR_Classification_Challenge", "AU_Detection_Challenge"]
    else:
        if task == "VA":
            tasks = ["VA_Estimation_Challenge"]
        if task == "EXPR":
            tasks = ["EXPR_Classification_Challenge"]
        if task == "AU":
            tasks = ["AU_Detection_Challenge"]
            
    print("Got {} tasks: {}".format(len(tasks), tasks))
     
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)
    output_file = args.output_file
    if not output_file.endswith(".pkl"):
        output_file = output_file + ".pkl"
        train_output_file = output_file + ".pkl"
        valid_output_file = output_file + ".pkl"
        
    if task != "all":
        train_output_file = task + "_train_" + output_file
        valid_output_file = task + "_valid_" + output_file
        output_file = task + "_" + output_file   
        
    

    # get all tasks
    #tasks = sorted(list(annotation_dir.glob("*_Set")))
    #tasks = sorted(list(annotation_dir.glob("*_Challenge")))
    #tasks = [str(t.name) for t in tasks]
    #print("Found {} tasks: {}".format(len(tasks), tasks))

    data_file = {}
    
    ext = "mp4"
    # get all Affwild videos
    old_videos = sorted(list((video_dir / "batch1").glob(f"*.{ext}")))
    # get all new videos from Affwild2
    new_videos = sorted(list((video_dir / "batch2").glob(f"*.{ext}")))
    all_videos = old_videos + new_videos

    print("Discovered {} videos".format(len(all_videos)))

    save_path = output_dir / output_file
    train_save_path = output_dir / train_output_file
    valid_save_path = output_dir / valid_output_file
    
    if save_path.is_file():
        print("Annotations file {} exists, loading ...".format(save_path))
        with open(save_path, "rb") as f:
            data_file = pickle.load(f)
            for t in data_file.keys():
                print(t)
                for m in data_file[t].keys():
                    print("-> {}".format(m))
            skip_parsing = True
    else:
        print("Creating new annotations file at {}...".format(save_path))
        skip_parsing = False
        
    if train_save_path.is_file():
        print("Train annotations file {} exists, loading ...".format(train_save_path))
        with open(save_path, "rb") as f:
            train_data_file = pickle.load(f)
            for t in train_data_file.keys():
                print(t)
                for m in train_data_file[t].keys():
                    print("-> {}".format(m))  
                    
        create_train_file = False              
    else:
        create_train_file = True
                    
    if valid_save_path.is_file():
        print("Validation annotations file {} exists, loading ...".format(save_path))
        with open(valid_save_path, "rb") as f:
            valid_data_file = pickle.load(f)
            for t in valid_data_file.keys():
                print(t)
                for m in valid_data_file[t].keys():
                    print("-> {}".format(m))  
        create_valid_file = False
    else:
        create_valid_file = True
    

    ################################################################
    # iteratively construct the nested dictionary for the challenges
    #modes = ["Training_Set", "Validation_Set"]
    
    
    # partitions are always named the same
    modes = ["Train_Set", "Validation_Set"]
    
    # tasks form highest level
    for task in tasks:
        # Action Units
        if task == "AU_Detection_Challenge":
            print("Processing AU annotations ... ")
            AU_list = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"] # some AUs are missing?
            #AU_list = ["AU1", "AU2", "AU4", "AU6", "AU12", "AU15", "AU20", "AU25"] # old version only has 8 AU annotations
            if not skip_parsing:
                data_file[task] = {}
                # modes form second highest level
                for mode in modes:
                    txt_files = (annotation_dir / task / mode).glob("*.txt")
                    data_file[task][mode] = {}
                    for txt_file in tqdm(txt_files):
                        name = txt_file.stem
                        print(name)
                        au_array = read_AU(txt_file)
                        # gather all frames for this file
                        frames_paths = sorted(list((face_dir / name).glob("*.jpg")))
                        au_array, frames_paths, frames_ids = frames_to_labels(au_array, frames_paths, invalid_value=-1)
                        # split by AU
                        data_dict = dict([(AU_list[i], au_array[:, i]) for i in range(len(AU_list))])
                        data_dict.update({"path": frames_paths, "frames_ids": frames_ids})
                        # find the video fps
                        fps = get_fps(all_videos, video_name=name)
                        data_dict.update({"fps": [fps] * len(frames_paths)})
                        # turn into DF
                        data_file[task][mode][name] = pd.DataFrame.from_dict(data_dict)
            if args.vis:
                #print("AU visualisation not implemented")
                
                total_dict = {}
                for mode in modes:
                    total_dict.update({**data_file[task][mode]})
                    all_samples = []
                    for name in total_dict.keys():
                        arr = []
                        for au in AU_list:
                            arr.append(total_dict[name][au].values)
                        arr = np.stack(arr, axis=1)
                        all_samples.append(arr)
                    all_samples = np.concatenate(all_samples, axis=0)
                    pos = np.sum(all_samples, axis=0)/all_samples.shape[0]
                    neg = - np.sum(all_samples-1, axis=0)/all_samples.shape[0]
                    plot_pie(AU_list, pos_freq=pos, neg_freq=neg)
                

        # Expressions
        if task == "EXPR_Classification_Challenge":
            print("*" * 40)
            print("Processing EXPR annotations ...")
            EXPR_List = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
            if not skip_parsing:
                data_file[task] = {}
                for mode in modes:
                    txt_files = (annotation_dir / task / mode).glob("*.txt")
                    data_file[task][mode] = {}
                    for txt_file in tqdm(txt_files):
                        name = txt_file.stem
                        print(name)
                        expr_array = read_EXPR(txt_file)
                        frames_paths = sorted(list((face_dir / name).glob("*.jpg")))
                        expr_array, frames_paths, frames_ids = frames_to_labels(expr_array, frames_paths, invalid_value=-1)
                        data_dict = {"label": expr_array.reshape(-1), "path": frames_paths, "frames_ids": frames_ids}
                        fps = get_fps(all_videos, name)
                        data_dict.update({"fps": [fps] * len(frames_paths)})
                        data_file[task][mode][name] = pd.DataFrame.from_dict(data_dict)
            if args.vis:
                #print("EXPR visualisation not implemented")
                total_dict = {}
                for mode in modes:
                    total_dict.update({**data_file[task][mode]})
                all_samples = np.concatenate([total_dict[x]["label"].values for x in total_dict.keys()], axis=0)
                histogram = np.zeros(len(EXPR_List))
                for i in range(len(EXPR_List)):
                    histogram[i] = sum(all_samples==i) / all_samples.shape[0]
                plt.bar(np.arange(len(EXPR_List)), histogram)
                plt.xticks(np.arange(len(EXPR_List)), EXPR_List)
                plt.show()

        # Valence Arousal
        if task == "VA_Estimation_Challenge":
            print("Processing VA annotations ...")
            VA_list = ["Valence", "Arousal"]
            if not skip_parsing:
                data_file[task] = {}
                for mode in modes:
                    txt_files = (annotation_dir / task / mode).glob("*.txt")
                    data_file[task][mode] = {}
                    # iterate over the annotation files
                    for txt_file in tqdm(txt_files):
                        # read labels and glob image paths
                        name = txt_file.stem
                        va_array = read_VA(txt_file, precision=np.float32)
                        frames_paths = sorted(list((face_dir / name).glob("*.jpg")))
                        
                        # filter the frames by labels 
                        va_array, frames_paths, frames_ids = frames_to_labels(va_array, frames_paths, invalid_value=-5.0)
                        
                         # turn the list of paths into a list of strings without the face dir (redundant info) 
                        frames = ["{}/{}".format(name, fp.name) for fp in frames_paths]
                        
                        # create a dataframe
                        data_dict = {"valence": va_array[:, 0], "arousal": va_array[:, 1], "path": frames,
                                     "frames_ids": frames_ids}
                        fps = get_fps(all_videos=all_videos, video_name=name)
                        data_dict.update({"fps": [fps] * len(frames)})
                        df = pd.DataFrame.from_dict(data_dict)
                        df = df.astype(dtype={"valence": "float16", "arousal": "float32", "path": "string", "fps": "uint8", "frames_ids": "uint32"})
                        data_file[task][mode][name] = df
            if args.vis:
                #print("VA visualisation not implemented")
                total_dict = {}
                all_samples = []
                # get all modes (partitions of dataset)
                for mode in modes:
                    total_dict.update(**data_file[task][mode])  # update the dict with the unpacked dictionaries
                # iterate over the videos
                for name in total_dict.keys():
                    samples = np.stack([total_dict[name][l].values for l in ["valence", "arousal"]], axis=1)
                    all_samples.append(samples)
                all_samples = np.concatenate(all_samples, axis=0)
                # create a 2D histogram of the labels
                plt.hist2d(all_samples[:, 0], all_samples[:, 1], bins=(20, 20), cmap=plt.cm.get_cmap("PuOr"))
                plt.xlabel("Valence")
                plt.ylabel("Arousal")
                plt.colorbar()
                plt.show()

    if not skip_parsing:
        print("Saving data ...")
        with open(save_path, "wb") as f:
            pickle.dump(data_file, f)
            
        # save for individual partitions
        
    if create_train_file:
        train_data = deepcopy(data_file)       
        # delete unnecessary data
        for task in tasks:
            train_data[task].pop("Validation_Set")
            
            
        with open(train_save_path, "wb") as f:
            pickle.dump(train_data, f)
       
    if create_valid_file:    
        valid_data = deepcopy(data_file)
        
        for task in tasks:
            valid_data[task].pop("Train_Set")
         
        with open(valid_save_path, "wb") as f:
            pickle.dump(valid_data, f)

 
        
if __name__ == "__main__":
    main()


