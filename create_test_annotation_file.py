"""
Create an "annotation file" for the test partition so it can be loaded into the dataset structure. 
Since test labels are unknown, fill those entries in the data structure with a pseudolabel. 
"""

import pandas as pd
import numpy as np

import cv2
from tqdm import tqdm
import argparse
import pickle
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt

from utils.file_io import read_AU, read_EXPR, read_VA, get_fps, get_frame_count


def frames_to_labels(frames_paths: List[Path], pseudo_value=100, task="VA"):
    """
    Similar to the other partitions, but since there are no labels and no invalid frames, instead gets the frames ids and replicates the pseudo label value into an array.
    :param frames: A list of >=N paths to frames
    :param pseudo_value: The value to save for the labels
    :param Task determines the number of columns. VA = 2, EXPR = 7 AU = 12
    :returns a tuple of a np array
    """
    
    num_outs = {
        "VA": 2,
        "EXPR": 7,
        "AU": 12,
    }[task]
    
    frames_ids = [int(frame.stem) - 1 for frame in frames_paths] 
    
    N = len(frames_paths)
    
    #labels = [pseudo_value] * N
    #pseudo_label_array = np.array(labels)
    pseudo_label_array = np.ones((N, num_outs)) * pseudo_value
    
    return pseudo_label_array, frames_paths, frames_ids


def get_frames(frames_paths: List[Path], frame_count:int):
    """
    Helper function which gets the number of frames in a video and pads the list of frames paths to that length by using the last valid frame.
    Returns: a tuple of a list of paths and a list of frames ids (file name - 1)
    """
      
    assert frame_count >= len(frames_paths), "Video must have at least as many frames as there are image files, but there are {} frames and {} files were passed!".format(frame_count, len(frames_paths))
    
    frames_ids = list(range(frame_count))   # there has to be an unique id for every frame in the video. So we can filter later
    
    if len(frames_paths) == frame_count:
        print("Number of video frames {} matches number of image frames. Nothing to do here".format(frame_count))
        
        return sorted(frames_paths), frames_ids
    
    else:
        print("Padding {} frames to a length of {}".format(frame_count - len(frames_paths), frame_count))
        
        #sorted_paths = sorted(frames_paths)
        padded_paths = []
        
        image_folder = frames_paths[0].parent # only works if all faces are from the same video and contained in the same folder
        
        # set the first path as the first valid frame
        valid_frame = frames_paths[0]
        
        for i in range(frame_count):
            
            target_frame = image_folder / "{:05d}.jpg".format(i + 1)
            if target_frame.exists():
                # update valid frame to the last one found
                valid_frame = target_frame
                padded_paths.append(target_frame)
            else:
                padded_paths.append(valid_frame)
        
        assert len(padded_paths) == len(frames_ids)
        return padded_paths, frames_ids   


def get_test_filenames(annotation_dir: Path=None, task="VA_Estimation_Challenge") -> List[str]:
    """
    Get a list of folder names that form the test partition (without file extensions)
    Not identical to video file names because of the additional people issue resulting in extra folders.
    """
    
    # grab all folder names from the face dir
    #all_folder_names = [f.stem for f in list(face_dir.glob("*")) if f.is_dir()]
    #print("Found {} folders total in the face dir".format(len(all_folder_names)))
    
    #task_folder_name = {
    #    "VA": "VA_Estimation_Challenge",
    #    "AU": "AU_Detection_Challenge",
    #    "EXPR": "EXPR_Classification_Challenge",
    #}[task]
    
    #assert task in ["VA_Estimation_Challenge",  "EXPR_Classification_Challenge", "AU_Detection_Challenge"]
    
    #train_files = [f.stem for f in list((annotation_dir / task / "Train_Set").glob("*.txt"))]
    #print("Found {} train set files".format(len(train_files)))
    
    #validation_files = [f.stem for f in list((annotation_dir / task / "Validation_Set").glob("*.txt"))]
    #print("Found {} validation set files".format(len(validation_files)))
    
    #trainval_files = train_files + validation_files
    
    # get test files by exclusion
    #test_files = [f for f in all_folder_names if not f in trainval_files ]
    
    # read in the text file
    with open(str(annotation_dir / task / "Test_Set" / "Valence_Arousal_Estimation_Challenge_test_set_release.txt")) as f:
        
        test_files = f.readlines()
        test_files = [fi.strip() for fi in test_files]
        test_files = [t for t in test_files if t]
    
    print("Found {} test files".format(len(test_files)))
    
    # make sure we are not missing anything
    assert len(test_files) == 152, "Need 152 test files!"
      
    return test_files
    


def main(args):
    
    pseudo_label = 100  # a label value that is set for all frames
    
    
    # process cmd line args
    annotation_dir = Path(args.annotation_dir)
    assert annotation_dir.exists(), "Annotations dir not found"
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
    
    
    # prepare the output file 
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)
    output_file = args.output_file
    if not output_file.endswith(".pkl"):
        output_file = output_file + ".pkl"
    if task != "all":
        output_file = task + "_" + output_file
        
    # parse all the video file in the video dir
    
    data_file = {}
    ext = "mp4"
    # get all Affwild videos
    old_videos = sorted(list((video_dir / "batch1").glob(f"*.{ext}")))
    # get all new videos from Affwild2
    new_videos = sorted(list((video_dir / "batch2").glob(f"*.{ext}")))
    all_videos = old_videos + new_videos
    
    # replace some of the mp4s with original avis to avoid frame count issues
    #avi_videos = sorted(list((video_dir.parent / "avi_videos = sorted(list(avi_path.glob("*.avi")))").glob("*.avi")))
    original_videos = []
    r = 0
    for v in all_videos:
        avi = (video_dir.parent / "avi_videos" / "{}.avi".format(v.stem))
        if avi.exists():
            original_videos.append(avi)
            r += 1
        else: 
            original_videos.append(v)
    all_videos = original_videos

    print("There are {} avi files".format(r))
    print("Discovered {} videos".format(len(all_videos)))

    save_path = output_dir / output_file
    if save_path.is_file():
        print("Annotations file {} exists, loading ...".format(save_path))
        with open(save_path, "rb") as f:
            data_file = pickle.load(f)
            skip_parsing = True
    else:
        skip_parsing = False
        
        
    # get the names of the test files
    all_filenames = [v.stem for v in all_videos]
    
    # 
    # tasks form highest level
    for task in tasks:
        #modes = ["Training_Set", "Validation_Set"]
        modes = ["Test_Set"]

        # Action Units
        if task == "AU_Detection_Challenge":
            print("Processing AU annotations ... ")
            AU_list = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"] # some AUs are missing?
            #AU_list = ["AU1", "AU2", "AU4", "AU6", "AU12", "AU15", "AU20", "AU25"] # old version only has 8 AU annotations
            if not skip_parsing:
                data_file[task] = {}
                # modes form second highest level
                for mode in modes:
                    # find the files that belong to the test partition
                    test_names = get_test_filenames(annotation_dir=annotation_dir, task=task)
                    data_file[task][mode] = {}
                    for name in tqdm(test_names):
                        print(name)
                        # gather all frames for this file
                        frames_paths = sorted(list((face_dir / name).glob("*.jpg")))
                        # find the number of frames in the video
                        frame_count = get_frame_count(all_videos, video_name=name)
                        
                        # account for a 1 frame difference
                        if len(frames_paths) - frame_count == 1:
                            frame_count += 1
                        
                        frames_paths, frames_ids = get_frames(frames_paths=frames_paths, frame_count=frame_count)
                        assert len(frames_ids) == frame_count
                        data_dict = {}
                        data_dict.update({"path": frames_paths, "frames_ids": frames_ids})
                        # find the video fps
                        fps = get_fps(all_videos, video_name=name)
                        # update with video info
                        data_dict.update({"fps": [fps] * len(frames_paths)})
                        data_dict.update({"frame_count": [frame_count] * len(frames_paths)})
                        # turn into DF
                        data_file[task][mode][name] = pd.DataFrame.from_dict(data_dict)
            if args.vis:
                print("AU visualisation not implemented for test set")
                """
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
                    #plot_pie(AU_list, pos_freq=pos, neg_freq=neg)
                """

        # Expressions
        if task == "EXPR_Classification_Challenge":
            print("*" * 40)
            print("Processing EXPR annotations ...")
            EXPR_List = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
            if not skip_parsing:
                data_file[task] = {}
                for mode in modes:
                    test_names = get_test_filenames(annotation_dir=annotation_dir, task=task)
                    data_file[task][mode] = {}
                    for name in tqdm(test_names):
                        print(name)
                        # gather all frames for this file
                        frames_paths = sorted(list((face_dir / name).glob("*.jpg")))
                        # find the number of frames in the video
                        frame_count = get_frame_count(all_videos, video_name=name)
                        
                        # account for a 1 frame difference
                        if len(frames_paths) - frame_count == 1:
                            frame_count += 1
                        
                        frames_paths, frames_ids = get_frames(frames_paths=frames_paths, frame_count=frame_count)
                        data_dict = {}
                        data_dict.update({"path": frames_paths, "frames_ids": frames_ids})
                        # find the video fps
                        fps = get_fps(all_videos, video_name=name)
                        # update with video info
                        data_dict.update({"fps": [fps] * len(frames_paths)})
                        data_dict.update({"frame_count": [frame_count] * len(frames_paths)})
                        # turn into DF
                        data_file[task][mode][name] = pd.DataFrame.from_dict(data_dict)
            if args.vis:
                #print("EXPR visualisation not implemented")
                """
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
                """

        # Valence Arousal
        if task == "VA_Estimation_Challenge":
            print("Processing VA annotations ...")
            VA_list = ["Valence", "Arousal"]
            if not skip_parsing:
                data_file[task] = {}
                for mode in modes:
                    test_names = get_test_filenames(annotation_dir=annotation_dir, task=task)
                    data_file[task][mode] = {}
                    for name in tqdm(test_names):
                        print(name)
                        # gather all frames for this file
                        frames_paths = sorted(list((face_dir / name).glob("*.jpg")))
                        # find the number of frames in the video
                        frame_count = get_frame_count(all_videos, video_name=name)
                        
                        # account for a 1 frame difference -this sometimes happens
                        if len(frames_paths) - frame_count == 1:
                            frame_count += 1
                        
                        frames_paths, frames_ids = get_frames(frames_paths=frames_paths, frame_count=frame_count)
                        
                        # convert paths to strings and drop cropped_aligned dir
                        frames_paths = ["{}/{}".format(name, fp.name) for fp in frames_paths]
                        assert len(frames_ids) == frame_count
                        data_dict = {}
                        data_dict.update({"path": frames_paths, "frames_ids": frames_ids})
                        # find the video fps
                        fps = get_fps(all_videos, video_name=name)
                        # update with video info
                        data_dict.update({"fps": [fps] * len(frames_paths)})
                        data_dict.update({"frame_count": [frame_count] * len(frames_paths)})
                        # turn into DF
                        df = pd.DataFrame.from_dict(data_dict)
                        df = df.astype(dtype={"path": "string", "fps": "uint8", "frame_count": "uint32", "frames_ids": "uint32"})
                        data_file[task][mode][name] = df
            if args.vis:
                print("VA visualisation not implemented")
                """
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
                """
                pass

    if not skip_parsing:
        print("Saving data ...")
        with open(save_path, "wb") as f:
            pickle.dump(data_file, f)
    



if __name__ ==  "__main__":
    
    parser = argparse.ArgumentParser(description="Create and save an annotation file for Affwild2")
    parser.add_argument("--vis", action="store_true", help="Visualise the label distribution")
    parser.add_argument("--annotation_dir", type=str,
                        default="/data/eihw-gpu5/karasvin/databases/ABAW3_Affwild2/annotations",
                        help="Path to the annotation folders of the challenge")
    parser.add_argument("--output_dir", type=str, default="/data/eihw-gpu5/karasvin/data_preprocessing/ABAW3_Affwild2/annotations", 
                        help="Folder where the output file will be saved")
    parser.add_argument("--output_file", type=str, default="test_files.pkl", help="Name of the output file")
    parser.add_argument("--video_dir", type=str, default="/data/eihw-gpu5/karasvin/databases/ABAW3_Affwild2/videos",
                        help="Path to the videos")
    parser.add_argument("--face_dir", type=str,
                        default="/data/eihw-gpu5/karasvin/databases/ABAW3_Affwild2/cropped_aligned",
                        help="Path to the extracted faces")
    parser.add_argument("--task", type=str, default="all", choices=["VA", "EXPR", "AU"], help="Task to process. Default is all")

    args = parser.parse_args()

    main(args)
