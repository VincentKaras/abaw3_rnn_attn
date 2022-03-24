"""
general file operations like reading annotation files

"""

import numpy as np
from typing import List
from pathlib import Path
import cv2
import matplotlib
import matplotlib.pyplot as plt


def read_AU(file:str):
    with open(file, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
        lines = [x.strip() for x in lines]
        lines = [x.split(",") for x in lines] # split into individual AUs
        lines = [[float(y) for y in x ] for x in lines]
        return np.array(lines)


def read_EXPR(file:str):
    with open(file, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
        lines = [x.strip() for x in lines]
        lines = [int(x) for x in lines]
        return np.array(lines)


def read_VA(file:str):
    with open(file, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
        lines = [x.strip() for x in lines]
        lines = [x.split(",") for x in lines]  # split into individual AUs
        lines = [[float(y) for y in x] for x in lines]
        return np.array(lines)
    

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


def get_fps(all_videos:List[Path], video_name:str) -> int:
    """
    Helper method that determines the fps of a video using cv2
    :param all_videos: List of Paths of all videos in the dataset
    :param video_name: the name of the video
    :return: The FPS rounded to integer
    """
    
    # get the path to the video
    video_path = [v for i, v in enumerate(all_videos) if v.stem == original_name(video_name)]
    assert len(video_path) == 1, "No unique match was found for {}: {}".format(original_name(video_name), video_path)
    assert video_path[0].exists(), "Video file {} does not exist".format(video_path[0])
    
    video = cv2.VideoCapture(str(video_path[0]))

    fps = int(np.round(video.get(cv2.CAP_PROP_FPS)))
    video.release()
    
    assert fps > 0, "FPS has to be greater than 0, is {}".format(fps)

    return fps


def get_frame_count(all_videos:List[Path], video_name:str) -> int:
    """
    Helper method that determines the total number of frames in a video
    """
    
    # get the video path
    video_path = [v for i, v in enumerate(all_videos) if v.stem == original_name(video_name)]
    assert len(video_path) == 1, "No unique match was found for {}: {}".format(original_name(video_name), video_path)
    assert video_path[0].exists(), "Video file {} does not exist".format(video_path[0])
    
    video = cv2.VideoCapture(str(video_path[0]))
    
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) + 1 # +1 because we seem to be always 1 off from the challenge org.
    #frame_count = 0
    # loop over the frames of the video
    #while True:
		# grab the current frame
    #    (grabbed, frame) = video.read()
	 
		# check to see if we have reached the end of the
		# video
    #    if not grabbed:
    #        break

		# increment the total number of frames read
    #    frame_count += 1
    
    
    video.release()
    
    assert frame_count > 0, "There need to be at least 1 frames, is {}".format(frame_count)
    
    return frame_count


def plot_pie(AU_list, pos_freq, neg_freq):
    ploting_labels = [x+'+ {0:.2f}'.format(y) for x, y in zip(AU_list, pos_freq)] + [x+'- {0:.2f}'.format(y) for x, y in zip(AU_list, neg_freq)]
    cmap = matplotlib.cm.get_cmap('coolwarm')
    colors = [cmap(x) for x in pos_freq] + [cmap(x) for x in neg_freq]
    fracs = np.ones(len(AU_list)*2)
    plt.pie(fracs, labels=ploting_labels, autopct=None, shadow=False, colors=colors,startangle =78.75)
    plt.title("AUs distribution")
    plt.show()
