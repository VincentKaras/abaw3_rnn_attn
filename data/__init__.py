from end2you.utils import Params
from pathlib import Path

PATHS = Params(dict_params={
    "train_val_file": Path("/data/eihw-gpu5/karasvin/databases/ABAW3_Affwild2/annotations/VA_train_val_annotations.pkl"),
    "train_file": Path("/data/eihw-gpu5/karasvin/databases/ABAW3_Affwild2/annotations/VA_train_annotations.pkl"),
    "valid_file": Path("/data/eihw-gpu5/karasvin/databases/ABAW3_Affwild2/annotations/VA_valid_annotations.pkl"),
    "test_file": Path("/data/eihw-gpu5/karasvin/databases/ABAW3_Affwild2/annotations/VA_test_files.pkl"),
    "tasks": {"AU": ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU15','AU23','AU24','AU25','AU26'],
              "EXPR": ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise'],
              "VA": ['valence', 'arousal']},
    "audio_dir": Path("/data/eihw-gpu5/karasvin/databases/ABAW3_Affwild2/audio_mono"),
    "face_dir": Path("/data/eihw-gpu5/karasvin/databases/ABAW3_Affwild2/cropped_aligned")
})

# Paths on GPU 6
PATHS_GPU6 = Params(dict_params={
    "train_val_file": Path("/data/eihw-gpu6/karasvin/databases/ABAW3_Affwild2/annotations/VA_train_val_efficient_annotations.pkl"),
    "train_file": Path("/data/eihw-gpu6/karasvin/databases/ABAW3_Affwild2/annotations/VA_train_efficient_annotations.pkl"),
    "valid_file": Path("/data/eihw-gpu6/karasvin/databases/ABAW3_Affwild2/annotations/VA_valid_efficient_annotations.pkl"),
    "test_file": Path("/data/eihw-gpu6/karasvin/databases/ABAW3_Affwild2/annotations/VA_test_files.pkl"),
    "tasks": {"AU": ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU15','AU23','AU24','AU25','AU26'],
              "EXPR": ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise'],
              "VA": ['valence', 'arousal']},
    "audio_dir": Path("/data/eihw-gpu6/karasvin/databases/ABAW3_Affwild2/audio_mono"),
    "face_dir": Path("/data/eihw-gpu6/karasvin/databases/ABAW3_Affwild2/cropped_aligned"),
})

# Paths on GPU 7
PATHS_GPU7 = Params(dict_params={
    "train_val_file": Path("/data/eihw-gpu7/karasvin/databases/ABAW3_Affwild2/annotations/VA_train_val_efficient_annotations.pkl"),
    "train_file": Path("/data/eihw-gpu7/karasvin/databases/ABAW3_Affwild2/annotations/VA_train_efficient_annotations.pkl"),
    "valid_file": Path("/data/eihw-gpu7/karasvin/databases/ABAW3_Affwild2/annotations/VA_valid_efficient_annotations.pkl"),
    "test_file": Path("/data/eihw-gpu7/karasvin/databases/ABAW3_Affwild2/annotations/VA_test_files.pkl"),
    "tasks": {"AU": ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU15','AU23','AU24','AU25','AU26'],
              "EXPR": ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise'],
              "VA": ['valence', 'arousal']},
    "audio_dir": Path("/data/eihw-gpu7/karasvin/databases/ABAW3_Affwild2/audio_mono"),
    "face_dir": Path("/data/eihw-gpu7/karasvin/databases/ABAW3_Affwild2/cropped_aligned"),
})