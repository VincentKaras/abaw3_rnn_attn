from end2you.utils import Params
from pathlib import Path

from models.modules import get_activation

MODEL_PATHS = Params(dict_params={
    "pretrained_dir": Path("/data/eihw-gpu5/karasvin/models/pretrained/"),
    "models_dir": Path("/data/eihw-gpu5/karasvin/trained_models/"),
    "vggface2_models": ["resnet50_ft", 
                      "resnet50_scratch",
                      "senet50_ft",
                      "senet50_scratch"],
    "mobilefacenet_pretrained": Path("/data/eihw-gpu5/karasvin/models/pretrained/MobileFaceNet/mobilefacenet_model_best.pth.tar"),
})