import torch
import argparse
from pathlib import Path
from end2you.utils import Params
from typing import Dict
import json

from datetime import datetime


"""
File which handles loading of commandline arguments and providing them to the program as options.

Vincent Karas, 29/11/2021
"""

class Options():
    """
    Helper class which wraps around a parser object and stores information.
    Also is compatible with the E2Y Params class
    """

    def __init__(self):

        self._parser = self._add_parsers()
        self._is_parsed = False
        self._is_initialized = False
        self._is_training = False
        self._process = ""

        self._params = None
        # get the current date
        self._date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        self.dict_options = {}


    def _add_train_args(self, parser:argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add train arguments
        :param parser: The training subparser
        :return: the modified training subparser
        """
        # train scheme
        parser.add_argument("--train_scheme", type=str, choices=["single", "finetune"], default="finetune", 
                            help="Type of training scheme. Train all model parameters (single) or freeze extractors first, then finetune (finetune)")
        
        # dataset
        parser.add_argument("--num_train_workers", type=int, default=4,
                            help="Number of workers for loading train data (defaults to 4")
        parser.add_argument("--pin_memory", type=bool, default=False, help="Whether to use pinned memory for the dataloaders. May result in issues! Default False")
        parser.add_argument("--train_dataset_file", type=str, required=True,
                            help="Path to the pkl file that describes the train dataset")
        # optimizer
        parser.add_argument("--optimizer", type=str, default="Adam", choices=["sgd", "rmsprop", "adam", "adamw"],
                            help="Type of optimizer to use for training (defaults to Adam)")
        parser.add_argument("--lr", type=float, default=1e-3, help="Default learning rate")
        parser.add_argument("--fine_lr", type=float, default=1e-4, help="Learning rate in the finetuning stage (default 1e-4)")
        parser.add_argument("--lr_policy", type=str, default="none", choices=["none", "step", "cosine"],
                            help="Which learning rate policy is applied. Default is none (no learning rate adaptation")
        parser.add_argument("--lr_decay", type=int, default=4, help="Reduce learning rate to 0.1*lr every n epochs")
        parser.add_argument("--fine_lr_decay", type=int, default=4, help="Reduce learning rate to 0.1*lr during fine tuning stage every n epochs")
        parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs to train")
        parser.add_argument("--fine_num_epochs", type=int, default=20, help="Number of epochs to train during fine-tuning stage")
        parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
        parser.add_argument("--save_summary_steps", type=int, default=10,
                            help="Perform evaluation every n steps during training")
        parser.add_argument("--pretrained_path", type=str, default=None, help="Path to checkpoint file of pretrained model")
        parser.add_argument("--loss", type=str, default="ccc", nargs="+", help="Which loss to use for training (default ccc). Supports multiple losses if desired")
        parser.add_argument("--loss_weights", type=float, default=1.0, nargs="+", help="Weights of the losses (defaults to 1.0). Specify for each loss")
        parser.add_argument("--lambda_mse", default=1.0, type=float, help="Weight for the mse loss term. Default 1.0")
        parser.add_argument("--lambda_dis", type=float, default=1.0, help="Weight for the discrete head loss term. Default 1.0")

        return parser


    def _add_eval_args(self, parser:argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add evaluation arguments
        :param parser: The training subparser
        :return: the modified training subparser
        """

        parser.add_argument("--num_valid_workers", type=int, default=4,
                            help="Number of workers for loading eval data (defaults to 4")
        parser.add_argument("--valid_dataset_file", type=str, required=True,
                            help="Path to the pkl file that describes the validation dataset")
        parser.add_argument("--metric_name", type=str, help="Metric to use for validation", default="ccc", choices=["ccc"])

        return parser


    def _add_test_args(self, parser:argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add test arguments
        :param parser: The testing subparser
        :return: The modified testing subparser
        """

        parser.add_argument("--num_test_workers", type=int, default=4,
                            help="Number of workers for loading test data (defaults to 4")
        parser.add_argument("--test_dataset_file", type=str, required=True,
                            help="Path to the pkl file that describes the test dataset")
        parser.add_argument("--model_path", type=str, required=True,
                            help="Path to the model to test")
        parser.add_argument("--prediction_file", type=str, default="predictions.csv",
                            help="The file to write predictions in csv format")

        return parser

    def _add_gen_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add generation arguments
        :param parser: The generation subparser
        :return: The modified generation subparser
        """

        return parser


    def _add_parsers(self) -> argparse.ArgumentParser():
        """
        construct the parser object.
        Adds arguments that are always needed, then adds process-specific arguments via subparsers
        :return: An ArgumentParser
        """
        #TODO add model choices

        # create the parser
        parser = argparse.ArgumentParser(description="A parser for this cool project. Adds flags")

        # MODEL args
        parser.add_argument("--name", type=str, default="experiment_1", help="Name of the experiment. Controls the saving location")
        parser.add_argument("--model_name", type=str, help="Which model architecture to use", choices=["concatselfattn", 
                                                                                                       "concatrnn", 
                                                                                                       "concatbirnn",
                                                                                                       "mult", 
                                                                                                       "multv2",
                                                                                                       "gatedmultv2",
                                                                                                       "concatlinear",
                                                                                                        "visualselfattn",
                                                                                                        "visualrnn",
                                                                                                        "audioselfattn",
                                                                                                        "audiornn",
                                                                                                        "hybrid",
                                                                                                        "auxattn",
                                                                                                        "auxrnn"],
                            default="multv2")
        parser.add_argument("--visual_backbone", type=str, help="The visual extraction network", 
                            choices=["senet50_ft", "senet50_scratch", "resnet50_ft", "resnet50_scratch", "inceptionresnetv1", "mobilefacenet"], default="inceptionresnetv1")
        parser.add_argument("--visual_pretrained", type=bool, default=True, help="Use pretrained network for visual cnn backbone. Default True")
        parser.add_argument("--audio_backbone", type=str, help="The audio extraction network", choices=["emo16", "emo18", "zhao19"], default="zhao19")
        parser.add_argument("--audio_pretrained", type=bool, default=True, 
                            help="Use pretrained network for the audio cnn. Default True. Careful - ensure pretrained weights for audio model are available")
        parser.add_argument("--base_modality", type=str, default="audio", #required=True,
                            help="Modality for the base model", choices=["audio", "visual"])
        parser.add_argument("--num_outputs", type=int, default=2, help="Number of outputs of the model (defaults to 2)")
        parser.add_argument("--num_bins", type=int, default=1, help="number of bins to split continuous output into (default 1 no splitting)")
        parser.add_argument("--d_embedding", type=int, default=128, help="Size of the embedding in the temporal model. Default 128")
        parser.add_argument("--d_feedforward", type=int, default=128, help="Size of the feedforward dimension in the attention model. Default 128")
        parser.add_argument("--num_heads", type=int, default=4, choices=[2, 4, 8], help="Number of heads for Multi Head Attention. Default 4")
        parser.add_argument("--num_layers", type=int, default=4, help="Number of layers for the temporal model. Default is 4")
        parser.add_argument("--num_crossaudio_layers", type=int, default=4, help="Number of layers in the crossmodal audio to visual module of MulT (default 4")
        parser.add_argument("--num_crossvisual_layers", type=int, default=4, help="Number of layers in the crossmodal visual to audio module of MulT (default 4")
        parser.add_argument("--num_layers_audio", type=int, default=4, help="Number of layers for the audio model. Default is 4")
        parser.add_argument("--num_layers_visual", type=int, default=4, help="Number of layers for the visual model. Default is 4")
        parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate. Default 0.3")
        parser.add_argument("--activation", type=str, default="selu", help="Activation function. Default selu")
        parser.add_argument("--bidirectional", type=bool, default=True, help="Use bidirectional layers for recurrent models. Default True")
        
        # DATA args
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size to use. (defaults to 8)")
        parser.add_argument("--seq_length", type=int, default=8, help="Length of sequences to use")
        parser.add_argument("--fps", type=int, default=30, help="Image frame rate per second. Default 30 (dataset frame rate). Changing to <30 allows for dilated sequences on the train set")
        parser.add_argument("--image_size", type=int, default=112, help="Width and height of an image frame")
        parser.add_argument("--image_channel_order", type=str, default="RGB", choices=["RGB", "BGR"], help="Order to load image channels (default RGB)")
        parser.add_argument("--sr", type=int, default=16000, help="Audio sample rate in Hz. (defaults to 16000)")
        parser.add_argument("--window_size", type=float, default=0.66, help="Length of an audio frame in seconds")

        # GPU args
        parser.add_argument("--cuda", type=bool, default=False, help="Use CUDA. (defaults to False")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (defaults to 1)")

        # Multitask use case, specifying multiple tasks is possible
        #parser.add_argument("--tasks", type=str, default=["EXPR", "AU", "VA"], nargs="+")
        #parser.add_argument("--dataset_names", type=str, default=["Mixed_EXPR", "Mixed_AU", "Mixed_VA"], nargs="+")
        # Single Task use case, only one task may be specified
        parser.add_argument("--task", type=str, default="VA", choices=["VA", "EXPR", "AU"], help="Task to process")

        # PATH args
        parser.add_argument("--root_dir", type=str, default="./embed", help="Root folder for the output")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Models are saved here")
        parser.add_argument("--log_dir", type=str, default="./logs", help="Logs are saved here")



        # add subparsers
        subparsers = parser.add_subparsers(help="Should be [train, test, generate]", required=True, dest="process")

        # add to the parser

        train_subparser = subparsers.add_parser(name="train", help="Training arguments")
        train_subparser = self._add_train_args(train_subparser)
        train_subparser = self._add_eval_args(train_subparser)

        test_subparser = subparsers.add_parser(name="test", help="Testing arguments")
        test_subparser = self._add_test_args(test_subparser)

        gen_subparser = subparsers.add_parser(name="generate", help="Generation arguments")
        gen_subparser = self._add_gen_args(gen_subparser)

        self._is_initialized = True

        return parser


    def parse(self):
        """
        Helper method that reads the commandline arguments and stores information internally
        :return: The internal Params object that contains the options from the commandline args + any additional info
        """

        #TODO figure out if process flag is supposed to do hierarchical dicts in the Params object, currently its just treated as another key

        # create parsers if they dont exist
        if not self._is_initialized:
            self._add_parsers()

        args = self._parser.parse_args()

        # turn args object into dict
        self.dict_options = vars(args)

        # package the arguments into a (nested) Params object following the E2Y example (see the docs )
        self._process = args.process
        
        if "train" in self._process.lower():
            self._params = self._get_train_params()
            
        elif "test" in self._process.lower():
            self._params = self._get_test_params()
            
        else:
            raise ValueError("Operation {} not recognized".format(self._process))
            self._params = Params(dict_params={})


        """
        self._process = args.process

        # select subparser option and set is_training flag
        if self._process == "train":
            # create a Params object - might not be necessary as args seems to behave much in the same way
            self._params = Params(dict_params={
                **dict_options, "is_training": True
            })
            self._is_training = True
        else: # currently no other options sets the is_training flag
            self._params = Params(dict_params={
                **dict_options, "is_training": False
            })

        """
        
        print("Current date: ", self._date)

        # print out args
        self._print(self.dict_options)
        # save args to json
        self._save(self.dict_options)

        self._is_parsed = True

        return self._params


    def _print(self, args:Dict):
        print("-" * 20 + " Options " + "-" * 20)
        for k, v in sorted(args.items()):
            print("{} : {}".format(str(k), str(v)))
        print("-" * 49)


    def _save(self, args:Dict):
        """
        Helper function that saves the commandline arguments to a file in JSON format
        :param args: A dictionary created from the commandline args
        :return:
        """

        #experiment_dir = Path(self._params.checkpoints_dir) / self._params.name
        experiment_dir = Path(self._params.checkpoints_dir)  / "options"    
        #print(experiment_dir)
        #if self._is_training and not experiment_dir.exists():
        if not experiment_dir.exists():
            experiment_dir.mkdir(parents=True, exist_ok=True)
        else:
            assert experiment_dir.exists(), "Experiment dir {} does not exist".format(experiment_dir)

        # save as json file
        file_name = experiment_dir / "options_{}.json".format(self._process)
        with open(file_name, "w") as f:
            json.dump(args, f, indent=6)
            
            
    def _get_train_params(self) -> Params:
        """
        Helper which assembles a Params object for the train process
        """
        
        # check the directories, if necessary, create new run
        root_path = Path(self.dict_options["root_dir"])
        if root_path.exists():
            #root_dir = self.dict_options["root_dir"]
            print("Warning: Root folder {} already exists!".format(str(root_path)))
            print("Creating new subfolder with current date and changing logs and ckpt dirs to that folders subfolders...")
            self.dict_options["root_dir"] = str(root_path / str(self._date))
            self.dict_options["checkpoints_dir"] = str(root_path / str(self._date) / "checkpoints")
            self.dict_options["log_dir"] = str(root_path / str(self._date) / "logs")
        
        
        train_params = Params(dict_params={
            "process": "train",         # necessary atm
            "train": Params(dict_params={
                "scheme": self.dict_options["train_scheme"],
                "loss": self.dict_options["loss"],
                "lambda_dis": self.dict_options["lambda_dis"],
                "lambda_mse": self.dict_options["lambda_mse"],
                "dataset_file": self.dict_options["train_dataset_file"],
                "optimizer": self.dict_options["optimizer"],
                "lr_policy": self.dict_options["lr_policy"],
                "lr": self.dict_options["lr"],
                "fine_lr": self.dict_options["fine_lr"],
                "lr_decay": self.dict_options["lr_decay"],
                "fine_lr_decay": self.dict_options["fine_lr_decay"],
                "num_epochs": self.dict_options["num_epochs"],
                "fine_num_epochs": self.dict_options["fine_num_epochs"],
                "weight_decay": self.dict_options["weight_decay"],
                "cuda": self.dict_options["cuda"],
                "base_modality": self.dict_options["base_modality"],
                "is_training": True,
                "partition": "Train",
                "seq_length": self.dict_options["seq_length"],
                "fps": self.dict_options["fps"],
                "image_size": self.dict_options["image_size"],
                "image_channel_order": self.dict_options["image_channel_order"],
                "audio_sr": self.dict_options["sr"],
                "audio_window_size": self.dict_options["window_size"],
                "batch_size": self.dict_options["batch_size"],
                "num_workers": self.dict_options["num_train_workers"],
                "pin_memory": self.dict_options["pin_memory"],
                "save_summary_steps": self.dict_options["save_summary_steps"],
            }),
            "valid": Params(dict_params={
                "dataset_file": self.dict_options["valid_dataset_file"],
                "cuda": self.dict_options["cuda"],
                "base_modality": self.dict_options["base_modality"],
                "is_training": False,
                "partition": "Validation",
                "seq_length": self.dict_options["seq_length"],
                "fps": self.dict_options["fps"],
                "image_size": self.dict_options["image_size"],
                "image_channel_order": self.dict_options["image_channel_order"],
                "audio_sr": self.dict_options["sr"],
                "audio_window_size": self.dict_options["window_size"],
                "batch_size": self.dict_options["batch_size"],
                "num_workers": self.dict_options["num_valid_workers"],
                "pin_memory": self.dict_options["pin_memory"],
                "save_summary_steps": self.dict_options["save_summary_steps"], 
                "metric_name": self.dict_options["metric_name"],    
            }),
            "model": Params(dict_params={
                "num_outs": self.dict_options["num_outputs"],
                "num_bins": self.dict_options["num_bins"],
                "model_name": self.dict_options["model_name"],
                "audio_backbone": self.dict_options["audio_backbone"],
                "visual_backbone": self.dict_options["visual_backbone"],
                "visual_pretrained": self.dict_options["visual_pretrained"],
                "audio_pretrained": self.dict_options["audio_pretrained"],
                "d_embedding": self.dict_options["d_embedding"],
                "d_feedforward": self.dict_options["d_feedforward"],
                "num_heads": self.dict_options["num_heads"],
                "num_layers": self.dict_options["num_layers"],
                "num_layers_audio": self.dict_options["num_layers_audio"],
                "num_layers_visual": self.dict_options["num_layers_visual"],
                "num_crossaudio_layers": self.dict_options["num_crossaudio_layers"],
                "num_crossvisual_layers": self.dict_options["num_crossvisual_layers"],
                "dropout": self.dict_options["dropout"],
                "activation": self.dict_options["activation"],
                "pretrained_path": self.dict_options["pretrained_path"],
                "image_channel_order": self.dict_options["image_channel_order"], 
                "bidirectional": self.dict_options["bidirectional"],     
            }),
            "name": self.dict_options["name"],
            "root_dir": self.dict_options["root_dir"],
            "checkpoints_dir": self.dict_options["checkpoints_dir"],
            "log_dir": self.dict_options["log_dir"],
            "cuda": self.dict_options["cuda"],
            "num_gpus": self.dict_options["num_gpus"],
            "task": self.dict_options["task"],
            
        })
        
        return train_params

    def _get_test_params(self) -> Params:
        """
        Helper which assembles a Params object for the test process
        """
        
        test_params = Params(dict_params={
            "process": "test", 
            "name": self.dict_options["name"],  # might not be necessary here?
            "prediction_file": self.dict_options["prediction_file"],
            "test": Params(dict_params={
                "dataset_path": self.dict_options["test_dataset_file"],
                "cuda": self.dict_options["cuda"],
                "num_workers": self.dict_options["num_test_workers"],
                "is_training": False,
                "partition": "Test",
                "seq_length": self.dict_options["seq_length"],
                "fps": self.dict_options["fps"],
                "image_size": self.dict_options["image_size"],
                "image_channel_order": self.dict_options["image_channel_order"],
                "audio_sr": self.dict_options["sr"],
                "audio_window_size": self.dict_options["window_size"],
                "metric_name": self.dict_options["metric_name"],
            }),
            
            "cuda": self.dict_options["cuda"],
            "num_gpus": self.dict_options["num_gpus"],
            "base_modality": self.dict_options["base_modality"],
            
            "root_dir": self.dict_options["root_dir"],
            "checkpoints_dir": self.dict_options["checkpoints_dir"],
            "log_dir": self.dict_options["log_dir"],
            "task": self.dict_options["task"],
            "model": Params(dict_params={
                "num_outs": self.dict_options["num_outputs"],
                "num_bins": self.dict_options["num_bins"],
                "model_name": self.dict_options["model_name"],
                "audio_backbone": self.dict_options["audio_backbone"],
                "visual_backbone": self.dict_options["visual_backbone"],
                "visual_pretrained": self.dict_options["visual_pretrained"],
                "audio_pretrained": self.dict_options["audio_pretrained"],
                "d_embedding": self.dict_options["d_embedding"],
                "d_feedforward": self.dict_options["d_feedforward"],
                "num_heads": self.dict_options["num_heads"],
                "num_layers": self.dict_options["num_layers"],
                "num_layers_audio": self.dict_options["num_layers_audio"],
                "num_layers_visual": self.dict_options["num_layers_visual"],
                "num_crossaudio_layers": self.dict_options["num_crossaudio_layers"],
                "num_crossvisual_layers": self.dict_options["num_crossvisual_layers"],
                "dropout": self.dict_options["dropout"],
                "activation": self.dict_options["activation"],
                "image_channel_order": self.dict_options["image_channel_order"],
                "pretrained_path": self.dict_options["pretrained_path"],     
            })
        })
        
        return test_params


class HPOptions():
    """
    Command line options for hyperparameter tuning
    """
    
    def __init__(self) -> None:
        
        self._parser = self._add_parsers()
        self._is_parsed = False
        self._is_initialized = False
        self._is_training = False
        self._process = ""

        self._params = None
        # get the current date
        self._date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        self.dict_options = {}


    def _add_parsers(self) -> argparse.ArgumentParser:
        
         # create the parser
        parser = argparse.ArgumentParser(description="A parser for this cool project. Adds flags")

        # MODEL args
        parser.add_argument("--name", type=str, default="experiment_1", help="Name of the experiment. Controls the saving location")
        parser.add_argument("--model_name", type=str, help="Which model architecture to use", choices=["concatselfattn", 
                                                                                                       "concatrnn", 
                                                                                                       "concatbirnn",
                                                                                                       "mult", 
                                                                                                       "multv2",
                                                                                                       "gatedmultv2",
                                                                                                       "concatlinear",
                                                                                                        "visualselfattn",
                                                                                                        "visualrnn",
                                                                                                        "audioselfattn",
                                                                                                        "audiornn",
                                                                                                        "hybrid",
                                                                                                        "auxattn",
                                                                                                        "auxrnn"])
        parser.add_argument("--visual_backbone", type=str, help="The visual extraction network", 
                            choices=["senet50_ft", "senet50_scratch", "resnet50_ft", "resnet50_scratch", "inceptionresnetv1", "mobilefacenet"])
        parser.add_argument("--visual_pretrained", type=bool, default=True, help="Use pretrained network for visual cnn backbone. Default True")
        parser.add_argument("--audio_backbone", type=str, help="The audio extraction network", choices=["emo16", "emo18", "zhao19"])
        parser.add_argument("--audio_pretrained", type=bool, default=True, 
                            help="Use pretrained network for the audio cnn. Default True. Careful - ensure pretrained weights for audio model are available")
        parser.add_argument("--base_modality", type=str, default="audio", #required=True,
                            help="Modality for the base model", choices=["audio", "visual"])
        parser.add_argument("--num_outputs", type=int, default=2, help="Number of outputs of the model (defaults to 2)")
        parser.add_argument("--num_bins", type=int, default=1, help="number of bins to split continuous output into (default 1 no splitting)")
        #parser.add_argument("--d_embedding", type=int, default=128, help="Size of the embedding in the temporal model. Default 128")
        #parser.add_argument("--num_heads", type=int, default=4, help="Number of heads for Multi Head Attention. Default 4")
        #parser.add_argument("--num_layers", type=int, default=4, help="Number of layers for the temporal model. Default is 4")
        #parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate. Default 0.3")
        
        # DATA args
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size to use. (defaults to 8)")
        parser.add_argument("--seq_length", type=int, default=8, help="Length of sequences to use")
        parser.add_argument("--fps", type=int, default=30, help="Image frame rate per second. Default 30 (dataset frame rate). Changing to <30 allows for dilated sequences on the train set")
        parser.add_argument("--image_size", type=int, default=160, help="Width and height of an image frame")
        parser.add_argument("--image_channel_order", type=str, default="RGB", choices=["RGB", "BGR"], help="Order to load image channels (default RGB)")
        parser.add_argument("--sr", type=int, default=16000, help="Audio sample rate in Hz. (defaults to 16000)")
        parser.add_argument("--window_size", type=float, default=0.66, help="Length of an audio frame in seconds")
        parser.add_argument("--train_dataset_file", type=str, required=True,
                            help="Path to the pkl file that describes the train dataset")
        parser.add_argument("--valid_dataset_file", type=str, required=True,
                            help="Path to the pkl file that describes the validation dataset")
        
        # Dataloader args
        parser.add_argument("--num_train_workers", type=int, default=4,
                            help="Number of workers for loading train data (defaults to 4")
        parser.add_argument("--num_valid_workers", type=int, default=4,
                            help="Number of workers for loading validation data (defaults to 4")
        parser.add_argument("--pin_memory", type=bool, default=False, help="Whether to use pinned memory for the dataloaders. May result in issues! Default False")

        # GPU args
        parser.add_argument("--cuda", type=bool, default=False, help="Use CUDA. (defaults to False")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (defaults to 1)")

        # Multitask use case, specifying multiple tasks is possible
        #parser.add_argument("--tasks", type=str, default=["EXPR", "AU", "VA"], nargs="+")
        #parser.add_argument("--dataset_names", type=str, default=["Mixed_EXPR", "Mixed_AU", "Mixed_VA"], nargs="+")
        # Single Task use case, only one task may be specified
        parser.add_argument("--task", type=str, default="VA", choices=["VA", "EXPR", "AU"], help="Task to process")

        # PATH args
        parser.add_argument("--root_dir", type=str, default="./embed", help="Root folder for the output")
        #parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Models are saved here")
        #parser.add_argument("--log_dir", type=str, default="./logs", help="Logs are saved here")
        
        parser.add_argument("--num_trials", help="Number of trials to run", default=10, type=int)
        
        
        self._is_initialized = True
        
        return parser
        

    def parse(self):
        
        # create parsers if they dont exist
        if not self._is_initialized:
            self._add_parsers()

        args = self._parser.parse_args()

        # turn args object into dict
        self.dict_options = vars(args)
        
        print("Current date: ", self._date)
        
        self._get_params()

        # print out args
        self._print(self.dict_options)
        # save args to json
        self._save(self.dict_options)

        self._is_parsed = True

        return self._params
    
    
    def _get_params(self):
        """
        Helper which constructs params object from args. All other options are drwan from ray config
        """
        
        # check the directories, if necessary, create new run
        root_path = Path(self.dict_options["root_dir"])
        if root_path.exists():
            #root_dir = self.dict_options["root_dir"]
            print("Warning: Root folder {} already exists!".format(str(root_path)))
            print("Creating new subfolder with current date and changing logs and ckpt dirs to that folders subfolders...")
            self.dict_options["root_dir"] = str(root_path / str(self._date))
            # update sub dirs
            self.dict_options["checkpoints_dir"] = str(root_path / str(self._date) / "checkpoints")
            self.dict_options["log_dir"] = str(root_path / str(self._date) / "logs")
        else:
            self.dict_options["checkpoints_dir"] = str(root_path / "checkpoints")
            self.dict_options["log_dir"] = str(root_path / "logs")
        
        
        
        self._params = Params(dict_params={
            "name": self.dict_options["name"],
            "train": Params(dict_params={
                "cuda": self.dict_options["cuda"],
                "base_modality": self.dict_options["base_modality"],
                "is_training": True,
                "partition": "Train",
                "seq_length": self.dict_options["seq_length"],
                "fps": self.dict_options["fps"],
                "image_size": self.dict_options["image_size"],
                "image_channel_order": self.dict_options["image_channel_order"],
                "audio_sr": self.dict_options["sr"],
                "audio_window_size": self.dict_options["window_size"],
                "num_workers": self.dict_options["num_train_workers"],
                "pin_memory": self.dict_options["pin_memory"],  
                "dataset_file": self.dict_options["train_dataset_file"],     
            }),
            "valid": Params(dict_params={
                "dataset_file": self.dict_options["valid_dataset_file"],
                "cuda": self.dict_options["cuda"],
                "base_modality": self.dict_options["base_modality"],
                "is_training": False,
                "partition": "Validation",
                "seq_length": self.dict_options["seq_length"],
                "fps": self.dict_options["fps"],
                "image_size": self.dict_options["image_size"],
                "image_channel_order": self.dict_options["image_channel_order"],
                "audio_sr": self.dict_options["sr"],
                "audio_window_size": self.dict_options["window_size"],
                "num_workers": self.dict_options["num_valid_workers"],
                "pin_memory": self.dict_options["pin_memory"],
                #"dataset_file": self.dict_options["valid_dataset_file"],  
            }),
            "model": Params(dict_params={
                "model_name": self.dict_options["model_name"],
                "visual_backbone": self.dict_options["visual_backbone"],
                "audio_backbone": self.dict_options["audio_backbone"],
                "visual_pretrained": self.dict_options["visual_pretrained"],
                "audio_pretrained": self.dict_options["audio_pretrained"],
                "image_channel_order": self.dict_options["image_channel_order"],
                "num_bins": self.dict_options["num_bins"],
                
            }),
            "cuda": self.dict_options["cuda"],
            "num_gpus": self.dict_options["num_gpus"],
            "base_modality": self.dict_options["base_modality"],
            "root_dir": self.dict_options["root_dir"],
            "checkpoints_dir": self.dict_options["checkpoints_dir"],
            "log_dir": self.dict_options["log_dir"],
            "task": self.dict_options["task"],
            "num_trials": self.dict_options["num_trials"]
            
        })
        


    def _print(self, args:Dict):
        print("-" * 20 + " Options " + "-" * 20)
        for k, v in sorted(args.items()):
            print("{} : {}".format(str(k), str(v)))
        print("-" * 49)


    def _save(self, args:Dict):
        """
        Helper function that saves the commandline arguments to a file in JSON format
        :param args: A dictionary created from the commandline args
        :return:
        """

        #experiment_dir = Path(self._params.checkpoints_dir) / self._params.name
        experiment_dir = Path(self._params.checkpoints_dir)  / "options"    
        #print(experiment_dir)
        #if self._is_training and not experiment_dir.exists():
        if not experiment_dir.exists():
            experiment_dir.mkdir(parents=True, exist_ok=True)
        else:
            assert experiment_dir.exists(), "Experiment dir {} does not exist".format(experiment_dir)

        # save as json file
        file_name = experiment_dir / "options_{}.json".format(self._process)
        with open(file_name, "w") as f:
            json.dump(args, f, indent=6)
            


class TestOptions(Options):
    """
    A helper class that handles command line args for testing N models.
    """
    
    def __init__(self) -> None:
        super().__init__()

    def parse(self):
        # create parsers if they dont exist
        if not self._is_initialized:
            self._add_parsers()

        args = self._parser.parse_args()

        # turn args object into dict
        self.dict_options = vars(args)
        
        print("Current date: ", self._date)
        
        self._get_params()

        # print out args
        self._print(self.dict_options)
        # save args to json
        #self._save(self.dict_options)

        self._is_parsed = True

        return self._params
    
    def _add_parsers(self):
        
        parser = argparse.ArgumentParser(description="Test N models and write out their predictions")
        
        # DATA args
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size to use. (defaults to 8)")
        parser.add_argument("--seq_length", type=int, default=16, help="Length of sequences to use")
        parser.add_argument("--fps", type=int, default=30, help="Image frame rate per second. Default 30 (dataset frame rate). Changing to <30 allows for dilated sequences on the train set")
        parser.add_argument("--image_size", type=int, default=160, help="Width and height of an image frame")
        parser.add_argument("--image_channel_order", type=str, default="RGB", choices=["RGB", "BGR"], help="Order to load image channels (default RGB)")
        parser.add_argument("--sr", type=int, default=16000, help="Audio sample rate in Hz. (defaults to 16000)")
        parser.add_argument("--window_size", type=float, default=0.66, help="Length of an audio frame in seconds")
        parser.add_argument("--test_dataset_file", type=str, required=True,
                            help="Path to the pkl file that describes the test dataset", default="/data/eihw-gpu5/karasvin/data_preprocessing/ABAW3_Affwild2/annotations/VA_test_files.pkl")
        # Dataloader args
        parser.add_argument("--num_test_workers", type=int, default=4,
                            help="Number of workers for loading test data (defaults to 4")
        parser.add_argument("--pin_memory", type=bool, default=False, help="Whether to use pinned memory for the dataloaders. May result in issues! Default False")

        # GPU args
        parser.add_argument("--cuda", type=bool, default=False, help="Use CUDA. (defaults to False")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (defaults to 1)")

        # Multitask use case, specifying multiple tasks is possible
        #parser.add_argument("--tasks", type=str, default=["EXPR", "AU", "VA"], nargs="+")
        #parser.add_argument("--dataset_names", type=str, default=["Mixed_EXPR", "Mixed_AU", "Mixed_VA"], nargs="+")
        # Single Task use case, only one task may be specified
        parser.add_argument("--task", type=str, default="VA", choices=["VA", "EXPR", "AU"], help="Task to process")

        # PATH args
        parser.add_argument("--root_dir", type=str, default="./embed", help="Root folder for the output")
        
        parser.add_argument("--checkpoints", type=str, default=["./checkpoints"], nargs="+", 
                            help="A list of model checkpoints. Should also have info to recreate the model and its data loader configuration")
        #parser.add_argument("--log_dir", type=str, default="./logs", help="Logs are saved here")
        parser.add_argument("--merge", type=str, choices=["mean", "centering"], default="mean", help="how to combine model predictions. Default is Mean")
        
        self._is_initialized = True
        
        self._parser = parser
    
    def _get_params(self):
        
        # check the directories, if necessary, create new run
        root_path = Path(self.dict_options["root_dir"])
        if root_path.exists():
            #root_dir = self.dict_options["root_dir"]
            print("Warning: Root folder {} already exists!".format(str(root_path)))
            print("Creating new subfolder with current date and changing logs and ckpt dirs to that folders subfolders...")
            self.dict_options["root_dir"] = str(root_path / str(self._date))
            
        self._params = Params(dict_params={
            "test": Params(dict_params={    # fallback options
                "dataset_file": self.dict_options["test_dataset_file"],
                "cuda": self.dict_options["cuda"],
                "is_training": False,
                "partition": "Test",
                "seq_length": self.dict_options["seq_length"],
                "fps": self.dict_options["fps"],
                "image_size": self.dict_options["image_size"],
                "image_channel_order": self.dict_options["image_channel_order"],
                "audio_sr": self.dict_options["sr"],
                "audio_window_size": self.dict_options["window_size"],
                "num_workers": self.dict_options["num_test_workers"],
                "pin_memory": self.dict_options["pin_memory"],
                
            }),
            "root_dir": self.dict_options["root_dir"],
            "checkpoints": self.dict_options["checkpoints"],
            "cuda": self.dict_options["cuda"],
            "merge": self.dict_options["merge"]
        })
    
        
def get_testparams(dict_options:dict) -> Params:
    """
    Helper which recreates a Params object from a dict
    Args: options A dict 
    """
    
    # catch missing params
    test_data_file = "/data/eihw-gpu5/karasvin/databases/ABAW3_Affwild2/annotations/VA_test_files.pkl"
    
    if not "d_feedforward" in dict_options.keys():
        if "d_ff" in dict_options.keys():
            dict_options.update({"d_feedforward": dict_options["d_ff"]})
        else:
            dict_options.update({"d_feedforward": 64})
    
    if not "bidirectional" in dict_options.keys(): # might be missing for older rnn models
        dict_options.update({"bidirectional": True})
    if not "num_crossaudio_layers" in dict_options.keys(): # cm transformer keys
        if "num_ca_layers" in dict_options.keys():
            dict_options.update({"num_crossaudio_layers": dict_options["num_ca_layers"]})
        else: 
            dict_options.update({"num_crossaudio_layers": 4})
    if not "num_crossvisual_layers" in dict_options.keys():
        if "num_cv_layers" in dict_options.keys():
            dict_options.update({"num_crossvisual_layers": dict_options["num_cv_layers"]})
        else: 
            dict_options.update({"num_crossvisual_layers": 4})
    if not "num_layers_audio" in dict_options.keys(): 
        dict_options.update({"num_layers_audio": 4})
    if not "num_heads" in dict_options.keys():
        dict_options.update({"num_heads": 8})
    # extra options
    if not "num_layers_visual" in dict_options.keys(): 
        dict_options.update({"num_layers_visual": 4})        
    if not "activation" in dict_options.keys():
        dict_options.update({"activation": "gelu"}) # default value
    if not "root_dir" in dict_options.keys():
        dict_options.update({"root_dir": "./"})      
    if not "checkpoints_dir" in dict_options.keys():
        dict_options.update({"checkpoints_dir": "./checkpoints"})
    if not "pretrained_path" in dict_options.keys():
        dict_options.update({"pretrained_path": None})
    if not "name" in dict_options.keys():
        dict_options.update({"trained_model"})
    
    
    params = Params(dict_params={
        "name": dict_options["name"],
        "process": "test",
        "test": Params(dict_params={    # fallback options
                "dataset_file": test_data_file, #hardcoded
                "batch_size": 32, # hardcoded
                "cuda": dict_options["cuda"],
                "is_training": False,
                "partition": "Test",
                "seq_length": dict_options["seq_length"],
                "fps": dict_options["fps"],
                "image_size":dict_options["image_size"],
                "image_channel_order": dict_options["image_channel_order"],
                "audio_sr": dict_options["sr"],
                "audio_window_size": dict_options["window_size"],
                "num_workers": 16, # hardcoded,
                "pin_memory": dict_options["pin_memory"],
        }),
        "model": Params(dict_params={
             "num_outs": dict_options["num_outputs"],
                "num_bins": dict_options["num_bins"],
                "model_name": dict_options["model_name"],
                "audio_backbone": dict_options["audio_backbone"],
                "visual_backbone": dict_options["visual_backbone"],
                "visual_pretrained": dict_options["visual_pretrained"],
                "audio_pretrained": dict_options["audio_pretrained"],
                "d_embedding": dict_options["d_embedding"],
                "d_feedforward": dict_options["d_feedforward"],
                "num_heads": dict_options["num_heads"],
                "num_layers": dict_options["num_layers"],
                "num_layers_audio": dict_options["num_layers_audio"],
                "num_layers_visual": dict_options["num_layers_visual"],
                "num_crossaudio_layers": dict_options["num_crossaudio_layers"],
                "num_crossvisual_layers": dict_options["num_crossvisual_layers"],
                "dropout": dict_options["dropout"],
                "activation": dict_options["activation"],
                "pretrained_path": dict_options["pretrained_path"],
                "image_channel_order": dict_options["image_channel_order"], 
                "bidirectional": dict_options["bidirectional"],     
            
        }),
        "root_dir": dict_options["root_dir"],
        "checkpoints_dir": dict_options["checkpoints_dir"],
        "cuda": dict_options["cuda"],
    })
    
    return params