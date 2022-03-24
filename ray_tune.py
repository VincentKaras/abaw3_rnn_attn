"""
HP search using ray tune
"""

from cli.parsers import HPOptions
from end2you.utils import Params

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from training.hp_search import mytrainable

from functools import partial
import pandas as pd
import pickle
from pathlib import Path
import os

from data.dataloader import get_dataset
from utils.transforms import train_transforms, test_transforms

def get_config(params:Params) -> dict:
    """
    Helper which turns the Params object into a config dict
    """
    
    config = {
        "optimizer": "adamw",
        "num_layers": tune.randint(1, 5),
        "d_embedding": tune.choice([32, 64, 128]),
        "lr": tune.loguniform(1e-5, 1e-2),
        "lr_policy": "cosine",
        "step_size": tune.choice([5, 10]),
        "T_max": tune.choice([100, 200]),
        "weight_decay": tune.loguniform(1e-3, 1e-1),
        #"batch_size": tune.choice([32, 64]),
        "batch_size": 16,   # limit for unfrozen model to avoid OOM
        "activation": tune.choice(["gelu", "selu"]),
        "dropout": tune.quniform(0.3, 0.6, 0.1),
        "num_bins": params.model.num_bins,
        "lambda_mse": tune.uniform(0.1, 1.0),
        "lambda_dis": tune.uniform(0.1, 1.0)
        
    }
    
    if params.model.model_name in ["visualselfattn", "audioselfattn", "concatselfattn", "mult", "multv2", "gatedmultv2", "auxattn", "hybridattn"]:
        config.update({
            #"d_ff": tune.sample_from(lambda spec: spec.config.d_embedding * 2),  # feedforward twice as big as embedding
            "d_ff": tune.choice([64, 128, 256]),
            "num_heads": tune.choice([4, 8]),
        })
    
    """
    elif params.model == "concatrnn":  
        config.update({
            "bidirectional": False
        })
    elif params.model == "concatbirnn":
        config.update({
            "bidirectional": True
        })
    """    
    # recurrent models
    if "rnn" in params.model.model_name:
        config.update({
            "bidirectional": tune.choice([True, False])
        })
    
    if params.model.model_name in ["mult", "multv2", "gatedmultv2", "auxattn", "hybridattn"]:
        config.update({
            "num_cv_layers": tune.randint(1, 5),
            "num_ca_layers": tune.randint(1, 5)
        })
        
    if params.model.model_name in ["auxattn", "auxrnn"]:
        config.update({
            "num_layers_visual": tune.randint(1, 4),
            "num_layers_audio": tune.randint(1, 4)
        })
    
    return config


def load_data(params:Params):

    pass


def main(config:dict, params:Params, num_samples=20, max_num_epochs=10, gpus_per_trial=0.5, cpus_per_trial=20):
    
    local_dir = params.root_dir
    exp_name = params.name
    
    print("Creating ray scheduler ...")
    scheduler = ASHAScheduler(max_t=max_num_epochs, 
                             grace_period=2, 
                             reduction_factor=2)
    
    # create dataset here and pass them to every trial
    print("Creating datasets ...")
    train_dataset = get_dataset(params.train, train_mode="Train", task=params.task)
    
    val_dataset = get_dataset(params.valid, train_mode="Validation", task=params.task)
    
    print("Starting tune ...")
    result = tune.run(
       tune.with_parameters(mytrainable, 
                            params=params,
                            train_ds=train_dataset,
                            val_ds=val_dataset),
       config=config,
       resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
       metric="ccc",
       mode="max",
       scheduler=scheduler,
       num_samples=num_samples,
       name=exp_name,
       local_dir=local_dir,
       max_concurrent_trials=2,
       progress_reporter=tune.CLIReporter(max_report_frequency=30)    # report every 30 s
    )
    
    # results
    print("*" * 40)
    
    best_trial = result.get_best_trial("ccc", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    best_result = result.best_result
    try:
        #print("Best trial validation loss: {:.6f}".format(best_trial.evaluated_params["val_loss"]))
        print("Best trial validation metric: {:.3f}, valence: {:.3f}, arousal: {:.3f}".format(best_trial.evaluated_params["ccc"], 
                                                                                            best_trial.evaluated_params["ccc_valence"],
                                                                                            best_trial.evaluated_params["ccc_arousal"]))
    except KeyError:
        print("Could not find requested keys in best trial's evaluated params. Printing the best result dict instead ...")
        
        for key, item in best_result.items():
            print(key, item)
        
        
        
    # get best log dir
    logdir = result.get_best_logdir("ccc", "max", "all")
    print("Best log dir: {}".format(logdir))
    
    # save results dataframe to log dir
    df_results = result.results_df
    ld = Path(params.log_dir)
    if not ld.exists():
        ld.mkdir(parents=True, exist_ok=True)
    out_file = ld / "df_results.csv"
    print("Writing results dataframe to {}".format(out_file))
    #with open(str(out_file), "wb") as f:
    #    
    #    pickle.dump(df_results)
    df_results.to_csv(str(out_file))
    
    

if __name__ == "__main__":
    options = HPOptions()
    
    params = options.parse()
    
    config = get_config(params)
    
    num_samples = params.num_trials
    
    nodename = os.getenv("SLURMD_NODENAME")
    if nodename:
        if nodename == "eihw-gpu5":
            cpus_per_trial = 32
            gpus_per_trial = 1
            print("running on eihw-gpu5.")
            main(config=config, params=params, num_samples=num_samples, cpus_per_trial=cpus_per_trial, gpus_per_trial=gpus_per_trial)
        elif nodename == "eihw-gpu7":
            print("Running on eihw-gpu7")
            cpus_per_trial = 30
            gpus_per_trial = 1
            main(config=config, params=params, num_samples=num_samples, cpus_per_trial=cpus_per_trial, gpus_per_trial=gpus_per_trial)
    else:   # standard params
        main(config, params, num_samples=num_samples)
        
    
   
    
   
    
    
