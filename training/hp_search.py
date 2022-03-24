"""
Pytorch Hyperparameter tuning
"""


import ray
from ray import tune
#from ray.train import Trainer

import torch 
import torch.nn as nn
import numpy as np
from models.modules import hybrid

from models.modules.audiovisual import AudioVisualAttentionNetwork, AudioVisualRecurrentModel
from models.modules.audio_model import AudioAttentionNetwork, AudioRecurrentNetwork
from models.modules.visual_model import VisualAttentionNetwork, VisualRecurrentNetwork
from models.modules.hybrid import AuxAttentionNetwork, AuxRecurrentNetwork

from utils.optimizer import get_optimizer, get_scheduler
from utils.losses_metrics import CCCLoss, CCC_score, MSELoss, CustomCrossEntropyLoss
from data.dataloader import get_dataloader
from torch.utils.data import DataLoader
from utils.losses_metrics import VA_metric
from utils.transforms import discretize_labels
from training import CLASS_WEIGHTS


from end2you.utils import Params

from pathlib import Path

def mytrainable(config, params:Params=None, train_ds=None, val_ds=None, checkpoint_dir=None):
    
    # create dataloaders 
    if train_ds:
        train_dataloader = DataLoader(
            train_ds, 
            shuffle=True,
            batch_size=config["batch_size"],
            drop_last=True,
            pin_memory=params.train.pin_memory,
            num_workers=params.train.num_workers
        )
    else:
        train_dataloader = get_dataloader(train_mode="Train", params=params.train, task=params.task)
        
    if val_ds: 
        val_dataloader = DataLoader(
            val_ds,
            shuffle=False,
            batch_size=config["batch_size"],
            pin_memory=params.valid.pin_memory,
            drop_last=False,
            num_workers=params.valid.num_workers
        )
    else: 
        val_dataloader = get_dataloader(train_mode="Validation", params=params.valid, task=params.task)
    
    
    
    
    # create a model
    model_name = params.model.model_name
    if model_name == "concatselfattn":
        net = AudioVisualAttentionNetwork(
            audio_input_size=int(params.train.audio_sr * params.train.audio_window_size),
            visual_input_size=params.train.image_size,
            audio_backbone=params.model.audio_backbone,
            visual_backbone=params.model.visual_backbone,
            audio_pretrained=params.model.audio_pretrained,
            visual_pretrained=params.model.visual_pretrained,
            audio_pool=True,
            temporal_model="selfattn",
            batch_first=True,
            d_embedding=config["d_embedding"],
            d_feedforward=config["d_ff"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            output_names=["VA"],
            num_bins=params.model.num_bins,
            activation=config["activation"],
        )
    elif model_name == "mult":
        net = AudioVisualAttentionNetwork(
            audio_input_size=int(params.train.audio_sr * params.train.audio_window_size),
            visual_input_size=params.train.image_size,
            audio_backbone=params.model.audio_backbone,
            visual_backbone=params.model.visual_backbone,
            audio_pretrained=params.model.audio_pretrained,
            visual_pretrained=params.model.visual_pretrained,
            audio_pool=True,
            temporal_model="mult",
            batch_first=True,
            d_embedding=config["d_embedding"],
            d_feedforward=config["d_ff"],
            num_layers=config["num_layers"],
            num_crossaudio_layers=config["num_ca_layers"],
            num_crossvisual_layers=config["num_cv_layers"],
            num_heads=config["num_heads"],
            output_names=["VA"],
            num_bins=params.model.num_bins,
            activation=config["activation"]
        )
    elif model_name == "multv2":
        net = AudioVisualAttentionNetwork(
            audio_input_size=int(params.train.audio_sr * params.train.audio_window_size),
            visual_input_size=params.train.image_size,
            audio_backbone=params.model.audio_backbone,
            visual_backbone=params.model.visual_backbone,
            audio_pretrained=params.model.audio_pretrained,
            visual_pretrained=params.model.visual_pretrained,
            audio_pool=True,
            temporal_model="multv2",
            batch_first=True,
            d_embedding=config["d_embedding"],
            d_feedforward=config["d_ff"],
            num_layers=config["num_layers"],
            num_crossaudio_layers=config["num_ca_layers"],
            num_crossvisual_layers=config["num_cv_layers"],
            num_heads=config["num_heads"],
            output_names=["VA"],
            num_bins=params.model.num_bins,
            activation=config["activation"]
        )
    
    # gated multv2
    elif model_name == "gatedmultv2":
        net = AudioVisualAttentionNetwork(
            audio_input_size=int(params.train.audio_sr * params.train.audio_window_size),
            visual_input_size=params.train.image_size,
            audio_backbone=params.model.audio_backbone,
            visual_backbone=params.model.visual_backbone,
            audio_pretrained=params.model.audio_pretrained,
            visual_pretrained=params.model.visual_pretrained,
            audio_pool=True,
            temporal_model="gatedmultv2",
            batch_first=True,
            d_embedding=config["d_embedding"],
            d_feedforward=config["d_ff"],
            num_layers=config["num_layers"],
            num_crossaudio_layers=config["num_ca_layers"],
            num_crossvisual_layers=config["num_cv_layers"],
            num_heads=config["num_heads"],
            output_names=["VA"],
            num_bins=params.model.num_bins,
            activation=config["activation"]
        )
    
    
    
    # recurrent
    elif model_name == "concatbirnn":
        net = AudioVisualRecurrentModel(
            audio_input_size=int(params.train.audio_sr * params.train.audio_window_size),
            visual_input_size=params.train.image_size,
            audio_backbone=params.model.audio_backbone,
            visual_backbone=params.model.visual_backbone,
            audio_pretrained=params.model.audio_pretrained,
            visual_pretrained=params.model.visual_pretrained,
            temporal_model="rnn",
            bidirectional=config["bidirectional"],
            audio_pool=True,
            batch_first=True,
            d_embedding=config["d_embedding"],
            num_layers=config["num_layers"],
            output_names=["VA"],
            activation=config["activation"],
        )
    elif model_name == "concatrnn":
        net = AudioVisualRecurrentModel(
            audio_input_size=int(params.train.audio_sr * params.train.audio_window_size),
            visual_input_size=params.train.image_size,
            audio_backbone=params.model.audio_backbone,
            visual_backbone=params.model.visual_backbone,
            audio_pretrained=params.model.audio_pretrained,
            visual_pretrained=params.model.visual_pretrained,
            temporal_model="rnn",
            audio_pool=True,
            batch_first=True,
            d_embedding=config["d_embedding"],
            num_layers=config["num_layers"],
            output_names=["VA"],
            activation=config["activation"],
        )
        
    ########################################################################
    ######################ä########## Unimodal #############################
    ########################################################################
    elif model_name == "visualselfattn":
        net = VisualAttentionNetwork(
            visual_input_size=params.train.image_size,
            visual_encoder=params.model.visual_backbone,
            pretrained=params.model.visual_pretrained,
            s2s_model="selfattn",
            batch_first=True,
            d_embedding=config["d_embedding"],
            d_ff=config["d_ff"],
            n_layers=config["num_layers"],
            n_heads=config["num_heads"],
            output_names=["VA"],
            activation=config["activation"],
        )
    
          
    elif model_name == "visualrnn":
        net = VisualRecurrentNetwork(
            visual_input_size=params.train.image_size,
            visual_encoder=params.model.visual_backbone,
            pretrained=params.model.visual_pretrained,
            s2s_model="rnn",
            batch_first=True,
            d_embedding=config["d_embedding"],
            n_layers=config["num_layers"],
            output_names=["VA"],
            activation=config["activation"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
        )
        
        
    elif model_name == "audioselfattn":
        net = AudioAttentionNetwork(
            audio_input_size=int(params.train.audio_sr * params.train.audio_window_size),
            audio_encoder=params.model.audio_backbone,
            pretrained=params.model.audio_pretrained,
            s2s_model="selfattn",
            batch_first=True,
            d_embedding=config["d_embedding"],
            d_ff=config["d_ff"],
            n_layers=config["num_layers"],
            n_heads=config["num_heads"],
            output_names=["VA"],
            activation=config["activation"],
            dropout=config["dropout"],
        )
    elif model_name == "audiornn":
        net = AudioRecurrentNetwork(
            audio_input_size=int(params.train.audio_sr * params.train.audio_window_size),
            audio_encoder=params.model.audio_backbone,
            pretrained=params.model.audio_pretrained,
            s2s_model="rnn",
            batch_first=True,
            d_embedding=config["d_embedding"],
            n_layers=config["num_layers"],
            output_names=["VA"],
            activation=config["activation"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
        )
        
        
    ###########################################
    ########### Aux Outputs ###################
    ###########################################
    
    elif model_name == "auxattn":
        
        net = AuxAttentionNetwork(
            visual_input_size=params.train.image_size,
            audio_input_size=int(params.train.audio_sr * params.train.audio_window_size),
            audio_backbone=params.model.audio_backbone,
            visual_backbone=params.model.visual_backbone,
            audio_pretrained=params.model.audio_pretrained,
            visual_pretrained=params.model.visual_pretrained,
            temporal_model="selfattn",
            d_embedding=config["d_embedding"],
            d_feedforward=config["d_ff"],
            num_layers=config["num_layers"],
            num_layers_ca=config["num_ca_layers"],
            num_layers_cv=config["num_cv_layers"],
            num_layers_audio=config["num_layers_audio"],
            num_layers_visual=config["num_layers_visual"],
            num_heads=config["num_heads"],
            output_names=["VA"],
            activation=config["activation"],
            dropout=config["dropout"])
        
    elif model_name == "auxrnn":
        net = AuxRecurrentNetwork(
            visual_input_size=params.train.image_size,
            audio_input_size=int(params.train.audio_sr * params.train.audio_window_size),
            audio_backbone=params.model.audio_backbone,
            visual_backbone=params.model.visual_backbone,
            audio_pretrained=params.model.audio_pretrained,
            visual_pretrained=params.model.visual_pretrained,
            temporal_model="rnn",
            d_embedding=config["d_embedding"],
            num_layers=config["num_layers"],
            num_layers_audio=config["num_layers_audio"],
            num_layers_visual=config["num_layers_visual"],
            output_names=["VA"],
            activation=config["activation"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
            
            
        )
        
    else:
        raise NotImplementedError("{} not implemented!".format(params.model.model_name))
        
    # move to gpu
    if torch.cuda.is_available():
        net.cuda()
    # freeze feature extractors
    #net.freeze_extractors()
    #print("Trainable parameters after freezing: {}".format(sum([p.numel() for p in net.parameters() if p.requires_grad])))
    net.unfreeze_extractors()
    print("Trainable parameters unfrozen: {}".format(sum([p.numel() for p in net.parameters()])))
        
    opt = get_optimizer(model=net, 
                        optimizer=config["optimizer"],
                        lr=config["lr"],
                        wd=config["weight_decay"])
    
    lrsched = get_scheduler(
        lr_scheduler=config["lr_policy"],
        optimizer=opt,
        step_size=config["step_size"],
        T_max=config["T_max"]
    )
    
    loss = CCCLoss(num_bins=config["num_bins"])
    mseloss = MSELoss(num_classes=1, reduction="mean")
    disloss = CustomCrossEntropyLoss(num_classes=24, weight=CLASS_WEIGHTS)    # always 24
    if torch.cuda.is_available():
        loss.cuda()
        mseloss.cuda()
        disloss.cuda()
    
    # used to restore checkpoints
    if checkpoint_dir:
        ckpt = Path(checkpoint_dir) / "checkpoint"
        
        state = torch.load(str(ckpt))   # dict
        net.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer_state"])
    
    is_blocking = params.train.pin_memory
    
    #########################
    # WEIGHT for valence and arousal
    w_valence = 0.75
    w_arousal = 0.25
    
    
    # run
    for epoch in range(1, 11):
        
        net.train()
        
        train_loss = 0.0
        train_valence_loss = 0.0
        train_arousal_loss = 0.0
        train_mse_loss = 0.0
        train_discrete_loss = 0.0
        
        num_batches = len(train_dataloader)
        for i, batch in enumerate(train_dataloader):
            
             # data preparation
            labels = batch["label"].cuda()
            qlabels = discretize_labels(labels).cuda()
            image = batch["image"].cuda()
            audio = batch["audio"].cuda()
            
            opt.zero_grad()
            
            predictions = net([audio, image])
            
            def compute_losses(pred_va, pred_cat, labels_va, labels_cat):
                
                loss_valence = loss(pred_va[..., 0], labels_va[..., 0])
                loss_arousal = loss(pred_cat[..., 1], labels_va[..., 1])
                mse_loss_valence = mseloss(pred_va[..., 0], labels_va[..., 0])
                mse_loss_arousal = mseloss(pred_va[..., 1], labels_va[..., 1])
                discrete_loss = disloss(pred_cat, labels_cat)
                
                return loss_valence, loss_arousal, mse_loss_valence, mse_loss_arousal, discrete_loss
            
            
            # compute losses
            if not model_name in ["auxattn", "auxrnn", "hybridattn", "hybridrnn"]:
                
                pred_va = predictions["VA"]
                pred_cat = predictions["discretized"]
                
                loss_valence, loss_arousal, mse_loss_valence, mse_loss_arousal, discrete_loss = compute_losses(pred_va=pred_va,
                                                                                                               pred_cat=pred_cat,
                                                                                                               labels_va=labels, 
                                                                                                               labels_cat=qlabels)
                
            else:
                
                # add up the losses of each branch
                loss_valence, loss_arousal, mse_loss_valence, mse_loss_arousal, discrete_loss = 0.0, 0.0, 0.0, 0.0, 0.0
                for branch in ["audio", "visual", "audiovisual"]:
                    
                    pred_va = predictions["VA_" + branch]
                    pred_cat = predictions["CAT_" + branch]
                    
                    ccc_v, ccc_a, mse_v, mse_a, dis = compute_losses(pred_va=pred_va, pred_cat=pred_cat, labels_va=labels, labels_cat=qlabels)
                    loss_valence += ccc_v
                    loss_arousal += ccc_a
                    mse_loss_valence += mse_v
                    mse_loss_arousal += mse_a
                    discrete_loss += dis
                
            
            #loss_valence = loss([..., 0], labels[..., 0])
            #loss_arousal = loss(predictions["VA"][..., 1], labels[..., 1])
            
            #mse_loss_valence = mseloss(predictions["VA"][..., 0], labels[..., 0])
            #mse_loss_arousal = mseloss(predictions["VA"][..., 1], labels[..., 1])
            
            #discrete_loss = disloss(predictions["discretized"], qlabels)
            
            # mean ccc and mse loss
            cl = w_valence * loss_valence + w_arousal * loss_arousal
            mse = 0.5 * mse_loss_valence + 0.5 * mse_loss_arousal
            #total = 0.5 * cl + 0.5 * mse
            total = 0.5 * cl + config["lambda_mse"] * mse + config["lambda_dis"] * discrete_loss
           
            # backward pass
            total.backward()
            opt.step()
            if config["lr_policy"] == "cosine":
                #lrsched.step(epoch + i / num_batches)
                lrsched.step()
            
            # add running losses
            train_loss = train_loss + total.item()
            train_valence_loss = train_valence_loss + loss_valence.item()
            train_arousal_loss = train_arousal_loss + loss_arousal.item()
            train_mse_loss += mse.item()
            train_discrete_loss += discrete_loss.item()
            
        # average loss over batches 
        train_loss /= num_batches
        train_valence_loss /= num_batches
        train_arousal_loss /= num_batches
        train_mse_loss /= num_batches
        train_discrete_loss /= num_batches
            
            
        batch_preds = []
        batch_targets = []
        
        val_loss = 0.0
        val_valence_loss = 0.0
        val_arousal_loss = 0.0
        val_mse_loss = 0.0
        val_discrete_loss = 0.0
        
        with torch.no_grad():
            
            net.eval()
            
            num_val_batches = len(val_dataloader)
            for i, batch in enumerate(val_dataloader):
                # data preparation
                labels = batch["label"].cuda(non_blocking=is_blocking)
                qlabels = discretize_labels(labels).cuda(non_blocking=is_blocking)
                batch_targets.append(batch["label"].numpy())    # makes use of non-blocking if possible
                image = batch["image"].cuda(non_blocking=is_blocking)
                audio = batch["audio"].cuda(non_blocking=is_blocking)
            
                predictions = net([audio, image])
            
                # compute losses
                #loss_valence = loss(predictions["VA"][..., 0], labels[..., 0])
                #loss_arousal = loss(predictions["VA"][..., 1], labels[..., 1]) 
                
                #mse_loss_valence = mseloss(predictions["VA"][..., 0], labels[..., 0])
                #mse_loss_arousal = mseloss(predictions["VA"][..., 1], labels[..., 1])
                
                #discrete_loss = disloss(predictions["discretized"], qlabels)
                
                 # compute losses
                if not model_name in ["auxattn", "auxrnn", "hybridattn", "hybridrnn"]:
                    # only one branch
                    pred_va = predictions["VA"]
                    pred_cat = predictions["discretized"]
                    
                    loss_valence, loss_arousal, mse_loss_valence, mse_loss_arousal, discrete_loss = compute_losses(pred_va=pred_va,
                                                                                                                pred_cat=pred_cat,
                                                                                                                labels_va=labels, 
                                                                                                                labels_cat=qlabels)
                    
                else:
                    
                    # add up the losses of each branch
                    loss_valence, loss_arousal, mse_loss_valence, mse_loss_arousal, discrete_loss = 0.0, 0.0, 0.0, 0.0, 0.0
                    for branch in ["audio", "visual", "audiovisual"]:
                        
                        pred_va = predictions["VA_" + branch]
                        pred_cat = predictions["CAT_" + branch]
                        
                        ccc_v, ccc_a, mse_v, mse_a, dis = compute_losses(pred_va=pred_va, pred_cat=pred_cat, labels_va=labels, labels_cat=qlabels)
                        loss_valence += ccc_v
                        loss_arousal += ccc_a
                        mse_loss_valence += mse_v
                        mse_loss_arousal += mse_a
                        discrete_loss += dis
                
                # mean loss
                cl = w_valence * loss_valence + w_arousal * loss_arousal
                mse = 0.5 * mse_loss_valence + 0.5 * mse_loss_arousal
                #total = 0.5 * cl + 0.5 * mse
                total = 0.5 * cl + config["lambda_mse"] * mse + config["lambda_dis"] * discrete_loss
                
                val_loss += total.item()
                val_valence_loss += loss_valence.item()
                val_arousal_loss += loss_arousal.item()
                val_mse_loss += mse.item()
                val_discrete_loss += discrete_loss.item()
                
                key = "VA" if not model_name in ["auxattn", "auxrnn", "hybridattn", "hybridrnn"] else "VA_audiovisual"
                
                batch_preds.append(predictions[key].cpu().numpy()) # store batch predictions
            
            # average losses
            
            val_loss /= num_val_batches
            val_valence_loss /= num_val_batches
            val_arousal_loss /= num_val_batches
            val_mse_loss /= num_val_batches
            val_discrete_loss /= num_val_batches
            
                
            # compute metrics
            pred = np.concatenate(batch_preds, axis=0)
            target = np.concatenate(batch_targets, axis=0)
            
            (ccc_valence, ccc_arousal), ccc = VA_metric(pred, target)
        
            mean_ccc =  w_valence * ccc_valence + w_arousal * ccc_arousal #(ccc_valence + ccc_arousal) / 2.0
        
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = Path(checkpoint_dir) / "checkpoint"
            
            state = {
                "model_state": net.state_dict(),
                "optimizer_state": opt.state_dict()
            }
            torch.save(state, str(path))
                
        # report the results to tune
        
        #if params.model.model_name in ["auxattn", "auxrnn"]:
        #    tune.report()
            
            
        #else:
        tune.report(ccc=mean_ccc,
                    ccc_valence=ccc_valence,
                    ccc_arousal=ccc_arousal,
                    train_loss=train_loss,
                    train_valence_loss=train_valence_loss,
                    train_arousal_loss=train_arousal_loss,
                    train_mse_loss=train_mse_loss,
                    train_discrete_loss=train_discrete_loss,
                    val_loss=val_loss,
                    val_arousal_loss=val_arousal_loss,
                    val_valence_loss=val_valence_loss,
                    val_mse_loss=val_mse_loss,
                    val_discrete_loss=val_discrete_loss)
        
    # end for loop epochs
    
    # does this help memory usage?
    del net
    del train_dataloader
    del val_dataloader
        
    print("Finished training")
    

class MyTrainable():
    pass
