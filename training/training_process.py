"""
Creates all the components based on Params to launch a train session

"""
from pathlib import Path
from re import M

from end2you.utils import Params
from training.trainer import SimpleTrainer, SimpleTrainerFinetune

from data.dataloader import get_dataloader
from models.model_factory import ModelFactory
from utils.losses_metrics import CCC_score, CCCLoss, VA_metric, ComboLoss, MSELoss, CustomCrossEntropyLoss
from utils.optimizer import get_optimizer, get_scheduler
from utils.transforms import discretize_labels
from torch.utils.tensorboard import SummaryWriter
import torch


class TrainingProcess():
    """
    holds objects for Trainer
    """
    
    def __init__(self, params:Params) -> None:
        
        
        #train_dl = get_dataloader(train_params, train_mode="Train", task="VA", tiny=True)
        #val_dl = get_dataloader(val_params, train_mode="Validation", task="VA", tiny=True)
        
        
        # see if there is a pre-trained path
        if params.model.pretrained_path:
            print("A pre-trained model checkpoint was passed!")
            ckpt = torch.load(str(params.model.pretrained_path))
            if "params" in ckpt.keys():
                print("Resetting the parameters for dataloaders to the saved state")
                saved_params = ckpt["params"]
                model_args = saved_params #instantiate from saved params
                train_params = saved_params.train
                val_params = saved_params.valid
            else:
                print("No saved params in checkpoint! Falling back to CLI args. Make sure they are correct!")
                train_params = params.train # use the cli params to instantiate dataloaders
                val_params = params.valid
                model_args = params
        else:
            model_args = params    # use the cli args to instantiate the model 
            train_params = params.train # use the cli params to instantiate dataloaders
            val_params = params.valid
        
        
        
        train_dl = get_dataloader(train_params, train_mode="Train", task="VA", tiny=False)
        val_dl = get_dataloader(val_params, train_mode="Validation", task="VA", tiny=False)       
        
        # model
        model_wrapper = ModelFactory.get_by_name(model_args)
        #model = model_wrapper.get_model()
        
        # opt 
        opt = get_optimizer(model_wrapper.get_model(),
                            params.train.optimizer,
                            lr=params.train.lr,
                            wd=params.train.weight_decay)
        
        # loss
        loss = CCCLoss(num_bins=1)
        #loss = ComboLoss()
        
        # scheduler 
        print("Creating scheduler")
        scheduler = get_scheduler(params.train.lr_policy,
                                  opt,
                                  step_size=params.train.lr_decay)
        
        #metric
        metric = VA_metric
        
        # writers
        print("Creating TB writers")
        tb_path = Path(params.log_dir) / "summarywriters"
        summary_writers = {"train": SummaryWriter(str(tb_path / "train")), 
                           "validation": SummaryWriter(str(tb_path / "validation"))
                           }
        if params.train.scheme == "single":
        
            self.trainer = SimpleTrainer(
                model=model_wrapper,
                train_dataloader=train_dl,
                val_dataloader=val_dl,
                opt=opt,
                loss=loss,
                params=params,
                writers=summary_writers,
                scheduler=scheduler
            )
            
        elif params.train.scheme == "finetune":
            
            fine_scheduler = None
            
            self.trainer = SimpleTrainerFinetune(
                model=model_wrapper,
                train_dataloader=train_dl,
                val_dataloader=val_dl,
                opt=opt,
                loss=loss,
                params=params,
                writers=summary_writers,
                scheduler=scheduler,
                fine_scheduler=fine_scheduler)
        
    
    def start(self):
        
        print("Launching a simple trainer ...")
        
        self.trainer.run()
        
        
class OverfitProcess():
    """
    Similar to TrainingProcess, but deliberately attempts to overfit the model 
    
    """
    
    def __init__(self, params:Params) -> None:
        self.params = params
        
        self.train_params = params.train
        
        # get a train DL with a single batch in it
        self.train_dl = get_dataloader(self.train_params, train_mode="Train", task="VA", persist=True, tiny=True)
        
        print("Length of Dataloader: {}".format(len(self.train_dl)))
        
        # model 
        self.model_wrapper = ModelFactory.get_by_name(self.params)
        
        # opt
   
        self.opt = get_optimizer(self.model_wrapper.get_model(),
                            params.train.optimizer,
                            lr=params.train.lr,
                            wd=params.train.weight_decay)
        # loss
        
         # loss
        self.loss = CCCLoss(num_bins=1)
        self.mseloss = MSELoss(num_classes=1, reduction="mean")
        self.disloss = CustomCrossEntropyLoss(num_classes=24)
        
        # move loss calculator to GPU
        if self.params.cuda:
            self.loss.cuda()
        
        # scheduler
        
        self.scheduler = get_scheduler(params.train.lr_policy,
                                  self.opt,
                                  step_size=params.train.lr_decay)
        
    def start(self):
        
        print("Attempting to overfit the model ...")
        
        print("Freezing extractor parts ...")
        self.model_wrapper.freeze_extractors()
        
        print("Trainable parameters: {}".format(sum(p.numel() for p in self.model_wrapper.get_model().parameters() if p.requires_grad)))
        
        num_epochs = self.train_params.num_epochs
        #non_blocking = self.train_params.pin_memory
        
        print("Running for {} epochs".format(num_epochs))
        for epoch in range(1, num_epochs + 1):
            
            self.opt.zero_grad()
            
            epoch_loss = 0.0
            epoch_loss_ccc = 0.0
            epoch_loss_mse = 0.0
            epoch_loss_dis = 0.0
            
            num_batches = len(self.train_dl)
            for i, batch in enumerate(self.train_dl):
                label = batch["label"].cuda()
                qlabel = discretize_labels(label)
                image = batch["image"].cuda()
                audio = batch["audio"].cuda()
                
                # fwd pass
                predictions = self.model_wrapper.get_model()([audio, image])
                
                 # compute losses
                pred_valence = predictions["VA"][..., 0]
                loss_valence = self.loss(pred_valence, label[..., 0])
                pred_arousal = predictions["VA"][..., 1]
                loss_arousal = self.loss(pred_arousal, label[..., 1])
                
                # mean ccc loss
                cl = 0.5 * loss_valence + 0.5 * loss_arousal
                
                # additional mse loss, averaged for valence and arousal
                mse = 0.5 * self.mseloss(pred_valence, label[..., 0]) + 0.5 * self.mseloss(pred_arousal, label[..., 1])
                
                # additional discretized loss
                pred_dis = predictions["discretized"]
                dl = self.disloss(pred_dis, qlabel)
            
                # add losses
                total_loss = cl +  mse +  dl
                
                # bwd pass
                total_loss.backward()
                self.opt.step()
                if self.scheduler is not None and self.train_params.lr_policy == "cosine":
                    self.scheduler.step(epoch + i / num_batches)
                
                epoch_loss = epoch_loss + total_loss.item()
                epoch_loss_ccc += cl.item()
                epoch_loss_mse += mse.item()
                epoch_loss_dis += dl.item()
                
            epoch_loss = epoch_loss / num_batches    # should be div by 1
            epoch_loss_ccc /= num_batches
            epoch_loss_dis /= num_batches
            epoch_loss_mse /= num_batches
                
            print("Loss in epoch {}: {:.6f}, ccc loss: {:.6f}, mse loss: {:.6f}, discretised loss: {:.6f}".format(epoch, epoch_loss, epoch_loss_ccc, epoch_loss_mse, epoch_loss_dis))
            
            if self.scheduler is not None and self.train_params.lr_policy != "cosine":
                self.scheduler.step()
                
        print("Done!")
            
    
        