from shutil import copy
import json
from pathlib import Path
from typing import Dict
import time

from training import CLASS_WEIGHTS

from models.model_factory import ModelFactory, ModelWrapper
from utils.losses_metrics import Losses, MetricProvider, CCCLoss, CCC_score, VA_metric, ComboLoss, MSELoss, CustomCrossEntropyLoss
from utils.optimizer import Scheduler, get_scheduler
from utils.transforms import discretize_labels

from end2you.utils import Params

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.autograd import Variable

from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity


class Trainer():
    """
    a trainer class that is based on the functionality of E2Y but integrates dataloaders etc like ABAW NISL2021.
    Basically, this is E2Y with a TrainingProcess class outside of the trainer calling it separately.
    And also, we dont do the extra step of NISL to wrap the Model into a ModelWrapper. We work directly on the model instead
    """

    def __init__(self, model:ModelWrapper, train_dataloader:DataLoader, val_dataloader:DataLoader, optimizer:torch.optim.Optimizer, losses:Losses, metrics:MetricProvider, summary_writers:Dict[str,SummaryWriter], params:Params, scheduler=None, *args, **kwargs) -> None:
        
        self._model = model
        
        # Dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # the top level Params object (should contain train and model Params)
        self.params = params

        # paths
        self.root_dir = Path(params.root_dir)
        self.root_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoints_dir = Path(params.checkpoints_dir)
        
        # TB writers
        self.summary_writers = summary_writers

        # Optimizer
        self.optimizer = optimizer
        
        # LR Scheduler
        self.scheduler = scheduler

        # Losses
        self.losses = losses
        self.loss_names = losses.loss_names
        #self.loss_fns = {}
        
        # Metrics
        self.metrics = metrics
        self.metric_name = metrics.metric_name
        
        # collect scores
        # log the best scores into a dictionary
        self.best_scores = {}


    def train(self):
        """
        Runs the training loop. Based on E2Y trainer but with different dataloaders
        """
        
        # move the model and losses to cuda
        #if self.params.train.cuda: 
        #    self.losses.to_cuda()
        #    self._model.model_to_cuda()
        
        best_score = float("-inf")
        # TODO load best score from model if checkpoint path exists
        load_epoch = 0      # if we load from a checkpoint, this would be the epoch were we left off. Currently 0
        
        epochs = self.params.train.num_epochs   # number of epochs to train
        
        print("Starting training loop ...")
        
        start_epoch = load_epoch + 1
        for epoch in range(start_epoch, start_epoch + epochs): 

            # do the actual training
            self._do_epoch(epoch, is_training=True)

            # do the validation
            with torch.no_grad():
                validation_score = self._validate(epoch)
                
            # scheduler step
            if self.scheduler is not None:
                self.scheduler.step()    
            
            # logging step
            self._log_results(validation_score=validation_score,
                              best_score=best_score,
                              epoch=epoch)
            

        # save the best scores as a dictionary with the epochs they were achieved
        best_json_path = self.root_dir / "best_validation_scores.json"
        with open(str(best_json_path), "w") as f:
            json.dump(self.best_scores, f)
            
        
        print("Train phase completed.")
        
    
    def _log_results(self, validation_score, best_score, epoch):
        """
        Helper which logs the results to json and checkpoints
        
        """
        
        is_best = validation_score > best_score

        # do logging

        if is_best:
            print("Found new best model with mean metric {}: {:05.3f}".format(self.metric_name, validation_score))
            best_score = validation_score
            
            # save the best score to a dict, appending if possible
            entry = {"Epoch_{}_{}".format(epoch, self.metric_name) : best_score}
            self.best_scores.update(entry)
                        
            # save the latest score to dict
            last_json_path = self.root_dir / "last_validation_score"
            with open(str(last_json_path), "w") as f:
                json.dump({self.metric_name: validation_score}, f)       
               
            # save this dictionary at the end of the epoch 
            save_dict = {
                "validation_score": validation_score,
                "metric_name": "",
                "loss_name": "",
                "epoch": epoch, # no need to add 1 here since the counter is already incremented by 1
                "state_dict": self._model.get_model().state_dict(),
                "optim_dict": self.optimizer.state_dict(),
                "params": self.params   # save the params to reinstate the model later
            }
                
            self.save_checkpoint(state_dict=save_dict, is_best=is_best, ckpt_path=self.checkpoints_dir)
        
        

    def _do_epoch(self, epoch: int, is_training:bool=True) -> float:
        """
        Runs an epoch on the train or val set, depending on the is_training flag
        Computes losses and metrics, and returns a total score (used for validation)
        Also logs losses and metrics to the summary writers
        """
        
        debug = True
        
        process = "train" if is_training else "validation"  
        # select the summary writer
        writer = self.summary_writers[process]
        # select the appropropriate params
        params = self.params.train if is_training else self.params.valid
        # select the appropriate dl
        dl = self.train_dataloader if is_training else self.val_dataloader
        
        # Use the torch profiler to time this operation
        with profile(with_stack=False, profile_memory=True) as prof:

            # set model to train or eval
            if is_training:
                self._model.set_train()
            else:
                self._model.set_eval()
            
            # unused code - perhaps useful later?    
            #num_outs = self.model.num_outs if not isinstance(
            #    self.model, nn.DataParallel) else self.model.module.num_outs

            # the mean loss across all batches
            #mean_loss = 0.0
            
            
            #if self.params.tasks == "VA":
                #batch_preds = {"VA": np.empty((0, 2), dtype=np.float32)}
                #batch_labels = {"VA": np.empty((0, 2), dtype=np.float32)}
            #else:
            #    raise NotImplementedError
            
             # initialise all losses to 0
            epoch_losses = self._init_losses()
            
            # store the batch labels and predictions for the task into lists. These are reset every epoch
            batch_preds = {str(self.params.task): []}
            batch_labels = {str(self.params.task): []}

            # descriptions
            bar_string = "Training" if is_training else "Validation"
            miniters = 200  # less frequent updates so log does not get polluted
            num_batches = len(dl)

            # iteration over dataloader
            # with tqdm(total=num_batches, disable=True) as bar:
                # bar.set_description("{} model ...".format(bar_string))

            for n_iter, batch in enumerate(dl):
                
                if is_training:
                    # reset the optimizer
                    self.optimizer.zero_grad()
                
                #tic = time.time()
                
                # time using cuda events
                #start = torch.cuda.Event(enable_timing=True)
                #end = torch.cuda.Event(enable_timing=True)
                
                #start.record()
                
                with record_function("Data to GPU"):
                    # move data to GPU if available
                    if self.params.cuda:
                        # move audiovisual inputs 
                        image = batch["image"].cuda()
                        audio = batch["audio"].cuda()
                    else:
                        image = batch["image"]
                        audio = batch["audio"]
                    
                    labels = batch["label"]
                
                if not is_training:
                    with record_function("val labels to numpy"):   
                    # numpy labels for validation
                        np_labels = labels.cpu().numpy()    # cpu() call should have no effect since data is on CPU already [BS, S, 2]
                    
                with record_function("Labels to GPU"):   
                    if self.params.cuda:
                        labels = labels.cuda()
                        
                #end.record()  
                # sync call
                #torch.cuda.synchronize()
                
                #if debug:
                #    print("Time for preparing batch data: {:.3f} milliseconds".format(start.elapsed_time(end)))
                        
                #toc = time.time()
                #if debug:
                #    print("Time for preparing batch data: {:.6f} seconds".format(toc - tic))
                    
                # predict. Assuming predictions is a dict of tensors
                #return_estimates = not is_training
                
                # GPU  check
                if self.params.cuda:
                    assert image.is_cuda
                    assert audio.is_cuda
                    #if is_training:
                    assert labels.is_cuda
                
                
                ########################################
                ############# forward pass #############
                ########################################
                
                # time this operation
                #tic = time.time()
                
                with record_function("forward pass"): 
                    if is_training:
                        predictions = self._model.forward([audio, image], return_estimates=False)
                    else:
                        predictions, estimates = self._model.forward([audio, image], return_estimates=True)
                
                #toc = time.time()
                #if debug:
                    #print("Elapsed time for forward pass: {:.6f} seconds".format(toc - tic))
                #    print(prof)
            
                # a scalar Tensor to hold the loss
                # total_loss = Variable(torch.Tensor([0.0]), requires_grad=is_training)
                
                
                
                ########################################
                ######### compute batch losses #########
                ########################################
                
                with record_function("loss computation"):
                
                    batch_losses = {}
                    # time this op
                    #tic = time.time()
                    for loss_name in self.loss_names:
                    
                        if loss_name == "ccc":
                            va_losses = self.losses.compute_single_loss("ccc")(pred=predictions["VA"], target=labels) 
                            batch_losses.update(va_losses)
                        else: 
                            #batch_losses.update()
                            pass 
                            # TODO more loss functions
                            
                    # add to the total loss
                    #for loss_name in self.loss_names:
                    #    if loss_name in batch_losses.keys():   
                    #        total_loss += batch_losses[loss_name] 
                    
                    total_loss = sum([batch_losses[l] for l in self.loss_names])
                    #toc = time.time()
                    #if debug:
                    #    print("Elapsed time for computing losses: {:.3f} seconds".format(toc - tic))
                
                
                
                # make the backward pass        
                if is_training:
                    # time this operation
                    #tic = time.time()
                    with record_function("backward pass"):
                        total_loss.backward()
                        self.optimizer.step()
                    #toc = time.time()
                #    if debug:
                #        print("Time Elapsed for backward pass: {:.3f} seconds".format(toc - tic))
                        
                
                #for l in self.loss_names:
                #    compute_losses.update(self.lo)
                
                # add up the losses
                #for key in batch_losses.keys():
                #    loss = batch_losses[key]
                #    total_loss = total_loss + loss
                        
                # write summary every n batches
                if n_iter % params.save_summary_steps == 0:
                    
                    # time this operation
                    tic = time.time()
                    for key in batch_losses.keys():
                        loss = batch_losses[key].item()
                        writer.add_scalar("{}_loss".format(key), loss)  
                    toc = time.time()
                    if debug: 
                        print("Time elapsed for writing summary on step {}/{}: {:.3f} seconds".format(n_iter, num_batches, toc - tic))
                        
                    # more info on GPU usage
                    if self.params.cuda:
                        print(torch.cuda.list_gpu_processes())
                        #pass
                
                    
                # update global losses
                # mean_loss += total_loss / num_outs
                for k in batch_losses.keys():
                    epoch_losses[k] += batch_losses[k]
                
                # gather task predictions for evaluation
                if not is_training:
                    # convert labels and predictions to numpy. We care about the CCC here
                    #np_preds = predictions[params.task].cpu().numpy()   # [BS, S, 2]
                    np_preds = estimates[self.params.task] # comes already processed
                    #np_labels = labels.cpu().numpy()    # cpu() call should have no effect since data is on CPU already [BS, S, 2]
                    
                    
                    """
                    batch_size = self.params.valid.batch_size
                    seq_length = self.params.valid.seq_length
                    if self.params.task == "VA":
                        num_outs = 2
                    elif self.params.task == "EXPR":
                        num_outs = 7
                    elif self.params.task == "AU":
                        num_outs = 12
                    else: #default case is VA, so expect 2
                        num_outs = 2
                    
                    dummy = np.ones((batch_size, seq_length, num_outs))
                        
                    assert np_preds.shape == dummy.shape, "Predictions shape should be {}, is {}".format(dummy.shape, np_preds.shape)
                    assert np_labels.shape == dummy.shape, "Labels shape should be {}, is {}".format(dummy.shape, np_labels.shape)
                    """
                    
                    batch_preds[str(self.params.task)].append(np_preds)
                    batch_labels[str(self.params.task)].append(np_labels)
                    
                    # stop after 20 batches for debugging
                if debug and n_iter == 100:
                    break
                
                """
                if n_iter % params.save_summary_steps == 0:
                    
                    # convert labels and predictions to numpy. We care about the CCC here
                    np_preds = predictions[params.task].cpu().numpy()   # [BS, S, 2]
                    np_labels = labels.cpu().numpy()    # [BS, S, 2]
                    
                    # reshape into [BS * S, 2]
                    np_preds = np.reshape(np_preds, (-1, num_outs))
                    np_labels = np.reshape(np_labels, (-1, num_outs))
                    
                    # append batch predictions
                    batch_preds = np.concatenate([batch_preds, np_preds], axis=0)
                    batch_labels = np.concatenate([batch_labels, np_labels], axis=0)
                """
                # update the tqdm bar   
                #bar.set_postfix({" loss": "{:05.3}".format(total_loss.item())})
                #bar.update()
                
            ########################################
            ############## End loop ################
            ########################################
                
            # end tqdm 
                
        # end profiler
        if debug:
            print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=10))
                
                # debugging 
                #if is_training:
                #    break #  stop train loop after 1 batch
                #if debug:
                #    break
                
               
                
        
        ######################
        # scoring step
        ######################
        
        scores = {}
        
        # compute global mean losses for this epoch by dividing the summed losses by the number of batches
        global_mean_losses = {}
        
        for k in epoch_losses.keys():
            mean_loss = epoch_losses[k] / (n_iter + 1)
            global_mean_losses[k] = mean_loss
            
        # total mean loss for this epoch
        loss = 0.0
        for l in self.loss_names:
            loss += global_mean_losses[l]
        
        # compute the metric function for validation
        if not is_training:
            tic = time.time()
            # concatenate the predictions along the sample axis
            global_preds = np.concatenate(batch_preds[str(self.params.task)], axis=0) # [N, S, 2]
            global_labels = np.concatenate(batch_labels[str(self.params.task)], axis=0) # [N, S, 2]
        
            if self.params.task == "VA":
                if self.metric_name == "ccc":
                    # call the VA Metric function on the preds and labels for the entire data
                    [valence_ccc, arousal_ccc], mean_ccc = self.metrics.metric_fn(global_preds, global_labels)
                    scores.update({"valence": valence_ccc,
                                   "arousal": arousal_ccc,
                                   "mean": mean_ccc})
                    
                    #scores["main"] = mean_ccc
            toc = time.time()
            if debug:
                print("Elapsed time for calculating metrics: {:.3f} seconds".format(toc - tic))
        
            #epoch_summaries = [scores]
        
            tic = time.time()
            # write metrics
            for key, score in scores.items():
                writer.add_scalar("{}_evaluation/{}".format(process, key), score)
            # write loss        
            writer.add_scalar("loss/", loss)
            toc = time.time()
            if debug:
                 print("Elapsed time for writing evaluation summary at epoch end: {:.6f} seconds".format(toc - tic))
            
            
            # bit hacky - may need to fix later
            label_names = ["valence", "arousal"]
            
            # log to output
            score_list = ["{}: {:05.3f}".format(name, scores[name]) for name in label_names]
            str_scores = " -- ".join(score_list)
            str_scores = str_scores + " || loss : {:05.3f} ".format(loss) 
            print("Evaluation step of epoch {} with metric {} : ".format(epoch, self.metric_name) + str_scores)
            
            mean_label_score = np.mean([scores[l] for l in label_names])
                
            return mean_label_score
        
        else: # train step 
            
            # just return the loss
            
            print("Epoch {} || Train loss: {:05.3f} ".format(epoch,  loss))
            
            return loss
        


    def _validate(self, epoch:int) -> float:
        """
        Method that handles the validation process. 
        Returns a single float that is the validation score by which we decide if we have a new best value
        """

        """
        val_score = 0.0

        #process = "train" if self.params.is_training else "validation"
        process = "validation"
        writer = self.summary_writers[process]
        
        self._model.set_eval()

        bar_string = "Validation"
        with tqdm(total=len(self.val_dataloader)) as bar:
            bar.set_description("{} model")

        return val_score
        """
        
        return self._do_epoch(epoch, is_training=False)
    
    
    def save_checkpoint(self, state_dict:dict, is_best:bool, ckpt_path:Path):
        """
        Saves the model into a checkpoint, additionally, also saves if it is the best so far.
        """
        
        #ckpt_path = Path(ckpt)
        ckpt_path.mkdir(exist_ok=True, parents=True)
        
        filepath = ckpt_path / "last.pth.tar"
        
        torch.save(state_dict, str(filepath))
        
        if is_best:
            copy(filepath, str(ckpt_path / "best.pth.tar"))
            
            
    def load_checkpoint(self, ckpt_path:Path):
        """
        Loads checkpoint
        """        
        
        return torch.load(str(ckpt_path))   
    
        
    def _init_losses(self) -> dict:
        """
        Helper which initialises the losses dict with 0
        """   
        
        loss_dict = {}
        
        for loss in self.loss_names:
            if loss == "ccc":
                # add individual arousal, valence losses
                loss_dict["ccc_valence"] = 0.0
                loss_dict["ccc_arousal"] = 0.0
                loss_dict["ccc"] = 0.0
                
            else: 
                loss_dict[loss] = 0.0
                
        return loss_dict
    
    def _update_losses(self):
        pass
    
    
class FineTuneTrainer(Trainer):
    """
    A subclass which does a 2-stage training process
    """
        
    def __init__(self, model: ModelWrapper, train_dataloader: DataLoader, val_dataloader: DataLoader, 
                 optimizer: torch.optim.Optimizer,
                 losses: Losses, metrics: MetricProvider, summary_writers: Dict[str, SummaryWriter], params: Params, 
                 scheduler=None, finetune_scheduler=None, *args, **kwargs) -> None:
        super().__init__(model, train_dataloader, val_dataloader, optimizer, losses, metrics, summary_writers, params, scheduler, *args, **kwargs)
        
        # additional scheduler and optimizer for second stage
        # self.finetune_optimizer = finetune_optimizer
        self.finetune_scheduler = finetune_scheduler
        
        self.num_total_params = sum(p.numel() for p in self._model.get_model().parameters())
        
        
    def train(self):
        """
        Modified train loop, runs twice
        """       
        
        ##################################
        ############ Frozen ##############
        ##################################
        
        # move the model and losses to cuda
        #if self.params.train.cuda: 
        #    self.losses.to_cuda()
        #    self._model.model_to_cuda()
            
        # freeze the model extractors
        print("Freezing feature extractors ...")
        self._model.freeze_extractors()
        
        # output helpful info
        num_trainable_params = sum(p.numel() for p in self._model.get_model().parameters() if p.requires_grad)
        
        # log the best scores into a dictionary
        best_scores = {}

        best_score = float("-inf")
        # TODO load best score from model if checkpoint path exists
        load_epoch = 0      # if we load from a checkpoint, this would be the epoch were we left off. Currently 0
        
        epochs = self.params.train.num_epochs   # number of epochs to train
        
        print("Starting first training loop ...")
        
        print("Training {} out of {} params".format(num_trainable_params, self.num_total_params))
        
        start_epoch = load_epoch + 1 
        
        for epoch in range(start_epoch, start_epoch + epochs):
            
            epoch_tic = time.time()
            
            # run train step
            train_tic = time.time()
            self._do_epoch(epoch=epoch, is_training=True)
            train_toc = time.time()
            print("Train phase took {:.1f} seconds".format(train_toc - train_tic))
            
            # run validation step
            val_tic = time.time()
            with torch.no_grad():
                val_score = self._validate(epoch=epoch)
            val_toc = time.time()
            print("Val phase took {:.1f} seconds".format(val_toc - val_tic))
            
            # step the scheduler
            sch_tic = time.time()
            if self.scheduler is not None:
                self.scheduler.step()
            sch_toc = time.time()
            print("Scheduler took {:.1f} seconds".format(sch_toc - sch_tic))
            
            # log to checkpoint and json
            log_tic = time.time()
            self._log_results(validation_score=val_score,
                              best_score=best_score,
                              epoch=epoch)
            log_toc = time.time()
            print("Checkpointing took {:.1f} seconds".format(log_toc - log_tic))
            
            
            epoch_toc = time.time()
            print("Epoch {}/{} took {:.1f} seconds".format(epoch, epochs, epoch_toc - epoch_tic))
            
        print("Initial training phase completed")
            
        ########################################
        ############## Unfrozen ################
        ########################################
        
        # load the best model checkpoint from the first stage
        checkpoint = self.load_checkpoint(self.checkpoints_dir / "best.pth.tar")
        # set the state dict of the model
        self._model.get_model().load_state_dict(checkpoint["state_dict"])
        # set the state dict of the optimizer
        self.optimizer.load_state_dict(checkpoint["optim_dict"])
        
        # set new optimizer learning rate for finetune stage
        for g in self.optimizer.param_groups:
            g["lr"] = self.params.train.fine_lr
            
        # set a new scheduler
        self.finetune_scheduler = Scheduler().get(lr_scheduler=self.params.train.lr_policy,
                                                optimizer=self.optimizer,
                                                step_size=self.params.train.fine_lr_decay)
        
        # unlock feature extractors
        print("Unfreezing feature extractors ...")
        self._model.unfreeze_extractors()
        
        # set the new start epoch 
        start_epoch = epochs + 1
        
        fine_epochs = self.params.train.fine_num_epochs
        
        print("Starting second training loop ...")
        
         # output helpful info
        num_trainable_params = sum(p.numel() for p in self._model.get_model().parameters() if p.requires_grad)
        print("Training {} out of {} params".format(num_trainable_params, self.num_total_params))
        
        for epoch in range(start_epoch, start_epoch + fine_epochs):
            
            epoch_tic = time.time()
            
            # run train step
            train_tic = time.time()
            self._do_epoch(epoch=epoch, is_training=True)
            train_toc = time.time()
            print("Train phase took {:.1f} seconds".format(train_toc - train_tic))
            
            # run validation step
            val_tic = time.time()
            with torch.no_grad():
                val_score = self._validate(epoch=epoch)
            val_toc = time.time()
            print("Val phase took {:.1f} seconds".format(val_toc - val_tic))
            
            # step the scheduler
            sch_tic = time.time()
            if self.scheduler is not None:
                self.scheduler.step()
            sch_toc = time.time()
            print("Scheduler took {:.1f} seconds".format(sch_toc - sch_tic))
            
            # log to checkpoint and json
            log_tic = time.time()
            self._log_results(validation_score=val_score,
                              best_score=best_score,
                              epoch=epoch)
            log_toc = time.time()
            print("Checkpointing took {:.1f} seconds".format(log_toc - log_tic))
            
            epoch_toc = time.time()
            print("Epoch {}/{} took {:.1f} seconds".format(epoch, epochs + fine_epochs, epoch_toc - epoch_tic))
        
         # save the best scores as a dictionary with the epochs they were achieved
        best_json_path = self.root_dir / "best_validation_scores.json"
        with open(str(best_json_path), "w") as f:
            json.dump(self.best_scores, f)
            
        print("Finetuning phase completed.")
            
        
        
class SimpleTrainer():
    """
    A minimalistic trainer implementation
    """     
    
    def __init__(self, 
                 model:ModelWrapper, 
                 train_dataloader:DataLoader, 
                 val_dataloader:DataLoader,
                 opt:torch.optim.Optimizer,
                 loss: nn.Module,
                 writers: Dict[str, SummaryWriter],
                 params:Params,
                 scheduler = None, 
                 patience=8) -> None:
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader  = val_dataloader
        self.optimizer = opt
        self.scheduler = scheduler
        self.loss = loss
        self.mseloss = MSELoss(num_classes=1, reduction="mean")
        self.disloss = CustomCrossEntropyLoss(num_classes=24, weight=CLASS_WEIGHTS)   #always 24 cats for now
        self.metric_name = "ccc"
        self.writers = writers
        self.params = params
        
        # early stopping
        self.patience = patience
        self.no_improvement = 0    # counter for epochs with no improvement
        
        #####
        
        self.epochs = self.params.train.num_epochs
        self.best_scores = {}
        # init to a negative number    
        self.best_score = - 1000
        self.best_path = Path("")
        
        self.root_dir = Path(params.root_dir)
        self.ckpt_dir = Path(params.checkpoints_dir)
        
        print("Current device: ", torch.cuda.current_device())
    
    def run(self):
        """
        launches the training of the model
        """
        print("Run training")
        
        # transfer loss module to GPU
        if self.params.cuda:
            self.loss.cuda()
            self.mseloss.cuda()
            self.disloss.cuda()
        
            
        # check if pinned memory is used
        #is_blocking = self.params.train.pin_memory
        
        ##############################
        ##  Frozen ##
        ##############################
        
        # lock/unlock extractors 
        self.model.freeze_extractors()
        #self.model.unfreeze_extractors()
        
        # set model for train
        #self.model.get_model().train()
        
        start_ep = 1
        end_ep = self.params.train.num_epochs
        
        for epoch in range(start_ep, end_ep + 1):
            
            self.no_improvement += 1    # increment counter by 1 
            
            self._epoch(is_training=True, epoch=epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            with torch.no_grad():
                val_score = self._epoch(is_training=False, epoch=epoch)
                
            # assess results and checkpoint model    
            self._log_results(val_score, epoch)
                
            # early stopping
            if self.no_improvement >= self.patience:
                print("Stopping because validation score did not improve for {} epochs".format(self.patience))
                break
                
        ##################################
        ## Unfrozen ##
        ##################################
        
        #self.model.unfreeze_extractors()
        
        #start_ep = 
        
        
         
        # warmup
        #predictions = self.model.get_model()([torch.zeros(1,16,1,4000, device="cuda:0"), torch.zeros(1,16,3,160,160, device="cuda:0")])
        
        # profile a forward pass
        #with profile(with_stack=False, profile_memory=True) as p:
        """
        for i, batch in enumerate(self.train_dataloader):
            
            with record_function("GPU transfer"):
                # move to GPU
                #image = batch["image"].cuda(non_blocking=is_blocking)
                #audio = batch["audio"].cuda(non_blocking=is_blocking)
                #labels = batch["label"].cuda(non_blocking=is_blocking)
                image = batch["image"].cuda(non_blocking=is_blocking)
                audio = batch["audio"].cuda(non_blocking=is_blocking)
                labels = batch["label"].cuda(non_blocking=is_blocking)
        
            # forward pass
            with record_function("fwd pass"):
                predictions = self.model.get_model()([audio, image])
                # unpack
                preds = predictions["VA"]
            
            # loss calc
            cl = self.loss(preds, labels)
            
            # backward pass
            with record_function("bwd pass"):
                cl.backward()
                self.optimizer.step()
        """
        # profile
        #print(p.key_averages().table(sort_by='self_cpu_time_total', row_limit=20))        
    
        ###########################
        # Validation
        ###########################
        
        """
        self.model.get_model().eval()
        
        with torch.no_grad():
            
            batch_preds = []
            batch_targets = []
        
            for i, batch in enumerate(self.val_dataloader):
                
                labels = batch["label"].cuda(non_blocking=is_blocking)
                batch_targets.append(batch["label"].numpy())    # makes use of non-blocking if possible
                
                image = batch["image"].cuda(non_blocking=is_blocking)
                audio = batch["audio"].cuda(non_blocking=is_blocking)
                
                predictions = self.model.get_model()([audio, image])
                
                # compute loss
                cl = self.loss(predictions["VA"], labels)
                
                batch_preds.append(predictions["VA"].cpu().numpy())
            
            
            pred = np.concatenate(batch_preds, axis=0)
            target = np.concatenate(batch_targets, axis=0)
                
            ccc = VA_metric(pred, target)
        """   
      
        
        print("Training complete")
        
        print("Overall best result: ccc {:.3f}".format(self.best_score))
        print("Best model stored at {}".format(str(self.best_path)))
        
    
    # for models with multiple outputs
    def _compute_losses(self, pred_va, pred_cat, labels_va, labels_cat):
            
        loss_valence = self.loss(pred_va[..., 0], labels_va[..., 0])
        loss_arousal = self.loss(pred_cat[..., 1], labels_va[..., 1])
        mse_loss_valence = self.mseloss(pred_va[..., 0], labels_va[..., 0])
        mse_loss_arousal = self.mseloss(pred_va[..., 1], labels_va[..., 1])
        mse_loss = 0.5 * mse_loss_valence + 0.5 * mse_loss_arousal
        discrete_loss = self.disloss(pred_cat, labels_cat)
        
        return loss_valence, loss_arousal, mse_loss, discrete_loss    
    
    
    def _epoch(self, is_training=True, epoch:int=0) -> float:
        """
        Helper method that runs code for train or validation epoch
        """
        
        if is_training:
            self.model.get_model().train()
            params = self.params.train
            dl = self.train_dataloader
            process = "train"
        else:
            self.model.get_model().eval()
            params = self.params.valid
            dl = self.val_dataloader
            process = "validation"
            
            # allocate lists to hold predictions for metric computation
            batch_preds = []
            batch_targets = []
            
        is_blocking = params.pin_memory
        save_summary_steps = params.save_summary_steps
        writer = self.writers[process]
        
        epoch_loss = 0.0
        epoch_valence_loss = 0.0
        epoch_arousal_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_discrete_loss = 0.0
        
        # branch specific for aux models
        epoch_loss_branches = {
            
        }
        for branch in ["audio", "visual", "audiovisual"]:
            
            epoch_loss_branches.update({
                branch: {
                    "valence": 0.0,
                    "arousal": 0.0,
                    "mse": 0.0,
                    "discrete": 0.0,
                }
            })
            
        #epoch_valence_loss_audio = 0.0
        #epoch_arousal_loss_audio = 0.0
        #epoch_mse_loss_audio = 0.0
        #epoch_discrete_loss_audio = 0.0
        #epoch_valence_loss_visual = 0.0
        #epoch_arousal_loss_visual = 0.0
        #epoch_mse_loss_visual = 0.0
        #epoch_discrete_loss_visual = 0.0
        #epoch_valence_loss_audiovisual = 0.0
        #epoch_arousal_loss = 0.0
        #epoch_mse_loss = 0.0
        #epoch_discrete_loss = 0.0
        

        num_batches = len(dl)
        for i, batch in enumerate(dl):
            # data preparation
            labels = batch["label"].cuda(non_blocking=is_blocking)
            
            # quantise the labels
            qlabels = discretize_labels(labels)
            qlabels.cuda(non_blocking=is_blocking)  # might not be necessary since labels already on GPU
            
            if not is_training:
                batch_targets.append(batch["label"].numpy())    # makes use of non-blocking if possible
                
            image = batch["image"].cuda(non_blocking=is_blocking)
            audio = batch["audio"].cuda(non_blocking=is_blocking)
            
            # forward pass
            predictions = self.model.get_model()([audio, image])
            
            if not self.params.model.model_name in ["auxattn", "auxrnn", "hybridattn", "hybridrnn"]:
                
                pred_va = predictions["VA"]
                pred_cat = predictions["discretized"]
                
                loss_valence, loss_arousal, mse, discrete = self._compute_losses(pred_va=pred_va, pred_cat=pred_cat, labels_va=labels, labels_cat=qlabels)
              
            else:
                # add up the losses of each branch
                loss_valence, loss_arousal, mse, discrete = 0.0, 0.0, 0.0, 0.0 
                for branch in ["audio", "visual", "audiovisual"]:
                    
                    pred_va = predictions["VA_" + branch]
                    pred_cat = predictions["CAT_" + branch]
                    
                    ccc_v, ccc_a, mse_branch, dis = self._compute_losses(pred_va=pred_va, pred_cat=pred_cat, labels_va=labels, labels_cat=qlabels)
                    loss_valence += ccc_v
                    loss_arousal += ccc_a
                    mse += mse_branch
                    discrete += dis  
                    
                    # update epoch loss 
                    epoch_loss_branches[branch]["valence"] += ccc_v.item()
                    epoch_loss_branches[branch]["arousal"] += ccc_a.item()
                    epoch_loss_branches[branch]["mse"] += mse_branch.item()
                    epoch_loss_branches[branch]["discrete"] += dis.item()
            
    
            # compute losses
            #pred_valence = predictions["VA"][..., 0]
            #loss_valence = self.loss(pred_valence, labels[..., 0])
            #pred_arousal = predictions["VA"][..., 1]
            #loss_arousal = self.loss(pred_arousal, labels[..., 1])
            
            # mean ccc loss
            cl = 0.5 * loss_valence + 0.5 * loss_arousal
            
            # additional mse loss, averaged for valence and arousal
            #mse = 0.5 * self.mseloss(pred_valence, labels[..., 0]) + 0.5 * self.mseloss(pred_arousal, labels[..., 1])
            
            # additional discretized loss
            #pred_dis = predictions["discretized"]
            #discrete = self.disloss(pred_dis, qlabels)
            
            # add losses
            total_loss = cl + self.params.train.lambda_mse * mse + self.params.train.lambda_dis * discrete
            
            # add running losses
            epoch_loss = epoch_loss + total_loss.item()
            epoch_valence_loss = epoch_valence_loss + loss_valence.item()
            epoch_arousal_loss = epoch_arousal_loss + loss_arousal.item()
            epoch_mse_loss += mse.item()
            epoch_discrete_loss += discrete.item()
            
            # backward pass
            if is_training:
                self.optimizer.zero_grad() # zero the optimizer
                total_loss.backward()
                self.optimizer.step()
                
                if (self.scheduler is not None) and (self.params.train.lr_policy.lower() == "cosine"):
                    self.scheduler.step(epoch + i / num_batches)
            
            if not is_training:
                
                key = "VA" if not self.params.model.model_name in ["auxattn", "auxrnn", "hybridattn", "hybridrnn"] else "VA_audiovisual"
                
                batch_preds.append(predictions[key].cpu().numpy()) # store batch predictions
                
            # summary writer
            if i % save_summary_steps == 0:
                writer.add_scalar("loss", total_loss.item())
                writer.add_scalar("ccc_loss_valence", loss_valence.item())
                writer.add_scalar("ccc_loss_arousal", loss_arousal.item())
                writer.add_scalar("mse_loss", mse.item())
                writer.add_scalar("discretized_loss", discrete.item())
                print("Batch {}/{} {} loss: {:.3f}".format(i, num_batches, process, total_loss.item()))
                
                
        # compute mean over batches
        epoch_loss = epoch_loss / num_batches
        epoch_valence_loss = epoch_valence_loss / num_batches
        epoch_arousal_loss = epoch_arousal_loss / num_batches
        epoch_mse_loss /= num_batches
        epoch_discrete_loss /= num_batches
        
        for branch in epoch_loss_branches.keys():
            for key in epoch_loss_branches[branch].keys():
                epoch_loss_branches[branch][key] /= num_batches
            
                
        # compute metrics
        if not is_training:
             
            pred = np.concatenate(batch_preds, axis=0)
            target = np.concatenate(batch_targets, axis=0)
                
            (ccc_valence, ccc_arousal), ccc = VA_metric(pred, target)
            
            mean_ccc = (ccc_valence + ccc_arousal) / 2.0
            
            final_score = mean_ccc
            
            # log the metrics and losses 
            print("Evaluation epoch {}/{} with metric {}: valence {:.3f}, arousal {:.3f} || loss: valence {:.3f}, arousal {:.3f}, mse: {:.3f}, discrete: {:.3f}".format(epoch, 
                                                                                                                self.epochs, 
                                                                                                                self.metric_name, 
                                                                                                                ccc_valence, 
                                                                                                                ccc_arousal, 
                                                                                                                epoch_valence_loss,
                                                                                                                epoch_arousal_loss,
                                                                                                                epoch_mse_loss,
                                                                                                                epoch_discrete_loss))
            # additional output
            if self.params.model.model_name in ["auxattn", "auxrnn", "hybridattn", "hybridrnn"]:
                fmtstr = ""
                for branch, data in epoch_loss_branches.items():
                    fmtstr += "{} valence: {:.3f}, arousal: {:.3f}, mse: {:.3f}, discrete: {:.3f}".format(branch, data["valence"], data["arousal"], data["mse"], data["discrete"])
                print("Losses per branch: " + fmtstr)
        
            writer.add_scalar("ccc valence", ccc_valence)
            writer.add_scalar("ccc_arousal", ccc_arousal)
            writer.add_scalar("ccc", mean_ccc)
        
        else:
            
            print("Training epoch {}/{} loss: valence {:.3f}, arousal {:.3f}, mse: {:.3f}, discrete: {:.3f}".format(
                epoch, self.epochs, epoch_valence_loss, epoch_arousal_loss, epoch_mse_loss, epoch_discrete_loss))
            
            # additional output
            if self.params.model.model_name in ["auxattn", "auxrnn", "hybridattn", "hybridrnn"]:
                fmtstr = ""
                for branch, data in epoch_loss_branches.items():
                    fmtstr += "{} valence: {:.3f}, arousal: {:.3f}, mse: {:.3f}, discrete: {:.3f}".format(branch, data["valence"], data["arousal"], data["mse"], data["discrete"])
                print("Losses per branch: " + fmtstr)
            
            writer.add_scalar("epoch loss", epoch_loss)
            writer.add_scalar("epoch valence loss", epoch_valence_loss)
            writer.add_scalar("epoch arousal loss", epoch_arousal_loss)
            
            final_score = 0.0
            
        return final_score
    
    
    def _log_results(self, validation_score, epoch):
        """
        Helper which logs the results to json and checkpoints
        
        """
        
        is_best = validation_score > self.best_score

        # do logging

        if is_best:
            print("Found new best model with mean metric {}: {:05.3f}".format(self.metric_name, validation_score))
            self.best_score = validation_score  # update the best score attribute
            
            # reset early stopping
            self.no_improvement = 0
            
            # save the best score to a dict, appending if possible
            entry = {"Epoch_{}_{}".format(epoch, self.metric_name) : self.best_score}
            self.best_scores.update(entry)
                        
            # save the latest score to dict
            if not self.root_dir.exists():
                self.root_dir.mkdir(exist_ok=True, parents=True)
            last_json_path = self.root_dir / "last_validation_score"
            with open(str(last_json_path), "w") as f:
                json.dump({self.metric_name: validation_score}, f)       
               
            # save this dictionary at the end of the epoch 
            save_dict = {
                "validation_score": validation_score,
                "metric_name": "",
                "loss_name": "",
                "epoch": epoch, # no need to add 1 here since the counter is already incremented by 1
                "state_dict": self.model.get_model().state_dict(),
                "optim_dict": self.optimizer.state_dict()
            }
                
            self.save_checkpoint(state_dict=save_dict, is_best=is_best, ckpt_path=self.ckpt_dir)
        
     
    def save_checkpoint(self, state_dict:dict, is_best:bool, ckpt_path):
        #ckpt_path = Path(ckpt)
        ckpt_path.mkdir(exist_ok=True, parents=True)
        
        filepath = ckpt_path / "last.pth.tar"
        
        torch.save(state_dict, str(filepath))
        
        if is_best:
            
            self.best_path = ckpt_path / "best.pth.tar"
            
            copy(filepath, str(self.best_path))
       
            
    def load_checkpoint(self, ckpt_path:Path):
        """
        Loads checkpoint
        """        
        
        return torch.load(str(ckpt_path))   
    
              
    
class SimpleTrainerFinetune(SimpleTrainer):
    """
    Extends the SimpleTrainer class by an additional train stage in which the feature extractors are unfrozen and a smaller lr is applied
    """
    def __init__(self, model: ModelWrapper, train_dataloader: DataLoader, val_dataloader: DataLoader, 
                 opt: torch.optim.Optimizer, loss: CCCLoss, writers: Dict[str, SummaryWriter], params: Params, 
                 scheduler=None,
                 fine_scheduler=None) -> None:
        super().__init__(model, train_dataloader, val_dataloader, opt, loss, writers, params, scheduler)
        # add new scheduler
        self.fine_scheduler = fine_scheduler
        # overwrite epochs
        self.epochs = self.params.train.num_epochs + self.params.train.fine_num_epochs
        
        
    def run(self):
        """
        launches the training of the model
        """
        print("Run training")
        
        # transfer loss module to GPU
        if self.params.cuda:
            self.loss.cuda()
            
        # if the model is pretrained, go directly to fine-tune stage. Else do a frozen stage pass
        if self.params.model.pretrained_path:
            
            checkpoint = torch.load(str(self.params.model.pretrained_path))
            # set the state dict of the model
            self.model.get_model().load_state_dict(checkpoint["state_dict"])
            # set the state dict of the optimizer
            self.optimizer.load_state_dict(checkpoint["optim_dict"])
            
            
        else: 
            ##############################
            ##########  Frozen ###########
            ##############################
            
            # lock/unlock extractors 
            self.model.freeze_extractors()
            
            start_ep = 1
            end_ep = self.params.train.num_epochs
            
            for epoch in range(start_ep, end_ep + 1):
                
                self._epoch(is_training=True, epoch=epoch)
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                with torch.no_grad():
                    val_score = self._epoch(is_training=False, epoch=epoch)
                    
                self._log_results(val_score, epoch)
                
            # load the best model state and optimizer
            # load the best model checkpoint from the first stage
            checkpoint = self.load_checkpoint(self.ckpt_dir / "best.pth.tar")
            # set the state dict of the model
            self.model.get_model().load_state_dict(checkpoint["state_dict"])
            # set the state dict of the optimizer
            self.optimizer.load_state_dict(checkpoint["optim_dict"])
         
            print("Frozen stage complete!")   

        ##################################
        ########### Unfrozen #############
        ##################################
        
        self.model.unfreeze_extractors()
        
        
        # set new optimizer learning rate for finetune stage
        for g in self.optimizer.param_groups:
            g["lr"] = self.params.train.fine_lr
            
        # create new scheduler with the updated optimizer
        self.fine_scheduler = get_scheduler(self.params.train.lr_policy,
                                            self.optimizer,
                                            step_size=self.params.train.fine_lr_decay)
        
        # adjust epoch range
        start_ep = end_ep + 1
        end_ep = self.epochs
        
        for epoch in range(start_ep, end_ep + 1):
            
            self._epoch(is_training=True, epoch=epoch)
            
            if (self.fine_scheduler is not None) and (self.params.train.lr_policy.lower() != "cosine"): # cosine is called per step
                self.fine_scheduler.step()
            
            with torch.no_grad():
                val_score = self._epoch(is_training=False, epoch=epoch)
                
            self._log_results(val_score, epoch)
        
        print("Unfrozen stage complete!")
        
        print("Training complete!")