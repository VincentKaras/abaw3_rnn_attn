"""
Wrapper classes for optimizer, learning rate scheduler, losses and metrics

"""

import torch
import torch.nn as nn
from typing import List, Optional
from utils.losses_metrics import NLL, CCCLoss, CustomCrossEntropyLoss, MAELoss, MSELoss, Accuracy
from utils.losses_metrics import NLLRegressionLoss, KLDRegressionLoss

class Optimizer:
    """
    A wrapper class whose get method returns an optimizer object.
    Could be refactored into a method since its behavior is static.
    """
    def __init__(self):
        pass

    def get(self,
            model: nn.Module,
            optimizer: str,
            lr: float,
            wd: float,
            momentum: float = 0,
            betas: List[float] = [0.9, 0.999]) -> torch.optim.Optimizer:

        if optimizer.lower() == "sgd":
            optim = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=momentum,
                                    weight_decay=wd,
                                    )
        elif optimizer.lower() == "adam":
            optim = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=wd,
                                     betas=betas)
        elif optimizer.lower() == "none":
            optim = Optimizer()
        else:
            raise ValueError("Optimizer {} unsupported".format(optimizer))
        return optim
    
    
def get_optimizer( model: nn.Module,
            optimizer: str,
            lr: float,
            wd: float,
            momentum: float = 0,
            betas: List[float] = [0.9, 0.999]) -> torch.optim.Optimizer:
    """
    Factory method that creates otptimizer objects
    """
    if optimizer.lower() == "sgd":
            optim = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=momentum,
                                    weight_decay=wd,
                                    )
    elif optimizer.lower() == "adam":
        optim = torch.optim.Adam(model.parameters(),
                                    lr=lr,
                                    weight_decay=wd,
                                    betas=betas)
    elif optimizer.lower() == "adamw":
        optim = torch.optim.AdamW(model.parameters(),
                                lr=lr,
                                weight_decay=wd,
                                betas=betas)
    elif optimizer.lower() == "none":
        optim = None
    else:
        raise ValueError("Optimizer {} unsupported".format(optimizer))
    return optim
    
    
class Scheduler:
    """
    Wrapper class which returns a learning rate scheduler object.
    Could be refactored into a method
    """
    def __init__(self):
        pass

    def get(self,
            lr_scheduler: str,
            optimizer: torch.optim.Optimizer,
            step_size: int,
            T_max: int = 200,
            gamma: float = 0.1):
        """
        Return a scheduler object
        :param lr_scheduler:
        :param optimizer:
        :param step_size: epochs for StepLR
        :param T_max: iterations for CosineAnnealingLR only
        :param gamma: float for StepLR
        :return:
        """

        if lr_scheduler.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=step_size,
                                                        gamma=gamma)
        elif lr_scheduler.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                   T_max=T_max)
        elif lr_scheduler.lower() == "none":
            scheduler = None
        else:
            raise ValueError("lr_scheduler {} unsupported".format(lr_scheduler))


def get_scheduler(lr_scheduler: str, optimizer: torch.optim.Optimizer, step_size: int, T_max: int = 200,gamma: float = 0.1):
    """
    Return a scheduler object
    :param lr_scheduler:
    :param optimizer:
    :param step_size: epochs for StepLR
    :param T_max: iterations for CosineAnnealingLR only
    :param gamma: float for StepLR
    :return:
    """
            
    if lr_scheduler.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=step_size,
                                                        gamma=gamma)
    elif lr_scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                T_0=T_max)
    elif lr_scheduler.lower() == "none":
        scheduler = None
    else:
        raise ValueError("lr_scheduler {} unsupported".format(lr_scheduler))
    
    return scheduler


class Criterion:
    def __init__(self):
        pass

    def get(self,
            loss: str,
            num_classes: int,
            reduction: str = "mean",
            weight: Optional[torch.Tensor] = None,
            pos_weight: Optional[torch.Tensor] = None):

        self.num_classes = num_classes
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.loss_name = loss
        loss_name = loss
        # split into components and make a list
        if "+" in loss_name:
            loss_name = loss_name.split("+")
        else:
            loss_name = [loss_name]
        loss_funcs = []
        for single_loss in loss_name:
            loss_func = self.get_single_loss(single_loss)
            loss_funcs.append(loss_func)

        # do we need to mess with decl here?
        def combine_losses(pred, target):
            loss = 0
            for loss_func in loss_funcs:
                loss += loss_func(pred, target)
            return loss

        return combine_losses()


    def get_single_loss(self, loss):
        if loss.lower() in ["ce", "cce", "bce", "mse", "l1", "l1loss", "ccc", "negative_ccc", "kld"]:

            def inner_func(pred, target):
                pred = pred[..., :self.num_classes]
                return loss_func(pred, target)
            return inner_func
        else:
            if loss.lower() == "nll_reg" or loss.lower() == "nll_regression":
                loss_func = NLLRegressionLoss(num_classes=self.num_classes,
                                              reduction=self.reduction)
            elif loss.lower() == "kld_reg" or loss.lower() == "kld_regression":
                loss_func = KLDRegressionLoss(num_classes=self.num_classes,
                                              reduction=self.reduction)
            else:
                raise ValueError("loss {} unsupported".format(loss))

class Metric:
    def __init__(self):
        pass

    def get(self,
            metric: str,
            num_classes: int,
            reduction: str = "mean",
            apply_activation: str = "softmax"
            ):
        """
        Return a metric object
        :param metric: The str selector
        :param num_classes: The number of classes
        :param reduction: The reduction strategy
        :param apply_activation: Activation function
        :return:
        """
        if metric.lower() == "mse":
            m_func = MSELoss(num_classes=num_classes,
                         reduction=reduction)
        elif metric.lower() == "mae":
            m_func = MAELoss(num_classes=num_classes,
                         reduction=reduction)
        elif metric.lower() == "acc" or metric.lower() == "accuracy":
            m_func = Accuracy(num_classes=num_classes,
                              apply_activation=apply_activation)
        elif metric.lower() == "nll":
            m_func = NLL(num_classes=num_classes,
                         reduction=reduction,
                         apply_activation=apply_activation)
        elif metric.lower() == "nll_reg" or metric.lower() == "nll_regression":
            m_func = NLLRegressionLoss(num_classes=num_classes,
                                       reduction=reduction)
        else:
            raise ValueError("Metric {} not supported".format(metric))
        return m_func


