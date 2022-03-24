import torch
import numpy as np
from scipy.special import expit, softmax
from sklearn.metrics import f1_score, accuracy_score

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from typing import List


def invert_sigmoid(y, eps = 1e-8):
    """
    Implementation of logit() that does not yield inf/nan by applying eps. Used for inverting the sigmoid
    :param y: array
    :return: log(ytol / (1-ytol)), where ytol is y with 0 and 1 shifted by eps
    """
    assert 0.0 < eps < 1.0
    y[y == 0] = eps
    y[y == 1] = 1 - eps
    return torch.log(torch.div(y, 1-y))


def invert_softmax(y, eps = 1e-8):
    """
    Invert the softmax function. Equivalent to log(y)
    :param y: The array
    :param eps: Small value to add so log does not go 0 or infinite
    :return: An array of the same shape as y
    """
    assert 0.0 < eps < 1.0
    y[y == 0] = eps
    y[y == 1] = eps
    return torch.log(y)


def averaged_f1_score(input:np.ndarray, target:np.ndarray):
    """
    Averages the f1 scores for each label
    :param input: predictions array N x Nlabels
    :param target: true labels array N x Nlabels
    :return: A tuple of fhe mean f1 score (float) and a list of the individual f1 scores
    """
    assert len(input.shape) == 2
    f1_scores = [f1_score(input[:, i], target[:, i]) for i in range(input.shape[-1])]
    return np.mean(f1_scores), f1_scores


def averaged_accuracy(input:np.ndarray, target:np.ndarray):
    """
    Averages the accuracy for each label
    :param input: predictions array N x Nlabels
    :param target: true labels array N x Nlabels
    :return: A tuple of mean accuracy score (float) and a list of individual accuracy scores
    """
    assert len(input.shape) == 2
    acc_scores = [accuracy_score(input[:, i], target[:, i]) for i in range(input.shape[-1])]
    return np.mean(acc_scores), acc_scores


def CCC_score(input: np.ndarray, target: np.ndarray):
    """
    Compute the ccc
    :param input: predictions
    :param target: true labels
    :return: A float between -1 and +1
    """

    input = input.reshape(-1,)
    target = target.reshape(-1,)

    cov = np.mean((input - np.mean(input)) * (target - np.mean(target)))
    return 2 * cov / (np.var(input) + np.var(target) + (np.mean(input) - np.mean(target)) ** 2)


################################################
################ Custom Losses #################
################################################

class CustomCrossEntropyLoss(torch.nn.Module):
    """
    A custom implementation of cross entropy loss which maps a discretised probability input onto a continuous target.
    """

    def __init__(self, num_bins=20, value_range=None):
        super(CustomCrossEntropyLoss, self).__init__()
        if value_range is None:
            self.value_range = [-1, 1]
        self.num_bins = num_bins
        assert self.num_bins != 1
        self.value_range = np.linspace(*self.value_range, self.num_bins + 1)

    def forward(self, inputs, targets):
        targets = targets.view(-1)
        digitized_targets = np.digitize(targets.data.cpu().numpy(), self.value_range) - 1
        digitized_targets[digitized_targets == self.num_bins] = self.num_bins - 1
        new_targets = Variable(torch.LongTensor(digitized_targets))
        if inputs.is_cuda:
            new_targets = new_targets.cuda()

        return F.cross_entropy(inputs, new_targets)


class CCCLoss(torch.nn.Module):
    """
    A custom implementation of ccc loss. Defined as 1 - CCC. 
    When number of bins > 1, maps the probability input onto a continuous target.
    """
    def __init__(self, num_bins:int=1, value_range=None):
        super(CCCLoss, self).__init__()
        assert num_bins > 0, "Number of bins must be an integer equal or greater to 1"
        if value_range == None:
            self.value_range = [-1, 1]
        else:
            self.value_range = value_range
        self.num_bins = num_bins
        if self.num_bins != 1:
            bins = np.linspace(*self.value_range, num=self.num_bins)
            self.bins = Variable(torch.as_tensor(bins, dtype=torch.float32).cuda()).view((1, -1))

    def forward(self, inputs, targets):

        # if input is not continuous, apply softmax and multiply with bins to get the expectation 
        if self.num_bins != 1:
            inputs = F.softmax(inputs, dim=-1)
            inputs = (self.bins * inputs).sum(-1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        inputs_mean = torch.mean(inputs)
        targets_mean = torch.mean(targets)
        inputs_var = torch.var(inputs)
        targets_var = torch.var(targets)

        cov = torch.mean((inputs - inputs_mean) * (targets - targets_mean))
        ccc = 2 * cov / (inputs_var + targets_var + torch.square(inputs_mean - targets_mean))

        return 1 - ccc

        #eps = 1e-8
        

class CustomLoss(torch.nn.Module):
    """
    Custom losses implemented as nn Modules. 
    """
    def __init__(self):
        super(CustomLoss, self).__init__()
        pass
    def forward(self, pred, target):
        raise NotImplementedError
    def to(self, device):
        pass
    def load_state_dict(self, state_dict):
        pass

def get_mean_sigma(pred):
    """
    Helper function that computes for NLL and KLD regression.
    """
    num_classes = pred.size(-1) // 2
    y_hat, sigma_square = pred[..., :num_classes], pred[..., :num_classes]
    sigma_square_pos = torch.log(1 + torch.exp(sigma_square)) + 1e-6
    return y_hat, sigma_square_pos


class NLLRegressionLoss(CustomLoss):

    def __init__(self,
        num_classes: int,
        reduction: str = "mean",
        apply_activation: str = "none"):
        super(NLLRegressionLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.optimal = "min"
        self.apply_activation = apply_activation

    def forward(self, pred, target):
        """
        Negative Log-Likelihood loss function for regression
        :param pred The pred tensor (N, 2*C) combination or log_sigma**2 and predictions
        :param target The target tensor (N, C)
        """
        N, C_twice = pred.size()
        assert C_twice % 2 == 0,  "Prediction vector must have dimension divisible by two"
        num_classes =  C_twice // 2
        assert self.num_classes == num_classes, "Expect tensor shape ({}), got ({})".format((N, self.num_classes), pred.size())
        y_hat, sigma_square = get_mean_sigma(pred)
        MSE = 0.5 * ((target - y_hat) ** 2)
        loss = 0.5 * torch.log(sigma_square) + torch.div(MSE, sigma_square) + 1e-6

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError("{} reduction not supported".format(self.reduction))

class KLDRegressionLoss(CustomLoss):
    def __init__(self,
                 num_classes: int,
                 reduction: str,
                 apply_activation: str,
                 ):
        super(KLDRegressionLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.optimal = "min"
        self.apply_activation = apply_activation

    def forward(self, pred, target) -> float:
        """
        Kullback Leibler Divergence for regression task
        :param pred: pred Tensor (N, 2*C), can be separated into y_hat and log_sigma_square
        :param target: target Tensor, (N, 2*C), teacher soft labels
        :return:
        """
        N, C_twice = pred.size()
        assert C_twice%2==0, "When using NLL as regression loss, prediction vector dimension must be divisible by two"
        num_classes = C_twice // 2
        assert num_classes == self.num_classes, "Expect tensor shape ({}), got ({})".format((N, self.num_classes), pred.size())
        y_hat, sigma_square_hat = get_mean_sigma(pred)
        y, sigma_square = get_mean_sigma(target)
        loss = torch.log(torch.sqrt(sigma_square)) - torch.log(torch.sqrt(sigma_square_hat))
        loss += torch.div(sigma_square_hat + (y_hat - y)**2, 2*sigma_square + 1e-6)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError("{} reduction not supported".format(self.reduction))

        #raise NotImplementedError

def nll_regression(mean, sigma, target):
    return (0.5 * torch.log(sigma) + 0.5 * torch.div((mean - target) ** 2, sigma) + 1e-6).mean()

class NLL(CustomLoss):
    """
    Wrapper that 
    """
    def __init__(self,
                 num_classes: int,
                 reduction: str = "mean",
                 apply_activation: str = "softmax"):
        super(NLL, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.optimal = "min"
        self.apply_activation = apply_activation

    def forward(self, pred, target) -> float:
        """
        NLL Loss
        :param pred: predictions Tensor (N, C)
        :param target: target Tensor (N, )
        :return:
        """
        # number of samples in the batch
        N = pred.size(0)

        pred = pred[..., : self.num_classes]
        if self.apply_activation == "softmax":
            pred = F.softmax(pred, dim=-1)
        elif self.apply_activation == "sigmoid":
            pred = nn.sigmoid(pred)

        return nn.NLLLoss(reduction=self.reduction)(torch.log(pred + 1e-6), target)

class Accuracy(CustomLoss):
    def __init__(self,
                 num_classes: int,
                 reduction: str = "mean",
                 apply_activation: str = "softmax"):
        super(Accuracy, self).__init__()
        self.apply_activation = apply_activation
        self.reduction = reduction
        self.num_classes = num_classes
        self.optimal = "max"

    def forward(self, pred, target) -> float:
        pred = pred[..., :self.num_classes]
        if self.apply_activation == "softmax":
            pred = F.softmax(pred, dim=-1)
        elif self.apply_activation == "sigmoid":
            pred = nn.sigmoid(pred)
        elif self.apply_activation == "none":
            pass
        else:
            raise ValueError("Activation {} unsupported".format(self.apply_activation))
        assert len(pred.size()) > 1, "Expected predictions tensor to have >2 dims, got {}".format(pred.size())
        # do an argmax
        pred = torch.max(pred, dim=-1)[1]
        # samples
        N = pred.size(0)
        assert pred.shape == target.shape
        return torch.eq(pred, target).sum() / float(N)

class MSELoss(nn.Module):
    def __init__(self,
                 num_classes: str,
                 reduction: str,
                 ):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.num_classes = num_classes
        self.optimal = "min"
        self.apply_activation = "none"

    def forward(self, pred, target):
        #pred = pred[..., :self.num_classes]
        assert pred.size() == target.size(), "pred tensor size must match target tensor, but is {}, {}".format(pred.size(), target.size())
        return F.mse_loss(pred, target, reduction=self.reduction)

class MAELoss:
    def __init__(self,
                 num_classes: str,
                 reduction: str = "mean"):
        super(MAELoss, self).__init__()
        self.reduction = reduction
        self.num_classes = num_classes
        self.optimal = "min"
        self.apply_activation = "none"

    def forward(self, pred, target):
        pred = pred[..., :self.num_classes]
        assert pred.size() == target.size(), "pred tensor size must match target tensor"
        return F.l1_loss(pred, target, reduction=self.reduction)

################################################
################# Metrics ######################
################################################

def VA_metric(input, target):
    """
    Custom Metric for the VA task which calculates the ccc scores of valence and arousal. 
    :param input: predictions, N x 2 [valence arousal]
    :param target: true labels, N x 2 [valence arousal]
    :return: a tuple of a list of valence and arousal scores and the sum of the scores
    """
    valence_score = CCC_score(input[..., 0], target[..., 0])
    arousal_score = CCC_score(input[..., 1], target[..., 1])
    return [valence_score, arousal_score], sum([valence_score, arousal_score])


def EXPR_metric(input, target):
    """
    Custom metric for the EXPR task. Calculates the F1 and accuracy scores
    :param input: predictions
    :param target: true labels
    :return: A tuple of a list of f1 and accuracy scores, as well as a float, the weighted sum of F1 and accuracy
    """

    def prep(x:np.ndarray):
        if not len(x.shape) == 1:
            if x.shape[1] == 1:
                return x.reshape(-1,)  # flatten to remove extra dim
            else:  # softmax
                return np.argmax(x, axis=-1)

    input = prep(input)
    target = prep(target)

    f1 = f1_score(input, target, average="macro")
    acc = accuracy_score(input, target)
    return [f1, acc], 0.67*f1 + 0.33 * acc


def AU_metric(input, target):
    """
    Custom metric for the AU task. Calculates the average f1 and accuracy scores
    :param input: predictions
    :param target: true labels
    :return: A tuple of: A list with the average f1 and accuracy scores and a float, the weighted sum
    """

    f1_avg, _ = averaged_f1_score(input, target)
    #input = input.reshape(-1,)
    #target = target.reshape(-1,)
    avg_acc, _ = averaged_accuracy(input, target)
    return [f1_avg, avg_acc], 0.5 * f1_avg + 0.5 * avg_acc


class MetricProvider:
    """
    Works similar to the provider in E2Y. 
    Replaces the Metric class in optimizer.py
    """

    def __init__(self, metric:str="ccc") -> None:
        self.metric_fn = self._get_metric(metric) if metric else None
        self.metric_name = metric

    def _get_metric(self, metric:str):
        """
        Factory method providing the proper metric
        """

        return {
            "ccc": VA_metric,
            "VA": VA_metric,
        }[metric]
        
        
class Losses:
    """
    Helper which contains the chosen losses and computes them on demand
    """
    
    def __init__(self, 
                 loss_names:List[str],
                 num_classes: int,
                 reduction: str = "mean",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None) -> None:
        
        self.loss_names = loss_names
        self.num_classes = num_classes
        self.reduction = reduction
        
        # get the losses
        self.losses = {}
        for loss in loss_names:
            loss_fn = self._get_loss_fn(loss)
            self.losses.update({loss: loss_fn})
         
         
    def get_loss(self, name: str):
        if name not in self.loss_names:
            return None
        else:
            return self.losses[name]
        
    def to_cuda(self, name: str = None):
        
        print("Moving losses to GPU ...")
        
        if name is not None:
            self.losses[name] = self.losses[name].cuda()     
        else: # move all losses to GPU
            for ln in self.loss_names:
                self.losses[ln] = self.losses[ln].cuda()  
            
    def _get_loss_fn(self, name:str):
        
        if name == "ccc":
            return CCCLoss(num_bins=self.num_classes)
        elif name == "mse":
            return MSELoss(num_classes=self.num_classes)
        
        else:
            raise ValueError("loss {} not implemented".format)
        
        
    def compute_single_loss(self, loss_name: str):
        """
        Wrapper that handles the fact that valence and arousal are separate quantities, thus the output must be a dict with 2 entries.
        Splits the predictions into num_classes for valence and num_classes for arousal. Always assumes valence comes first.
        VA losses append the affect dim to the loss name.
        Any other loss returns a dict with a single entry, key matching the loss name.
        """
        
        if loss_name not in self.loss_names:
            raise ValueError
        
        if loss_name.lower() == "cce":
            
            def VA_losses(pred, target):
                losses =  {
                    "cce_valence": self.losses["cce"](pred[..., :self.num_classes], target[..., 0]), 
                    "cce_arousal": self.losses["cce"](pred[..., self.num_classes:], target[..., 1])
                }
                losses.update({"cce": 0.5 * losses["cce_valence"] + 0.5 *  losses["cce_arousal"]})
                
                return losses
                
            loss_func = VA_losses  
            
        elif loss_name.lower() == "ccc":
            
            def VA_losses(pred, target):
                losses = {
                    "ccc_valence": self.losses["ccc"](pred[..., :self.num_classes], target[..., 0]),
                    "ccc_arousal": self.losses["ccc"](pred[..., self.num_classes:], target[..., 1])
                }
                losses.update({"ccc": 0.5 * losses["ccc_valence"] + 0.5 * losses["ccc_arousal"]})
                
                return losses
            
            loss_func = VA_losses    
            
        # TODO implement further losses
            
        return loss_func 

        
class ComboLoss(nn.Module):
    """
    Helper which combines a MSE and CCC loss
    """        
    def __init__(self, w_ccc=0.5, w_mse=0.5) -> None:
        super().__init__()
        
        self.w_ccc = w_ccc
        self.w_mse = w_mse
        self.cccloss = CCCLoss(num_bins=1)
        self.mseloss = MSELoss(num_classes=1, reduction="mean")
        
    
    def forward(self, pred, target):
        
        ccc = CCCLoss(pred, target)
        mse = MSELoss(pred, target)
        
        loss = self.w_ccc * ccc + self.w_mse * mse
        
        return loss
            

class CustomCrossEntropyLoss(nn.Module):
    """
    Wraps around a cross entropy loss for calculating on a sequence
    """
    def __init__(self, num_classes=24, weight=None) -> None:
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.num_classes = num_classes
        
    def forward(self, pred, target):
        
        assert pred.size(-1) == self.num_classes, "Predictions should have {} classes, has {}".format(self.num_classes, pred.size(-1))
        #assert target.size(-1) == self.num_classes, "Targets should have {} classes, has {}".format(self.num_classes, target.size(-1))
        
        # batch and sequence dimension combined
        pred = pred.view(-1, self.num_classes)
        #target = target.view(-1, self.num_classes)
        target = target.view(-1)
        
        return self.loss(pred, target)
        
        