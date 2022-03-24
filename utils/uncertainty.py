from scipy.stats import entropy
import numpy as np

"""
Helper file for computing uncertainty
"""


def calc_uncertainty(probabilities: np.ndarray):
    """
    Calculates aleatoric and epistemic uncertainty
    :param probabilities: An Array of shape NxNc, N predictions for Nc classes
    :return: aleatoric and epistemic uncertainty
    """

    nclasses = probabilities.shape[1]
    uncertainty = entropy(probabilities.mean(axis=0), base=2, axis=-1) / np.log2(nclasses)
    aleatoric_uncertainty = np.stack([entropy(p, base=2, axis=-1) for p in probabilities], axis=0).mean(axis=0)/ np.log2(nclasses)
    epistemic_uncertainty = uncertainty - aleatoric_uncertainty
    if isinstance(epistemic_uncertainty, float):
        assert epistemic_uncertainty >= 0
    else:
        assert epistemic_uncertainty.min() >= 0

    return aleatoric_uncertainty, epistemic_uncertainty

