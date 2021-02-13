
# Adapted implementation of multivariate noise normalisation from Guggenmos et al. (2018)
# entitled "Multivariate pattern analysis for MEG: A comparison of dissimilarity measures".
# This version implements the epoch-version specifically and is adapted from their
# code and further edited by me. The Euclidean-norm addition is credited to Máté Aller.
# Please see original paper for discussion of algorithm.
#
# Alex Murphy <murphyalex(dot)gmail.com>

import copy
import numpy as np
from scipy.linalg import fractional_matrix_power, norm
from sklearn.discriminant_analysis import _cov

def multivariate_noise_norm(X, y, euclNorm=False, flatten=True):
    """ Multivariate Noise Normalisation (Epoch version)
    
    Args:
        X: Unnormalised NumPy array of shape (n_batch, n_channels, n_time)
        y: 1D vector of class labels
    Returns:
        X_new: Normalised NumPy array (same size as input dimensions)
    """    

    X_ = copy.deepcopy(X) # Do not change underlying array
    assert id(X_) != id(X)

    n_classes = len(set(y))
    _, n_channels, n_time = X_.shape
    sigma_ = np.empty((n_classes, n_channels, n_channels))

    for c in range(n_classes):
        # compute sigma for each time point, then average across time
        # 'auto' parameter of _cov selects Ledoit-Wolf optimal shrinkage method
        cov_mat_x_time = [_cov(X_[y==c, :, t], shrinkage='auto') for t in range(n_time)]
         # average cov mat over class-specific time points
        sigma_[c] = np.mean(cov_mat_x_time, axis=0)
        
    sigma = sigma_.mean(axis=0)  # average across conditions
    sigma_inv = fractional_matrix_power(sigma, -0.5) # whitening matrix
    X_new = (X_.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2) # compute normalisation
    
    if euclNorm: # calculate L2 norm in a trial-by-channel basis
        for i in range(n_channels):
            for j in range(n_time):
                X_new[:,i,j] = X_new[:,i,j] / norm(X_new[:,i,j])
                
    if flatten:
        X_new = X_new.reshape((len(X_new), -1))

    return X_new