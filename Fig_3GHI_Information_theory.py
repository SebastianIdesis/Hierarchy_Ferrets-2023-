#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 05 2023

Author: Sebastian Idesis. sebastian.idesis@gmail.com

Adapted from Thomas Varley

Simple functions for calculating determinism, degeneracy,
and temporal mutual information from time t to t+1.

Original Referencies:
    
    Varley, T. F., Denny, V., Sporns, O., & Patania, A. (2021). 
    Royal Society Open Science, 8(6), 201971. https://doi.org/10.1098/rsos.201971

    Klein, B., & Hoel, E. (2020). 
    Complexity, 2020, e8932526. https://doi.org/10.1155/2020/8932526

"""

import numpy as np 
from scipy.stats import entropy

from os.path import dirname, join as pjoin
import scipy.io as sio


#data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
mat_fname = 'tr_st3_an3.mat'
import mat73

# Making a random TPM to demonstrate
n_states = 5
tpm = np.random.randn(n_states,n_states)
tpm = np.abs(tpm)



def determinism(X):
    """
    Determinism is how predicable is the future given the present.
    
    Parameters
    ----------
    X : np.ndarray (states x states)
        A TPM - every row must sum to 1.

    Returns
    -------
    float
        The determinism of the network.

    """
    
    ents = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        ents[i] = entropy(X[i], base=2)
    
    avg_ent = np.mean(ents)
    
    return (np.log2(X.shape[0]) - avg_ent) / np.log2(X.shape[0])


def degeneracy(X):
    """
    Degeneracy is how much information about the past is lost when 
    multiple states run together.

    Parameters
    ----------
    X : np.ndarray (states x states)
        A TPM - every row must sum to 1.

    Returns
    -------
    float
        The degeneracy of the network.

    """
    
    ent_avg = entropy(np.mean(X, axis=0), base=2)
    
    return (np.log2(X.shape[0]) - ent_avg) / np.log2(X.shape[0])


def temporal_mi(X, past_state):
    """
    

    Parameters
    ----------
    X : np.ndarray (states x states)
        A TPM - every row must sum to 1.
    past_state : np.ndarray (states)
        The initial probability of each state (the prior.)

    Returns
    -------
    float
        The mutual information between the past state and the future.

    """
    X_norm = X / X.sum()
    
    future_state = np.matmul(past_state, X)
    
    h_past = entropy(past_state, base=2)
    h_future = entropy(future_state, base=2)
    h_joint = entropy(X_norm.flatten(), base=2)
    
    return h_past + h_future - h_joint


# Determinism - Degeneracy = Temporal MI / log2(N)
# When the input distribution is maximum entropy!

det = determinism(tpm)
deg = degeneracy(tpm)
mi = temporal_mi(tpm, [1/n_states for i in range(n_states)])

print(det - deg == mi / np.log2(n_states))

