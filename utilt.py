#! /usr/bin/python

''' helper functions '''
import numpy as np
from scipy.special import gammaln, psi, logsumexp, expit, logit
from sklearn.preprocessing import normalize

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha+1e-3) - psi(np.sum(alpha+1e-3, 1))[:, np.newaxis])

def pg_mean(b,c=0):
    if np.isscalar(c) and c == 0:
        return b/4
    if not np.isscalar(c):
        if np.isscalar(b):
            b = np.ones(c.shape[0]) * b
        v = b/4
        indx = (c != 0)
        v[indx] = b[indx]/2/c[indx] * np.tanh(c[indx]/2)
        return v
    else:
        return (b/2/c * np.tanh(c/2))

def real_to_sb(mtx):
    assert len(mtx.shape) == 2, "Invalid matrix"
    n, K = mtx.shape
    phi = expit(mtx)
    s = phi[:, 0]
    for k in range(1, K-1):
        phi[:, k] = phi[:, k] * (1. - s)
        s +=  phi[:, k]
    phi[:, K - 1] = 1. - s
    phi = np.clip(phi, 1e-8, 1.-1e-8)
    phi = normalize(phi, norm='l1', axis=1)
    return phi

def sb_to_real(phi):
    assert len(phi.shape) == 2, "Invalid matrix"
    phi = np.clip(phi, 1e-8, 1.-1e-8)
    phi = normalize(phi, norm='l1', axis=1) * (1.-1e-6)
    n, K = phi.shape
    mtx = np.zeros((n, K))
    mtx[:, 0] = logit(phi[:, 0])
    s = phi[:, 0]
    for k in range(1, K):
        mtx[:, k] = logit(phi[:, k] / (1. - s) )
        s += phi[:, k]
    return mtx
