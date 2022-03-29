#! /usr/bin/python

''' helper functions '''
import numpy as np
import pandas as pd
import copy
from scipy.special import gammaln, psi, logsumexp, expit, logit
from sklearn.preprocessing import normalize
import scipy.optimize
import sklearn.cluster


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


def match_factors(mtx1, mtx2, c1, n, cmap, mode='beta'):
    """
    Harmonize a pair of LDA results
    """
    k1 = mtx1.shape[0]
    k2 = mtx2.shape[0]
    assert mtx1.shape[1] == mtx2.shape[1], "Invalid input matrices"

    # Try to match two sets of factors
    Q = np.zeros((k1, k2))
    if mode == 'beta':
        beta1 = mtx1 / mtx1.sum(axis = 1)[:, np.newaxis]
        beta2 = mtx2 / mtx2.sum(axis = 1)[:, np.newaxis]
        for k in range(k1):
            q,r = scipy.optimize.nnls(beta2.transpose(), beta1[k, :].transpose())
            Q[k, ] = q
            indx2 = [(-Q[:,k]).argsort() for k in range(k2)]
            indx1 = [(-Q[k,:]).argsort() for k in range(k1)]
    else:
        for k in range(k1):
            q,r = scipy.optimize.nnls(mtx2.transpose(), mtx1[k, :].transpose())
            Q[k, ] = q
            indx2 = [(-Q[:,k]).argsort() for k in range(k2)]
            indx1 = [(-Q[k,:]).argsort() for k in range(k1)]
            Q = Q * mtx2.sum(axis = 1)[np.newaxis, :]
            Q = Q / (Q.sum(axis = 1)[:, np.newaxis])

    model = sklearn.cluster.SpectralBiclustering(n_clusters=(k1,k1), method="bistochastic").fit(Q+1e-5)
    prio = (-Q.sum(axis = 0)).argsort() # sorted from most consistent to less consistent
    step = n//k1

    dup = [1] * len(c1)
    c1_ref = copy.copy(c1)
    c2 = [-1]*k2
    for i,k in enumerate(prio):
        c2[k] = c1_ref[indx2[k][0]]
        c1_ref[indx2[k][0]] += step - int(step / (2*dup[indx2[k][0]]))
        dup[indx2[k][0]] += 1

    indx1 = np.array(c1).argsort()
    v1 = mtx1.sum(axis = 1)
    v1 = v1/v1.sum()
    v1 = v1[indx1]

    indx2 = np.array(c2).argsort()
    v2 = mtx2.sum(axis = 1)
    v2 = v2/v2.sum()
    v2 = v2[indx2]

    st = []
    ed = []
    st_y = []
    ed_y = []
    cl = []

    st_y += [0] + list(np.cumsum(v1)[:-1])
    ed_y += list(np.cumsum(v1))
    cl += [c1[x] for x in indx1]

    offset = 0
    st += [k1+offset]*k1
    ed += [k1+offset]*k1

    st_y += [0] + list(np.cumsum(v2)[:-1])
    ed_y += list(np.cumsum(v2))
    cl += [c2[x] for x in indx2]

    st += [k2-offset]*k2
    ed += [k2-offset]*k2

    df = pd.DataFrame({'xst':st,'yst':st_y,'xed':ed,'yed':ed_y,'Color':cl})

    mid1 = []
    mid2 = []
    v = np.cumsum(v1)
    mid1+=list(v - v1*0.5)
    v = np.cumsum(v2)
    mid2+=list(v - v2*0.5)

    xst = []
    xed = []
    yst = []
    yed = []
    sl = []
    cl = []
    offset = 0.3
    # for ii,i in enumerate(indx1):
    for ii,i in enumerate(indx1):
        for jj,j in enumerate(indx2):
            if Q[i,j] > 1e-3:
                xst.append(k2-offset)
                xed.append(k1+offset)
                yst.append(mid2[jj])
                yed.append(mid1[ii])
                sl.append(Q[i, j])
                cl.append(c1[i])
    df_arrow = pd.DataFrame({'xst':xst,'yst':yst,'xed':xed,'yed':yed,'Weight':sl,'Color':cl})
    return df, df_arrow, c2
