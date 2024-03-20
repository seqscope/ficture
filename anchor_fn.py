"""
Find anchor genes from co-expression matrix
"""
import sys, io, os, gzip, glob, copy, re, time, pickle, warnings
from collections import defaultdict,Counter

import numpy as np
import pandas as pd

from scipy.sparse import *
import scipy.sparse.linalg
from sklearn.preprocessing import normalize
from sklearn import random_projection
import scipy.optimize
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utilt import gen_even_slices


def find_farthest_point(points, basis = None, eval_basis = False):
    if basis is None:
        if issparse(points):
            distances = scipy.sparse.linalg.norm(points, axis=1)
        else:
            distances = np.linalg.norm(points, axis=1)
        return np.argmax(distances)
    assert basis.shape[1] == points.shape[1]
    u, s, vt = np.linalg.svd(basis, full_matrices=False)
    pts_prj = points @ vt.T
    pts_hat = pts_prj @ vt # n x M
    distances = np.linalg.norm(points - pts_hat, axis=1)
    idx = np.argmax(distances)
    if not eval_basis:
        return idx
    var_tot = np.var(points, axis = 0).sum()
    var_explained = np.var(pts_prj, axis = 0).sum()
    rec_error2 = np.sqrt((distances ** 2).sum() )
    rec_error1 = np.mean(np.abs(points - pts_hat).sum(axis = 1) )

    return idx, var_explained / var_tot, rec_error2, rec_error1

def prj_eval(pts, vt, orthonormal = False):
    if orthonormal is False:
        u, s, vt = np.linalg.svd(vt, full_matrices=False)
    pts_prj = pts @ vt.T
    pts_hat = pts_prj @ vt # n x M
    var_tot = np.var(pts, axis = 0).sum()
    var_explained = np.var(pts_prj, axis = 0).sum()
    rec_error2 = np.linalg.norm(pts - pts_hat)
    rec_error1 = np.mean(np.abs(pts - pts_hat).sum(axis = 1) )
    return var_explained / var_tot, rec_error2, rec_error1

def simplex_vertices(Q, epsilon, K, verbose = 0, info = None, seed = None, fixed_vertices = None):
    V, M = Q.shape

    if epsilon > 0:
        prj_dim = int(4 * np.log(V) / epsilon**2)
        while prj_dim > M:
            prj_dim = prj_dim // 2
        Qprj = random_projection.SparseRandomProjection(prj_dim, random_state = seed).fit_transform(Q)
    else:
        Qprj = Q

    fixed = 0
    S_indices = []
    if fixed_vertices is not None:
        S_indices = [x for x in fixed_vertices if x < V and x >= 0]
        fixed = len(S_indices)
        assert fixed == len(fixed_vertices), f"{len(fixed_vertices) - fixed} input vertices are out of range"
        assert fixed < K, f"{fixed} input vertices are given, but K = {K}"
    if fixed == 0:
        # Initialize S with the farthest point from the origin
        far_idx = find_farthest_point(Qprj)
        S_indices = [far_idx]

    # Iteratively add the farthest point from the span of S
    scores = []
    for i in range(fixed, K):
        far_idx, var_e, rec_e2, rec_e1 = find_farthest_point(Qprj, basis=Qprj[S_indices, :], eval_basis=True)
        scores.append([i, var_e, rec_e2, rec_e1, 0])
        S_indices.append(far_idx)
        if verbose:
            print(i, info.iloc[far_idx, :].values, f"{var_e:.4f}", f"{rec_e2:.4f}", f"{rec_e1:.4f}")
    var_e, rec_e2, rec_e1 = prj_eval(Q, Q[S_indices, :])
    scores.append([K, var_e, rec_e2, rec_e1, 1])

    # Replace each point in S with the farthest point from the (K-1)-dimensional span of S\{v_i}
    final_indices = copy.copy(S_indices)
    it = 0
    while it < 10:
        n_change = 0
        for i in range(fixed, K):
            temp_indices = final_indices[:i] + final_indices[i+1:]
            far_idx = find_farthest_point(Qprj, basis=Qprj[temp_indices, :])
            if far_idx not in final_indices:
                n_change += 1
                if verbose:
                    print(it, n_change, info.iloc[S_indices[i], :].values, info.iloc[far_idx, :].values )
                final_indices[i] = far_idx
        it += 1
        if n_change == 0:
            break
        var_e, rec_e2, rec_e1 = prj_eval(Q, Q[final_indices, :])
        scores.append([K, var_e, rec_e2, rec_e1, 1])
        if verbose:
            print(f"{it}-th loo refit, {n_change} changes, updated score: {var_e:.4f}, {rec_e2:.4f}, {rec_e1:.4f}")
        if n_change < 2 and it > 2:
            break
        S_indices = copy.copy(final_indices)

    return final_indices, scores






def objective(x, p, H):
    q = np.clip(np.dot(x, H), 1e-10, np.inf)
    return np.sum(np.where(p != 0, - p * np.log(q), 0))

def objective_gradient(x, p, H):
    q = np.clip(np.dot(x, H), 1e-10, np.inf)
    return -((H * p.reshape((1, -1)))/q.reshape((1, -1))).sum(axis = 1).squeeze()

cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1},  # sum to 1
        {'type': 'ineq', 'fun': lambda x: x})  # non-negative

def optim_kl(p,H):
    K = H.shape[0]
    w0 = np.ones(K) / K
    w = scipy.optimize.minimize(objective, w0, method='SLSQP', \
            jac=objective_gradient, constraints=cons, args=(p,H))
    return w.x

def wrap_optim(X,H):
    N = X.shape[0]
    K = H.shape[0]
    w0 = np.ones(K) / K
    res = np.zeros((N, K))
    for i in range(X.shape[0]):
        w = scipy.optimize.minimize(objective, w0, method='SLSQP', \
                jac=objective_gradient, constraints=cons, args=(X[i, :],H))
        res[i, :] = w.x
    return res

def recover_kl(Q, anchors, p0, thread = 1, debug = 0):
    assert Q.shape[0] == len(p0)
    M = Q.shape[1]
    k = len(anchors)
    H = np.zeros((k, M))
    for i,idx in enumerate(anchors):
        w = p0[idx] / p0[idx].sum()
        w = 1. / np.sqrt(np.clip(w, .1/len(idx), None) )
        w /= w.sum()
        if debug:
            print(i, len(idx), w.round(3))
        H[i, :] = (Q[idx, :] * w.reshape((-1, 1))).sum(axis = 0)
    H = normalize(H, axis = 1, norm = 'l1')
    if thread > 1:
        idx_list = [idx for idx in gen_even_slices(M, thread) ]
        results = Parallel(n_jobs=thread)(\
            delayed(wrap_optim)(Q[idx, :], H) for idx in idx_list)
        W = np.zeros((M, k))
        for i,v in enumerate(results):
            W[idx_list[i], :] = v
    else:
        W = wrap_optim(Q, H)
    W = np.clip(W, 1e-8, 1-1e-8)
    W = normalize(W, axis = 1, norm = 'l1')
    beta = np.diag(p0) @ W
    beta = normalize(beta, axis = 0, norm = 'l1')
    return beta
