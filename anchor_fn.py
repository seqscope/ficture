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

def find_farthest_point(points, basis = None):
    if basis is None:
        if issparse(points):
            distances = scipy.sparse.linalg.norm(points, axis=1)
        else:
            distances = np.linalg.norm(points, axis=1)
        return np.argmax(distances)
    assert basis.shape[1] == points.shape[1]
    u, s, vt = np.linalg.svd(basis, full_matrices=False)
    prj = points @ vt.T @ vt # n x M
    distances = np.linalg.norm(points - prj, axis=1)
    return np.argmax(distances)

def simplex_vertices(Q, epsilon, K, verbose = 0, info = None, seed = None, fixed_vertices = None):
    V = Q.shape[0]
    prj_dim = int(4 * np.log(V) / epsilon**2)
    while prj_dim > V:
        prj_dim = prj_dim // 2

    # Project V points to a random 4*log(V)/epsilon^2 dimensional subspace
    Qprj = random_projection.SparseRandomProjection(prj_dim, random_state = seed).fit_transform(Q)

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
    for i in range(fixed+1, K):
        far_idx = find_farthest_point(Qprj, basis=Qprj[S_indices, :])
        S_indices.append(far_idx)
    if verbose:
        print("Greedy init:")
        print(info.iloc[S_indices, :] )

    # Replace each point in S with the farthest point from the (K-1)-dimensional span of S\{v_i}
    final_indices = copy.copy(S_indices)
    it = 0
    while it < 10:
        n_change = 0
        for i in range(fixed, K):
            temp_indices = final_indices[:i] + final_indices[i+1:]
            far_idx = find_farthest_point(Qprj, basis=Qprj[temp_indices, :])
            if far_idx not in S_indices:
                n_change += 1
                if verbose:
                    print(it, n_change, info.iloc[S_indices[i], :].values, info.iloc[far_idx, :].values )
            final_indices[i] = far_idx
        it += 1
        if verbose:
            print(f"{it}-th loo refit, {n_change} changes")
        if n_change < 2:
            break
        S_indices = copy.copy(final_indices)

    return final_indices
