import sys, io, os, gzip, glob, copy, re, time, pickle, warnings, argparse
from collections import defaultdict,Counter
import numpy as np
import pandas as pd

from scipy.sparse import *
from sklearn.preprocessing import normalize
from sklearn import random_projection

import scipy.optimize
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anchor_fn import simplex_vertices
from utilt import gen_even_slices

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

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--feature', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--K', type=int, help='')
parser.add_argument('--R', type=int, default = 1, help='')
parser.add_argument('--anchor_min_ct', type=int, default=500, help='')
parser.add_argument('--key', type=str, default='gn', help='')
parser.add_argument('--epsilon', type=float, default = 0.2, help='')
parser.add_argument('--thread', type=int, default=1, help='')
parser.add_argument('--debug', action='store_true', help='')
args = parser.parse_args()

K = args.K

with open(args.input, 'rb') as rf:
    Q = np.load(rf, allow_pickle=True)
    gene_list = np.load(rf,  allow_pickle=True)

M = len(gene_list)
ft_dict = {x:i for i,x in enumerate(gene_list)}
feature = pd.read_csv(args.feature, sep='\t')
feature.drop(index=feature.index[~feature.gene.isin(ft_dict)], inplace=True)
feature.index = [ft_dict[x] for x in feature.gene.values]
feature.sort_index(inplace=True)
candi = feature.index[feature[args.key].ge(args.anchor_min_ct)].values
p0 = feature[args.key].values.astype(float)
p0 /= p0.sum()
if args.debug:
    prj_dim = int(4 * np.log(len(candi)) / args.epsilon**2)
    print(M, Q.shape, len(candi), prj_dim)

rng = np.random.default_rng(int(time.time() % 100000000) )
result = []
for r in range(args.R):
    if args.debug:
        idx = simplex_vertices(Q[candi, :], args.epsilon, K, \
                verbose = 1, info = feature[['gene', args.key]],
                seed = rng.integers(low = 1, high = 2**31))
    else:
        idx = simplex_vertices(Q[candi, :], args.epsilon, K, \
                            seed = rng.integers(low = 1, high = 2**31))
    result.append(candi[idx])

anchor_list = pd.DataFrame()
for r,v in enumerate(result):
    sub = copy.copy(feature.loc[v, :] )
    sub["Cluster"] = range(K)
    sub["Run"] = r
    anchor_list = pd.concat([anchor_list, sub])

anchor_list.to_csv(args.output + ".anchor.tsv", sep='\t', index=False)

model = pd.DataFrame()
for r,v in enumerate(result):
    print(f"Recover model for run {r}")
    anchor_list = feature.loc[v, 'gene'].values
    if args.thread > 1:
        idx_list = [idx for idx in gen_even_slices(M, args.thread) ]
        results = Parallel(n_jobs=args.thread)(\
            delayed(wrap_optim)(Q[idx, :], Q[v, :]) for idx in idx_list)
        W = np.zeros((M, K))
        for i,v in enumerate(results):
            W[idx_list[i], :] = v
    else:
        W = wrap_optim(Q, Q[v, :])
    beta = np.diag(p0) @ W
    beta = normalize(beta, axis = 0, norm = 'l1')
    sub = pd.DataFrame({"gene": gene_list, "Run": r})
    sub = pd.concat([sub, pd.DataFrame(beta, columns = np.arange(K).astype(str)) ], axis = 1)
    model = pd.concat([model, sub])

model.to_csv(args.output + ".model.tsv.gz", sep='\t', float_format = "%.4e", index=False)
