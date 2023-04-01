import sys, io, os, re, time, copy, warnings, pickle, argparse
import numpy as np
import pandas as pd
import scipy.stats
from scipy.sparse import *
from sklearn.preprocessing import normalize
from joblib.parallel import Parallel, delayed
from scipy.optimize import minimize, Bounds, LinearConstraint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilt import gen_even_slices

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--marker', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--thread', type=int, default = 1, help='')
args = parser.parse_args()
eps = 1e-8

obj = pickle.load(open(args.input, 'rb'))
ft_dict = {x:i for i,x in enumerate(obj['feature'])}
M = len(ft_dict)
Q = normalize(obj['Q'], norm='l1', axis=1)
background = Q.sum(axis = 0) / Q.shape[0]

marker = pd.read_csv(args.marker, sep='\t', names=["cell_type","gene"])
marker = marker.loc[marker.gene.isin(ft_dict), :]
marker["INDEX"] = marker.gene.map(ft_dict)
anchor_name = list(marker.cell_type.unique() )
K = len(anchor_name)

anchor = np.zeros((K+1, M))
for k in range(K):
    indx = marker.loc[marker.cell_type.eq(anchor_name[k]), 'INDEX'].values
    amtx = Q[indx, :].toarray()
    anchor[k, ] = scipy.stats.hmean(amtx, axis = 0)
anchor[K, :] = background
anchor = np.clip(anchor, 1e-8, 1)

K += 1
anchor_name.append("Background")

cnstr = LinearConstraint(np.ones(K), lb=1-eps, ub=1+eps)
bnds = Bounds(lb = 0, ub = 1)
x0 = np.ones(K) / K

def obj_fun(x, *args):
    yhat = x.reshape((1, -1)) @ anchor # 1xM
    return - np.dot(np.log(yhat.reshape(-1)), args[0].reshape(-1))

def grad_fun(x, *args):
    yhat = x.reshape((1, -1)) @ anchor # 1xM
    return - (np.multiply(anchor, 1/yhat.reshape((1, -1))) @ args[0].reshape((-1, 1))).reshape(-1)

def optim(idx):
    res_mtx = np.zeros((len(idx), K))
    for i,j in enumerate(idx):
        res = minimize(fun=obj_fun, x0=x0, args=(Q.getrow(j).toarray()),\
                       bounds=bnds, constraints=cnstr,\
                       jac=grad_fun, method='trust-constr')
        res_mtx[i, :] = res.x
    return res_mtx

with Parallel(n_jobs=args.thread, verbose=0) as parallel:
    result = parallel(delayed(optim)(idx)
    for idx in gen_even_slices(M, args.thread))

result = pd.DataFrame(np.vstack(result), columns = anchor_name)
result["gene"] = obj['feature']
result.to_csv(args.output, sep='\t', index=False, header=True, float_format="%.5f")
