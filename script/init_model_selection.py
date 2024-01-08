import sys, io, os, gzip, glob, copy, re, time, warnings, argparse, logging
from collections import defaultdict,Counter
import numpy as np
import pandas as pd

from scipy.sparse import *
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kloptim import optim_kl
from utilt import gen_even_slices, scale_to_prob


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--models', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--key', type=str, default='gn', help='')
parser.add_argument('--iter', type=int, default=3, help='')
parser.add_argument('--thread', type=int, default=1, help='')
parser.add_argument('--debug', action='store_true', help='')
args = parser.parse_args()

key = args.key
thread = args.thread
niter = args.iter
logging.basicConfig(level= getattr(logging, "INFO", None))

df = pd.DataFrame()
for chunk in pd.read_csv(args.input, sep='\t', usecols = ['random_index','X','Y','gene',key], dtype={'random_index':str}, chunksize=500000):
    if chunk.random_index.iloc[-1][:2] != "00":
        df = pd.concat([df, chunk[chunk.random_index.str.contains('^00') & chunk[key].ge(1)] ])
        break
    df = pd.concat([df, chunk[chunk[key].ge(1)] ])

models = pd.read_csv(args.models, sep='\t')
Rlist = sorted(models.Run.unique())
Klist = [x for x in models.columns if x.isnumeric()]
R = len(Rlist)
K = len(Klist)
print(f"Read {R} models with {K} components")

gene_list = models.loc[models.Run.eq(Rlist[0]), "gene"].values
ft_dict = {x:i for i,x in enumerate(gene_list)}
models["gene_id"] = models.gene.map(ft_dict)
models.sort_values(by = ["Run", "gene_id"], inplace=True)
M = len(ft_dict)
df.drop(index = df.index[~df.gene.isin(ft_dict)], inplace=True)

brc = df.groupby(by = 'random_index').agg({key:sum}).reset_index()
brc.drop(index = brc.index[brc[key].lt(100)], inplace=True)
N = brc.shape[0]
brc.index = np.arange(N)
bc_dict = {x:i for i,x in enumerate(brc.random_index)}
brc['j'] = brc.random_index.map(bc_dict)
df.drop(index = df.index[~df.random_index.isin(bc_dict)], inplace=True)
df['j'] = df.random_index.map(bc_dict)
df.drop(columns = "random_index", inplace=True)
brc = brc.merge(right = df[['j','X','Y' ]].drop_duplicates(subset='j'), on = 'j', how = 'left')

mtx = coo_array((df[key].values, (df.j.values, df.gene.map(ft_dict))), shape=(N, M)).tocsr()
xsum = mtx.sum(axis = 1).reshape((-1, 1))
print(f"Read {N} units with {M} genes")

model_beta = {}
model_theta = {}
model_score = np.zeros((R, niter))
for ri, r in enumerate(Rlist):
    Ht = np.array(models.loc[models.Run.eq(r), Klist] ).T
    Wt = normalize(np.random.beta(1,1,size=(N,K)), axis = 1, norm = 'l1') * xsum
    Ht0 = Ht.copy()
    obj0 = np.inf
    it = 0
    while it < niter:
        t0 = time.time()
        results = Parallel(n_jobs=thread)(\
                    delayed(optim_kl)(mtx[idx,:], Wt[idx, :], Ht, tol=1e-4, verbose=args.debug, method="SLSQP") for idx in gen_even_slices(N, thread) )
        t1 = time.time() - t0
        Wt = np.vstack([x[0] for x in results])
        objv = np.mean(np.hstack([x[1] for x in results]))
        logging.info(f"{r}, {it} - update W {t1/60:.2f} min, obj {objv:.3f}")

        t0 = time.time()
        results = Parallel(n_jobs=thread)(\
                    delayed(optim_kl)(mtx[:, idx].T, Ht[:,idx].T, Wt.T, tol=1e-6, verbose=args.debug, method="SLSQP") for idx in gen_even_slices(M, thread) )
        t1 = time.time() - t0
        Ht = np.hstack([x[0].T for x in results])
        objv = np.mean(np.hstack([x[1] for x in results])) * (M/N)
        logging.info(f"{r}, {it} - update H {t1/60:.2f} min, obj {objv:.3f}")
        model_score[ri, it] = objv
        if objv > obj0:
            logging.warning(f"{r}, {it} - Objective value increased from {obj0:.3f} to {objv:.3f}, roll back to saved checkpoint")
            Ht = Ht0
            Wt = normalize(np.random.beta(1,1,size=(N,K)), axis = 1, norm = 'l1') * xsum
        else:
            obj0 = objv
            Ht0 = Ht
        it += 1

    model_theta[r], model_beta[r] = scale_to_prob(Wt, Ht)

best_r = Rlist[np.argmin(model_score[:, -1]) ]

f = args.output + ".model_score.tsv"
pd.DataFrame(model_score.T, index = np.arange(niter), columns = Rlist).to_csv(f, sep='\t', index=True, float_format='%.4e')

beta = model_beta[best_r]
beta = pd.DataFrame(beta, columns = Klist, index = gene_list)
f = args.output + ".init_model.tsv.gz"
beta.to_csv(f, sep='\t', index=True, float_format='%.4e')


theta = model_theta[best_r]
brc = brc[["j","X","Y",key]].merge(right = pd.DataFrame(theta, columns = Klist, index = brc.index), left_index=True, right_index=True)
f = args.output + ".init_fit.tsv.gz"
brc.to_csv(f, sep='\t', index=False, float_format='%.4e')
