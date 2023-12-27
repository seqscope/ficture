import sys, io, os, gzip, glob, copy, re, time, warnings, argparse, logging
from collections import defaultdict,Counter
import numpy as np
import pandas as pd

from scipy.sparse import *
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilt import gen_slices_from_list, make_mtx_from_dge
from online_lda import OnlineLDA
from lda_minibatch import Minibatch


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--anchors', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--output_tmp', type=str, help='')
parser.add_argument('--key', type=str, default='gn', help='')
parser.add_argument('--init_max_unit', type=int, default=300, help='')
parser.add_argument('--init_min_unit', type=int, default=5, help='')
parser.add_argument('--init_min_ct', type=int, default=100, help='')
parser.add_argument('--bsize', type=int, default=256, help='')
parser.add_argument('--thread', type=int, default=1, help='')
parser.add_argument('--debug', action='store_true', help='')
args = parser.parse_args()

key = args.key
thread = args.thread
bsize = args.bsize
logging.basicConfig(level= getattr(logging, "INFO", None))

anchors = pd.read_csv(args.anchors, sep='\t')
Rlist = sorted(anchors.Run.unique())
R = len(Rlist)
print(f"Read {R} sets of anchors")

df, feature, brc, mtx, ft_dict, bc_dict = make_mtx_from_dge(args.input,\
    min_ct_per_feature = 50, min_ct_per_unit = 50,\
    feature_white_list = anchors.gene.unique(),\
    unit = "random_index", key = key)

feature["Weight"] = feature[key] / feature[key].sum()
N, M = mtx.shape
anchors.drop(index = anchors.index[~anchors.gene.isin(ft_dict)], inplace=True)
anchors["gene_id"] = anchors.gene.map(ft_dict)
xsum = mtx.sum(axis = 1).reshape((-1, 1))
kept_unit = brc.index[brc[key].ge(args.init_min_ct)]

models = {}
for r in Rlist:
    tab = anchors.loc[anchors.Run.eq(r), :]
    clusters = sorted(list(tab.Cluster.unique()))
    clist = {x:i for i,x in enumerate(clusters)}
    K = len(clusters)
    assign = coo_array((np.ones(len(tab)), (tab.Cluster.map(clist), tab.gene_id)), shape=(K, M)).tocsr()
    min_enrich = 2/assign.sum(axis = 1).max()
    prior_score = ((normalize(mtx, axis = 1, norm = 'l1') / \
                   feature.Weight.values.reshape((1, -1)) \
                   ) @ normalize(assign, axis = 1, norm = 'l1').T).toarray()
    tab = pd.DataFrame(prior_score, columns = clusters)
    tab["ID"] = brc.index
    tab = tab.loc[tab.ID.isin(kept_unit), :]
    tab = tab.melt(id_vars = "ID", value_vars = clusters, var_name = "Cluster", value_name = "Score")
    tab["Score_sum"] = tab.groupby(by = "ID").Score.transform(sum)
    tab["Score_adj"] = np.clip(2 * tab.Score / tab.Score_sum, 0, np.inf)
    init_c = []
    init_b = []
    kept_c = []
    criterion = tab.Score_adj.gt(min_enrich)
    for i,v in enumerate(clusters):
        sub = tab.loc[tab.Cluster.eq(v) & criterion, :].iloc[:args.init_max_unit, :]
        if sub.shape[0] < args.init_min_unit:
            continue
        init_c.extend([i] * sub.shape[0])
        init_b.extend(sub.ID.values)
        kept_c.append(i)
    init_assign = coo_array((np.ones(len(init_c), dtype=int), (init_c, init_b)), shape=(K, N)).tocsr()
    if len(kept_c) < K:
        K = len(kept_c)
        clusters = [clusters[i] for i in kept_c]
        init_assign = init_assign[kept_c, :]
    init_ct = (init_assign @ mtx).toarray().T
    init_ct = pd.DataFrame(init_ct, columns = clusters)
    init_ct["gene"] = feature.gene.values
    f = args.output_tmp + f".{r}.init_prior.tsv.gz"
    init_ct.to_csv(f, sep='\t', index=False)
    models[r] = {"anchor": tab, "init_prior": init_ct, "init_n":init_assign.sum(axis = 1)}
    print(r, K, len(init_b), models[r]["init_n"])

idx_shuffle = np.random.permutation(N)
idx_list = [idx for idx in gen_slices_from_list(idx_shuffle, bsize) ]
model_score = []
model_lambda = {}
for r in Rlist:
    prior = np.clip(np.array(models[r]["init_prior"].drop(columns = "gene")).T, 0, np.inf)
    K = prior.shape[0]
    n = models[r]["init_n"].sum()
    lda = OnlineLDA(vocab=feature.gene.values,K=K,N=n,alpha=1/K,eta=1/np.sqrt(M),tau0=int(n//bsize+10),thread=args.thread,tol=1e-4,verbose=0)
    lda.init_global_parameter(prior)
    logl_rec = []
    for i, idx in enumerate(idx_list):
        logl = lda.update_lambda(Minibatch(mtx[idx, :]))
        logl_rec.append([len(idx), logl])
        print(r, i, logl)
    logl_rec = np.array(logl_rec)
    avg_logl = np.mean((logl_rec[:, 0]/bsize) * logl_rec[:, 1])
    model_score.append(avg_logl )
    model_lambda[r] = lda._lambda
    print(r, avg_logl)

r_best = Rlist[np.argmax(model_score)]

f = args.output_tmp + f".{r_best}.init_prior.tsv.gz"
newf = args.output + ".init_prior.tsv.gz"
os.system(f"cp {f} {newf}")

f = args.output + ".model_scores.tsv"
pd.DataFrame({"Run":Rlist, "Score":model_score}).to_csv(f, sep='\t', index=False, float_format='%.4e')

K = model_lambda[r_best].shape[0]
f = args.output + ".model.tsv"
pd.DataFrame(model_lambda[r_best].T, index = pd.Index(feature.gene.values, name='gene'), columns = range(K)).to_csv(f, sep='\t', index=True, float_format='%.2f')

lda = OnlineLDA(vocab=feature.gene.values,K=K,N=N,alpha=.1/K,thread=args.thread,tol=1e-4,verbose=0)
lda.init_global_parameter(model_lambda[r_best])
theta = lda.transform(mtx)

res = pd.DataFrame(theta, columns = np.arange(K), index = brc.index)
res["topK"] = theta.argmax(axis=1)
res["topP"] = theta.max(axis=1)
brc = brc[["X","Y",key]].merge(right =res, left_index=True, right_index=True)
f = args.output + ".init_fit.tsv.gz"
brc.X = brc.X.map(lambda x: "%.2f" % x)
brc.Y = brc.Y.map(lambda x: "%.2f" % x)
brc.to_csv(f, sep='\t', index=False, float_format='%.5f')
