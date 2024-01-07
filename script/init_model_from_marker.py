import sys, io, os, gzip, glob, copy, re, time, warnings, argparse, logging
from collections import defaultdict,Counter
import numpy as np
import pandas as pd
import scipy.stats
from scipy.sparse import *
from sklearn.preprocessing import normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilt import gen_slices_from_list, make_mtx_from_dge
from online_lda import OnlineLDA
from lda_minibatch import Minibatch
from itertools import combinations
from scipy.optimize import linear_sum_assignment

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
parser.add_argument('--epoch', type=int, default=1, help='')
parser.add_argument('--thread', type=int, default=1, help='')
parser.add_argument('--debug', action='store_true', help='')
args = parser.parse_args()

key = args.key
thread = args.thread
bsize = args.bsize
unit="random_index"
logging.basicConfig(level= getattr(logging, "INFO", None))

anchors = pd.read_csv(args.anchors, sep='\t')
Rlist = sorted(anchors.Run.unique())
R = len(Rlist)
print(f"Read {R} sets of anchors")
# Check if anchor sets are redundant (highly overlap)
ncls = anchors.groupby(by = 'Run').agg({'Cluster':lambda x : len(set(x))}).Cluster
matches_info = {}
for r1, r2 in combinations(Rlist, 2):
    clusters_r1 = anchors[anchors['Run'] == r1].groupby('Cluster')['gene'].apply(set)
    clusters_r2 = anchors[anchors['Run'] == r2].groupby('Cluster')['gene'].apply(set)
    cost_matrix = -1 * np.array([[len(g1 & g2) for g2 in clusters_r2] for g1 in clusters_r1])
    thre_matrix = -np.clip(np.array([[min(len(g1),len(g2)) for g2 in clusters_r2] for g1 in clusters_r1]) *.3, 1, None)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    shared_features = -cost_matrix[row_ind, col_ind].sum()
    shared_clusters = (cost_matrix[row_ind, col_ind] < thre_matrix[row_ind, col_ind]).astype(int).sum()
    matches_info[(r1, r2)] = {'matches': list(zip(clusters_r1.index[row_ind], \
                                                  clusters_r2.index[col_ind])),
                              'shared_features': shared_features, 'shared_clusters':shared_clusters}
rm_r = []
for i,r1 in enumerate(Rlist[:-1]):
    if r1 in rm_r:
        continue
    for r2 in Rlist[i+1:]:
        if matches_info[(r1,r2)]['shared_clusters'] == min(ncls[r1],ncls[r2]):
            rm_r.append(r2)
rm_r = set(rm_r)
if len(rm_r) > 0:
    print(f"Remove {len(rm_r)} redundant anchor sets")
    anchors.drop(index=anchors.index[anchors.Run.isin(rm_r)], inplace=True)
    Rlist = [r for r in Rlist if r not in rm_r]
    R = len(Rlist)


feature, brc, mtx, ft_dict, bc_dict = make_mtx_from_dge(args.input,\
    min_ct_per_feature = 50, min_ct_per_unit = 50,\
    feature_white_list = anchors.gene.unique(),\
    unit = unit, key = key, epoch = args.epoch)

feature["Weight"] = feature[key] / feature[key].sum()
N, M = mtx.shape
anchors.drop(index = anchors.index[~anchors.gene.isin(ft_dict)], inplace=True)
anchors["gene_id"] = anchors.gene.map(ft_dict)
xsum = mtx.sum(axis = 1).reshape((-1, 1))
kept_unit = brc.index[brc[key].ge(args.init_min_ct)]
logging.info(f"Read {N} units and {M} features")
logging.info(f"Kept {len(kept_unit)}/{N} units with at least {args.init_min_ct} counts")
pval_cutoff = .1

models = {}
for r in Rlist:
    tab = anchors.loc[anchors.Run.eq(r), :]
    clusters = sorted(list(tab.Cluster.unique()))
    clist = {x:i for i,x in enumerate(clusters)}
    K = len(clusters)
    # meta pvalue for each cluster
    cauchy_p = np.ones((N, K))
    for k in range(K):
        idx = tab.loc[tab.Cluster.eq(k), "gene_id"].values
        mtx_short = mtx[:, idx].toarray()
        p0 = feature.loc[idx, "Weight"].values
        w = np.sqrt(p0.max() / p0)
        w = w / w.sum()
        # p-value for each marker gene
        sf = scipy.stats.binom.sf(mtx_short, xsum, p0.reshape((1, -1))) + .5 * scipy.stats.binom.pmf(mtx_short, xsum, p0.reshape((1, -1)))
        tscore = (np.tan((.5 - sf) * np.pi ) * w.reshape((1, -1)) ).sum(axis = 1)
        cauchy_p[:, k] = .5 - np.arctan2(tscore, np.ones(N)) / np.pi
    tab = pd.DataFrame(cauchy_p, columns = clusters)
    tab["ID"] = brc.index
    tab = tab.melt(id_vars = "ID", value_vars = clusters, var_name = "Cluster", value_name = "pval")
    tab["pval_min"] = tab.groupby(by = "ID").pval.transform(min)
    tab["Rank"] = tab.groupby(by = "ID").pval.rank(ascending = True, method = "max")
    tab.sort_values(by = ['Cluster', 'pval'], inplace=True)
    init_c = []
    init_b = []
    kept_c = []
    max_p = []
    for i,k in enumerate(clusters):
        sub = tab.loc[tab.Cluster.eq(k) & tab.Rank.lt(2), :]
        if sub.shape[0] < args.init_min_unit:
            sub = pd.concat([sub, tab.loc[tab.Cluster.eq(k) & tab.Rank.eq(2) & \
                            (tab.pval_2nd/tab.pval_min).le(2), :]])
            if len(sub) < args.init_min_unit:
                continue
            sub = sub.sort_values(by = "pval")
        idx = sub.loc[sub.pval.lt(pval_cutoff), "ID"].values[:args.init_max_unit]
        if len(idx) < args.init_min_unit:
            idx = sub.ID.iloc[:args.init_min_unit].values
        init_c.extend([i] * len(idx))
        init_b.extend(idx)
        kept_c.append(i)
        max_p.append(sub.loc[sub.ID.isin(idx), "pval"].max())
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
    print(r, K, models[r]["init_n"], len(init_b), np.around(max_p, 3))



idx_shuffle = np.random.permutation(N)
init_N = N//args.epoch
idx_list = [idx for idx in gen_slices_from_list(idx_shuffle[:(N//args.epoch)], bsize) ] # Does not use all units to pick a model
model_snapshot = {}
model_score = []
# model_lambda = {}
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
    # model_lambda[r] = lda._lambda
    model_snapshot[r] = lda
    print(r, avg_logl)

r_best = Rlist[np.argmax(model_score)]

tab = anchors.loc[anchors.Run.eq(r_best), :]
anchor_gene_info = tab.groupby(by = "Cluster").agg({'gene': lambda x : ", ".join(x) })
anchor_gene_info.sort_index(inplace=True)
f = args.output + ".model_anchors.tsv"
anchor_gene_info.to_csv(f, sep='\t', index=True, header=False)

f = args.output_tmp + f".{r_best}.init_prior.tsv.gz"
newf = args.output + ".init_prior.tsv.gz"
os.system(f"cp {f} {newf}")

f = args.output + ".model_scores.tsv"
pd.DataFrame({"Run":Rlist, "Score":model_score}).to_csv(f, sep='\t', index=False, float_format='%.4e')

lda = model_snapshot[r_best]
if args.epoch > 1:
    idx_list = [idx for idx in gen_slices_from_list(idx_shuffle[(N//args.epoch):], bsize) ]
    for i, idx in enumerate(idx_list):
        logl = lda.update_lambda(Minibatch(mtx[idx, :]))
        print(i, logl)

K = lda._lambda.shape[0]
f = args.output + ".model_matrix.tsv.gz"
pd.DataFrame(lda._lambda.T, index = pd.Index(feature.gene.values, name='gene'), columns = range(K)).to_csv(f, sep='\t', index=True, float_format='%.2f')

one_pass = brc.epoch.eq(brc.epoch.iloc[0])
idx = brc.index[one_pass].values
theta = lda.transform(mtx[idx, :])
res = pd.DataFrame(theta, columns = np.arange(K), index = idx)

# theta = lda.transform(mtx)
# res = pd.DataFrame(theta, columns = np.arange(K), index = brc.index)

res["topK"] = theta.argmax(axis=1)
res["topP"] = theta.max(axis=1)
brc = brc[["X","Y",key]].merge(right =res, left_index=True, right_index=True, how = 'right')
brc.X = brc.X.map(lambda x: "%.2f" % x)
brc.Y = brc.Y.map(lambda x: "%.2f" % x)
f = args.output + ".fit_result.tsv.gz"
brc.to_csv(f, sep='\t', index=False, float_format='%.5f')
