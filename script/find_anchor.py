import sys, io, os, gzip, glob, copy, re, time, logging, warnings, argparse
import numpy as np
import pandas as pd

from scipy.sparse import *
from sklearn.preprocessing import normalize

import kneed
import scipy.optimize
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anchor_fn import simplex_vertices, recover_kl, prj_eval
from utilt import gen_even_slices


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--feature', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--output_tmp', type=str, default='', help='')
parser.add_argument('--K', type=int, help='')
parser.add_argument('--R', type=int, default = 1, help='')
parser.add_argument('--candidate', type=str, default='', help='')
parser.add_argument('--fixed', type=str, nargs="*", default=[], help='')
parser.add_argument('--anchor_min_ct', type=int, default=-1, help='')
parser.add_argument('--search_cutoff_qt', type=float, nargs="*", default=[0.2, 0.3, 0.4, 0.5, 0.6], help='')
parser.add_argument('--search_cutoff_upper', type=float, default = .99, help='')
parser.add_argument('--key', type=str, default='gn', help='')
parser.add_argument('--epsilon', type=float, default = 0.2, help='')
parser.add_argument('--n_anchor_per_cluster', type=int, default=-1, help='')
parser.add_argument('--max_anchor_per_cluster', type=int, default=10, help='')
parser.add_argument('--min_anchor_per_cluster', type=int, default=3, help='')
# parser.add_argument('--anchor_list', type=str, default='', help='')
parser.add_argument('--thread', type=int, default=1, help='')
parser.add_argument('--recoverKL', action='store_true', help='')
parser.add_argument('--recoverKL_allmodel', action='store_true', help='')
parser.add_argument('--debug', action='store_true', help='')
args = parser.parse_args()
logging.basicConfig(level= getattr(logging, "INFO", None))

K = args.K
cut_upper = args.search_cutoff_upper

with open(args.input, 'rb') as rf:
    Q = np.load(rf, allow_pickle=True)
    gene_list = np.load(rf,  allow_pickle=True)

M0 = len(gene_list)
ft_dict = {x:i for i,x in enumerate(gene_list)}
feature = pd.read_csv(args.feature, sep='\t')
feature.sort_values(by = args.key, ascending = False, inplace = True)
feature.drop_duplicates(subset = 'gene', inplace = True, keep = 'first')
feature.drop(index=feature.index[~feature.gene.isin(ft_dict)], inplace=True)
if len(feature) != len(gene_list):
    warnings.warn(f"Feature file does not contain all the genes in the co-expression matrix. Only genes in the feature file will be used.")
feature.drop(index=feature.index[feature[args.key].lt(10)], inplace=True)
# Ignore genes not in the feature file, subset Q
kept_idx = [ft_dict[x] for x in gene_list if x in feature.gene.values]
Q = Q[kept_idx, :][:, kept_idx]
gene_list = gene_list[kept_idx]
ft_dict = {x:i for i,x in enumerate(gene_list)}
M = len(ft_dict)
logging.info(f"Read co-expression with {M0} genes, kept {M}")

feature.index = [ft_dict[x] for x in feature.gene.values]
feature.sort_index(inplace=True)
fixed_idx = [ft_dict[x] for x in args.fixed if x in ft_dict]
if len(fixed_idx) > 0:
    logging.info(f"{len(fixed_idx)}/{len(args.fixed)} pre-selected anchors are found in the input file")

ct_cutoffs = []
ct_upper = int(np.quantile(feature[args.key], q = cut_upper) )
if args.anchor_min_ct > 0:
    ct_cutoffs = [args.anchor_min_ct]
else:
    ct_cutoffs = np.quantile(feature[args.key], q = args.search_cutoff_qt).astype(int)
    logging.info(f"Search for anchors with different total count cutoffs: {ct_cutoffs}")
n_cutoffs = len(ct_cutoffs)

rng = np.random.default_rng(int(time.time() % 100000000) )
anchors_full = []
scores_full = []
final_score = []
kept_idx = feature.index.values
if os.path.isfile(args.candidate):
    with open(args.candidate, 'r') as rf:
        kept_list = [x.strip().split()[0] for x in rf.readlines()]
    kept_idx = np.array([ft_dict[x] for x in kept_list if x in ft_dict] )
    miss = [x for x in fixed_idx if x not in kept_idx]
    logging.info(f"Read {len(kept_idx)} candidate genes")
    if len(miss) > 0:
        logging.info(f"{len(miss)} pre-selected anchors are not among the input candidates.")
        kept_idx = np.append(kept_idx, miss)
kept_idx = kept_idx[feature.loc[kept_idx, args.key].lt(ct_upper)]
print(feature.loc[:, args.key].describe( percentiles=[.25, .5, .75, .95, .98] ))
print(feature.loc[kept_idx, args.key].describe( percentiles=[.25, .5, .75, .95, .98] ))
for j, cutoff in enumerate(ct_cutoffs):
    candi = kept_idx[feature.loc[kept_idx, args.key].ge(cutoff)]
    candi_map = {x:i for i,x in enumerate(candi)}
    fixed_vtx = [candi_map[x] for x in fixed_idx]
    logging.info(f"Min/max count of candidate anchor genes {cutoff}/{ct_upper}, start finding anchors among {len(candi)} candidates.")
    if args.epsilon > 0:
        prj_dim = int(4 * np.log(len(candi)) / args.epsilon**2)
        logging.info(f"Random projection to {prj_dim} dimensions.")
    else:
        logging.info(f"Random projection is inactive, anchor discovery will be deterministic. --R is ignored.")
        args.R = 1
    anchors = []
    scores = []
    if args.thread > 1 and args.R > 1 and not args.debug:
        results = Parallel(n_jobs=args.thread)( \
            delayed(simplex_vertices)(Q[candi, :], args.epsilon, K,\
                fixed_vertices = fixed_vtx, \
                seed = rng.integers(low = 1, high = 2**31)) for i in range(args.R))
        for r, y in enumerate(results):
            idx, score = y
            anchors.append(candi[idx])
            scores.extend([[cutoff, r] + x for x in score])
    else:
        for r in range(args.R):
            if args.debug:
                idx, score = simplex_vertices(Q[candi, :], args.epsilon, K,\
                        verbose = 1, info = feature.loc[candi, ['gene', args.key]],\
                        fixed_vertices = fixed_vtx, \
                        seed = rng.integers(low = 1, high = 2**31))
            else:
                idx, score = simplex_vertices(Q[candi, :], args.epsilon, K,\
                        fixed_vertices = fixed_vtx, \
                        seed = rng.integers(low = 1, high = 2**31))
            anchors.append(candi[idx])
            scores.extend([[cutoff, r] + x for x in score])
    for r, idx in enumerate(anchors):
        var_e, rec_e2, rec_e1 = prj_eval(Q, Q[idx, :])
        final_score.append([cutoff, r, K, var_e, rec_e2, rec_e1])
        logging.info(f"Evaluating with all genes - {cutoff}, {r}: {var_e:.4f}, {rec_e2:.4f}, {rec_e1:.4f}")
    scores_full.extend(scores)
    anchors_full.extend(anchors)

pd.DataFrame(scores_full, columns = ["Cutoff", "Run", "k","Variance_explained","Reconstruction_error_l2", "Reconstruction_error_l1", "OrgSpace"]).to_csv(args.output + ".anchor_trace.tsv", sep='\t', index=False)


kept_idx = feature[feature[args.key].lt(ct_upper)].index.values
Qsym = np.minimum(Q, Q.T)
np.fill_diagonal(Qsym, 0)
Qsym = Qsym[:, kept_idx]
anchor_df = pd.DataFrame()
for r,v in enumerate(anchors_full):
    cutoff = final_score[r][0]
    run = final_score[r][1]
    candi_list = []
    logging.info(f"Anchor set {r}: " + ", ".join(gene_list[v]))
    for k in range(K):
        idx = [v[k]]
        if args.n_anchor_per_cluster > 0:
            idx += list(kept_idx[np.argsort(-Qsym[v[k], :])[:args.n_anchor_per_cluster] ] )
        else:
            m = max(100, args.max_anchor_per_cluster)
            v_idx = np.argsort(-Qsym[v[k], :])[:m]
            y = Qsym[v[k], :][v_idx]
            kn = kneed.KneeLocator(x=np.arange(m), y=y, S=1, curve="convex", direction="decreasing")
            m = max(kn.knee + 1, args.min_anchor_per_cluster)
            m = min(m, args.max_anchor_per_cluster)
            idx = [v[k]] + list(kept_idx[v_idx[:m] ])
            logging.info(f"Anchor set {r}, cluster {k}, add {m} genes (knee {kn.knee + 1}): " + ",".join(gene_list[idx]) )
        candi_list.append(list(gene_list[idx]))
    to_rm = set()
    for k in range(K-1):
        if k in to_rm:
            continue
        for l in range(k+1, K):
            cap = set(candi_list[k]) & set(candi_list[l])
            thres = min(len(candi_list[k]), len(candi_list[l])) * .5
            if len(cap) >= thres:
                candi_list[k] = list(np.unique([candi_list[k][0], candi_list[l][0]] + list(cap) ) )
                to_rm.add(l)
                logging.info(f"Anchor set {r}, merge cluster {k} and {l} with {len(cap)} common genes: " + ",".join(list(cap)) )
    candi_list = [x for i,x in enumerate(candi_list) if i not in to_rm]
    for k,x in enumerate(candi_list):
        anchor_df = pd.concat([anchor_df, \
            pd.DataFrame({"Cutoff": cutoff, "Run": run, "Cluster": k, "gene": x})])
    logging.info(f"Anchor set {r}, kept {len(candi_list)} clusters")
anchor_df.to_csv(args.output + ".anchor.tsv", sep='\t', index=False)

final_score = pd.DataFrame(final_score, columns = ["Cutoff", "Run", "K", "Variance_explained", "Reconstruction_error_l2", "Reconstruction_error_l1"])
final_score.to_csv(args.output + ".anchor_score.tsv", sep='\t', index=False)
print(final_score[["Cutoff", "Variance_explained", "Reconstruction_error_l2"]])

# Pick one best anchor set
tab = final_score.groupby(by = "Cutoff").agg({"Variance_explained": max}).reset_index()
tab.sort_values(by = "Cutoff", inplace = True)
i = np.argmax(tab.Variance_explained.values)
i = max(0, i-1)
cutoff0 = tab.Cutoff.iloc[i]
tab = final_score.loc[final_score.Cutoff.eq(cutoff0), :]
r0 = tab.loc[tab.Variance_explained.idxmax(), "Run"]
logging.info(f"Final anchor set if generated with cutoff {cutoff0}")


tab = anchor_df.loc[anchor_df.Cutoff.eq(cutoff0) & anchor_df.Run.eq(r0), :]
tab.to_csv(args.output + ".anchor_picked.tsv", sep='\t', index=False)
anchor_gene_info = tab.groupby(by = "Cluster").agg({'gene': lambda x : ", ".join(x) })
anchor_gene_info.sort_index(inplace=True)
anchor_gene_info.to_csv(args.output + ".anchor_picked.list.tsv", sep='\t', index=True, header=False)

if not args.recoverKL:
    sys.exit(0)

clst = sorted(list(tab.Cluster.unique()))
k = len(clst)
logging.info(f"Recover model for selected anchor set (cutoff {cutoff0} run {r0}) with {k} clusters")
aks = []
for i,l in enumerate(clst):
    v = tab.loc[tab.Cluster.eq(l), 'gene'].values
    aks.append([ft_dict[x] for x in v])
    if args.debug:
        print(i, ", ".join( f"{x[0]} ({x[1]})" for x in feature.loc[aks[-1], ['gene', args.key]].values ) )

p0 = feature[args.key].values.astype(float)
p0 /= p0.sum()
beta = recover_kl(Q, aks, p0, thread = args.thread, debug = args.debug)
sub = pd.DataFrame({"gene": gene_list})
sub = pd.concat([sub, pd.DataFrame(beta, columns = np.arange(k).astype(str)) ], axis = 1)
sub.to_csv(args.output + ".model.tsv.gz", sep='\t', float_format = "%.4e", index=False)
sub.to_csv(args.output + f".{cutoff0}_{r0}.model.tsv.gz", sep='\t', float_format = "%.4e", index=False)


Clist = sorted(list(anchor_df.Cutoff.unique()))
Rlist = sorted(list(anchor_df.Run.unique()))
if not args.recoverKL_allmodel or (len(Rlist) == 1 and len(Clist) == 1):
    sys.exit(0)

if args.output_tmp == '' or not os.path.exists(os.path.dirname(args.output_tmp)):
    print("Path to store andidate models is invalid, will use the same path as for the final model.")
    args.output_tmp = args.output

os.system(f"cp {args.output}.model.tsv.gz {args.output_tmp}.{cutoff0}_{r0}.model.tsv.gz")

for cutoff in Clist:
    for r in Rlist:
        if cutoff == cutoff0 and r == r0:
            continue
        tab = anchor_df.loc[anchor_df.Cutoff.eq(cutoff) & anchor_df.Run.eq(r), :]
        clst = sorted(list(tab.Cluster.unique()))
        k = len(clst)
        logging.info(f"Recover model for cutoff {cutoff} run {r} with {k} clusters")
        aks = []
        for i,l in enumerate(clst):
            v = tab.loc[tab.Cluster.eq(l), 'gene'].values
            aks.append([ft_dict[x] for x in v])
            if args.debug:
                print(i, ", ".join( f"{x[0]} ({x[1]})" for x in feature.loc[aks[-1], ['gene', args.key]].values ) )
        beta = recover_kl(Q, aks, p0, thread = args.thread, debug = args.debug)
        sub = pd.DataFrame({"gene": gene_list})
        sub = pd.concat([sub, pd.DataFrame(beta, columns = np.arange(k).astype(str)) ], axis = 1)
        sub.to_csv(args.output_tmp + f".{cutoff}_{r}.model.tsv.gz", sep='\t', float_format = "%.4e", index=False)
