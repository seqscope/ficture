import sys, io, os, copy, re, time, pickle, argparse
import numpy as np
import pandas as pd
from scipy.sparse import *
from datetime import datetime
from random import choices

import anndata as ad
import diffxpy.api as de

# sc.settings.verbosity = 3

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='')
parser.add_argument('--input', type=str, help='')
parser.add_argument('--label', type=str, help='')
parser.add_argument('--outpref', type=str, help='')
parser.add_argument('--gene_info', type=str, help='')
parser.add_argument('--simu_cell_size', default=400, type=int, help='')
parser.add_argument('--simu_group_size', default=500, type=int, help='')
parser.add_argument('--max_group_size', default=1000, type=int, help='')
parser.add_argument('--unit_col_name', default='random_index', type=str, help='')
parser.add_argument('--key', default='gn', type=str, help='')
parser.add_argument('--min_ct_per_unit', default=50, type=int, help='')
parser.add_argument('--min_ct_per_gene', default=50, type=int, help='')
args = parser.parse_args()

info = pd.read_csv(args.gene_info, sep='\t', header=0)
info = info[info[args.key] > args.min_ct_per_gene]

test_list = {}
test_list['gene_info'] = info

mtx = pd.read_csv(args.model,sep='\t',header=0)
mtx = mtx[mtx.gene.isin(info.gene.values)]
M = mtx.shape[0]
K = mtx.shape[1] - 1
print(f"Testing {M} genes over {K} factors")

for k in range(K):
    mtx.iloc[:, 1:] /= mtx.iloc[:, 1:].sum()

rng = np.random.default_rng(int(datetime.now().timestamp()))
cmtx = csr_matrix(([], ([],[])),shape=(0,M),dtype=int)
for k in range(K):
    cmtx = vstack([cmtx, csr_matrix(rng.multinomial(n=args.simu_cell_size,
                                                    pvals=mtx['Factor_'+str(k)].values,
                                                    size=args.simu_group_size) ) ])
data = ad.AnnData(
    X=cmtx.toarray(),
    var = pd.DataFrame(index=mtx.gene.values),
    dtype=int
)
obs = pd.DataFrame({'Type':np.kron(np.arange(K),np.ones(args.simu_group_size,dtype=int))},index=range(K*args.simu_group_size))
for k in range(K):
    obs['K'+str(k)] = np.zeros(K*args.simu_group_size,dtype=int)
    obs.loc[(args.simu_group_size*k):(args.simu_group_size*(k+1)),'K'+str(k)] = 1
obs = obs.astype({x:str for x in obs.columns})
data.obs = obs


### Test based on the model
# One v.s. Rest
t0=time.time()
test_vr = de.test.versus_rest(
    data=cmtx,
    gene_names=mtx.gene.values,
    sample_description=obs,
    grouping='Type',
    test="rank",noise_model=None
)
t1 = time.time() - t0
print(f"One v.s. rest, wilcoxon. {t1:.1f}")
test_list["Simu_OneVSRest_Wilcoxon"] = copy.copy(test_vr)

# Pairwise
t0=time.time()
test_pw = de.test.pairwise(
    data=cmtx.toarray(),
    gene_names=mtx.gene.values,
    sample_description=obs,
    grouping="Type",
    test="rank",lazy=False,noise_model=None
)
t1 = time.time() - t0
print(f"Pairwise, wilcoxon. {t1:.1f}")
test_list["Simu_Pairwise_Wilcoxon"] = copy.copy(test_pw)

tab = pd.DataFrame(test_vr.qval.squeeze().T, columns=test_vr.groups)
tab['gene'] = test_vr.gene_ids
tab['qMin'] = tab[test_vr.groups].min(axis = 1)
tab = tab[['gene','qMin']+test_vr.groups]
df=copy.copy(tab)
tab = pd.DataFrame(test_vr.log2_fold_change().squeeze().T, columns=test_vr.groups)
tab['gene'] = test_vr.gene_ids
tab['log2fcMax'] = tab[test_vr.groups].max(axis = 1)
df = df.merge(right = tab[['gene','log2fcMax']+test_vr.groups], on = 'gene', how = 'inner')
df.sort_values(by = ['qMin'], inplace=True)
df = df.merge(right = info, on = 'gene', how = 'left')
outf = args.outpref + ".Simu_OvR_Wilcoxon.qval.tsv"
df.to_csv(outf,sep='\t',float_format="%.2e",index=False)







### Test based on real data

df=pd.read_csv(args.input, sep='\t',header=0,usecols=[args.unit_col_name,'gene',args.key])
df.rename(columns = {args.unit_col_name:'j'},inplace=True)
df = df[df.gene.isin(info.gene.values)]
res = pd.read_csv(args.label,sep='\t',header=0)
grp = res.loc[res.j.isin(df.j.unique()), ['j','topK','topP']]
grp_sub = pd.DataFrame()
for k in range(K):
    if grp.topK.eq(k).sum() > args.max_group_size:
        indx = choices(grp.index[grp.topK.eq(k)], k=args.max_group_size)
        grp_sub = pd.concat([grp_sub, copy.copy(grp.loc[indx, :]) ])
    else:
        grp_sub = pd.concat([grp_sub, copy.copy(grp.loc[grp.topK.eq(k), :]) ])

df=df[df.j.isin(grp_sub.j.values)]
flt=df.groupby(by="gene",as_index=False).agg({'gn':sum})
df=df[df.gene.isin(flt.gene.values)]
brc=df.groupby(by="j",as_index=False).agg({'gn':sum})
brc=brc[brc[args.key] > args.min_ct_per_unit]
brc=brc.merge(right = grp_sub, on = 'j', how = 'left')
brc['topK'] = brc.topK.astype(str)
df=df[df.j.isin(brc.j.values)]

brc_dict={x:i for i,x in enumerate(brc.j.values)}
flt_dict={x:i for i,x in enumerate(flt.gene.values)}
indx_row = [ brc_dict[x] for x in df['j']]
indx_col = [ flt_dict[x] for x in df['gene']]
M = flt.shape[0]
N = brc.shape[0]
print(f"Subset DGE to {N} x {M}")
cmtx = coo_matrix((df.gn.values, (indx_row, indx_col)), shape=(N, M)).tocsr()

t0=time.time()
test_vr = de.test.versus_rest(
    data=cmtx.toarray(),
    gene_names=flt.gene.values,
    sample_description=brc,
    grouping="topK",
    test="rank",noise_model=None
)
t1 = time.time() - t0
print(f"One v.s. rest, wilcoxon. {t1:.1f}")
test_list["Real_OneVSRest_Wilcoxon"] = test_vr

t0=time.time()
test_pw = de.test.pairwise(
    data=cmtx.toarray(),
    gene_names=flt.gene.values,
    sample_description=brc,
    grouping="topK",
    test="rank",lazy=False,noise_model=None
)
t1 = time.time() - t0
print(f"Pairwise, wilcoxon. {t1:.1f}")
test_list["Real_Pairwise_Wilcoxon"] = test_pw

tab = pd.DataFrame(test_vr.qval.squeeze().T, columns=test_vr.groups)
tab['gene'] = test_vr.gene_ids
tab['qMin'] = tab[test_vr.groups].min(axis = 1)
tab = tab[['gene','qMin']+test_vr.groups]
df=copy.copy(tab)
tab = pd.DataFrame(test_vr.log2_fold_change().squeeze().T, columns=test_vr.groups)
tab['gene'] = test_vr.gene_ids
tab['log2fcMax'] = tab[test_vr.groups].max(axis = 1)
df = df.merge(right = tab[['gene','log2fcMax']+test_vr.groups], on = 'gene', how = 'inner')
df.sort_values(by = ['qMin'], inplace=True)
df = df.merge(right = info, on = 'gene', how = 'left')
outf = args.outpref + ".Real_OvR_Wilcoxon.qval.tsv"
df.to_csv(outf,sep='\t',float_format="%.2e",index=False)





outf = args.outpref + ".tests.p"
pickle.dump(test_list, open( outf, "wb" ) )
