### Simple differential expression tests

import sys, io, os, gzip, copy, re, time, pickle, argparse
import numpy as np
import pandas as pd
import scipy.stats
from scipy.sparse import *
from datetime import datetime
from random import choices

import anndata as ad
import diffxpy.api as de

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
parser.add_argument('--max_qval_pairwise_output', default=1e-3, type=float, help='')
parser.add_argument('--min_fold_pairwise_output', default=2, type=float, help='')
parser.add_argument('--bulk_only', action='store_true')
parser.add_argument('--skip_bulk', action='store_true')


args = parser.parse_args()

qcut=args.max_qval_pairwise_output
fcut=args.min_fold_pairwise_output
info = pd.read_csv(args.gene_info, sep='\t', header=0)

key = args.key
if key not in info.columns:
    with gzip.open(args.input, 'rt') as rf:
        header = rf.readline().strip().split('\t')
    ct_header = [x for x in info.columns if info[x].dtype in [int, float]]
    ct_header = [x for x in ct_header if x in header]
    if len(ct_header) == 0:
        sys.exit("Gene info and input files do not contain matching count measure")
    key = ct_header[0]
    print(key)

info = info[info[key] > args.min_ct_per_gene]
info.index = info.gene.values
tt = info[key].sum()

test_list = {}
test_list['gene_info'] = info

mtx = pd.read_csv(args.model,sep='\t',header=0)
factor_header = []
for x in mtx.columns:
    y = re.match('^[A-Za-z]+_\d+$', x)
    if y:
        factor_header.append(y.group(0))
mtx = mtx[mtx.gene.isin(info.gene.values)]
mtx['Weight'] = mtx[factor_header].sum(axis = 1)
mtx.sort_values(by = 'Weight', ascending=False, inplace=True)
mtx.drop_duplicates(subset='gene', keep='first', inplace=True)
M = mtx.shape[0]
K = len(factor_header)
print(f"Testing {M} genes over {K} factors")

kv_str=[str(k) for k in range(K)]
for k in range(K):
    mtx.iloc[:, 1:] /= mtx.iloc[:, 1:].sum()

for k in range(K):
    info[str(k)] = 0


grp = pd.read_csv(args.label,sep='\t',header=0)
grp = grp[grp[key] > args.min_ct_per_unit]
grp['j'] = grp.j.astype(str)
grp.index = grp.j.values
pmtx = grp.loc[:, ["Topic_"+str(k) for k in range(K)]]
pmtx.rename(columns = {"Topic_"+str(k) : str(k) for k in range(K)}, inplace=True)
grp = grp.loc[:, ['j','topK','topP',key]]
grp.sort_values(by='topP', ascending=False, inplace=True)
for k in range(K):
    if grp.topK.eq(k).sum() > args.max_group_size:
        indx = choices(grp.index[grp.topK.eq(k)], weights=grp.loc[grp.topK.eq(k), 'topP'].values, k=args.max_group_size)
        grp = grp.loc[ (~grp.topK.eq(k))|(grp.index.isin(indx)), :]

kept_unit = set(grp.j.values)
print(f"Kept {len(kept_unit)} units")

df = pd.DataFrame()
adt = {args.unit_col_name:str, 'gene':str, key:int}
for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=200000, header=0, usecols=[args.unit_col_name,"gene",key], dtype=adt):
    chunk.rename(columns = {args.unit_col_name:'j'},inplace=True)
    chunk = chunk.loc[chunk.gene.isin(info.gene.values), :]
    for k in range(K): # Count for bulk test
        chunk = chunk[chunk.j.isin(pmtx.index)]
        v = pmtx.loc[chunk.j.values, str(k)].values * chunk[key].values
        ct = pd.DataFrame({"gene":chunk.gene.values, "Count":v}).groupby(by = "gene").agg({'Count':sum})
        for i,v in ct.iterrows():
            info.loc[i, str(k)] += v['Count']
    chunk = chunk.loc[chunk.j.isin(kept_unit), :]
    df = pd.concat([df, chunk])
    print(df.shape[0])
print(f"Read data ({df.shape[0]}")

v = info.loc[:, [str(k) for k in range(K)]].sum(axis = 1).values
gene_scale = np.zeros(info.shape[0])
gene_scale[v>0] = info.loc[v>0, key].values / v[v>0]
total_k = np.zeros(K, dtype=int)
for k in range(K):
    info.loc[:, str(k)] = info.loc[:, str(k)].values * gene_scale
    info[str(k)] = np.around(info[str(k)], 0).astype(int)
    indx=info[str(k)] > info[key]
    info.loc[indx, str(k)] = info.loc[indx, key]
    total_k[k] = info[str(k)].sum()

print(f"{len(df.j.unique())} x {info.shape[0]})")
print(total_k)

### Bulk
if not args.skip_bulk:
    res=[]
    for name, v in info.iterrows():
        for k in range(K):
            if total_k[k] <= 0:
                continue
            tab=np.zeros((2,2))
            tab[0,0]=v[str(k)]
            tab[0,1]=v[key]-tab[0,0]
            tab[1,0]=total_k[k]-tab[0,0]
            tab[1,1]=tt-total_k[k]-v[key]+tab[0,0]
            fd=tab[0,0]/total_k[k]/tab[0,1]*(tt-total_k[k])
            tab = np.around(tab, 0).astype(int) + 1
            chi2, p, dof, ex = scipy.stats.chi2_contingency(tab, correction=False)
            res.append([name,k,chi2,p,fd,v[key],v['gene_id']])

    chidf=pd.DataFrame(res,columns=['gene','factor','Chi2','pval','FoldChange','gene_total','gene_id'])
    chidf=chidf.loc[(chidf.pval<1e-3)&(chidf.FoldChange>1), :].sort_values(by=['factor','FoldChange'],ascending=[True,False])
    outf = args.outpref + ".bulk_chisq.tsv"
    chidf.to_csv(outf,sep='\t',float_format="%.2e",index=False)

if args.bulk_only:
    sys.exit()

### pseudo-sc
### Test based on real data
flt=df.groupby(by="gene",as_index=False).agg({key:sum})
brc=df.groupby(by="j",as_index=False).agg({key:sum})
brc=brc.merge(right = grp, on = 'j', how = 'inner')
brc['topK'] = brc.topK.astype(str)
df=df[df.j.isin(brc.j.values)]

brc_dict={x:i for i,x in enumerate(brc.j.values)}
flt_dict={x:i for i,x in enumerate(flt.gene.values)}
indx_row = [ brc_dict[x] for x in df['j']]
indx_col = [ flt_dict[x] for x in df['gene']]
m = flt.shape[0]
n = brc.shape[0]
print(f"Subset DGE to {n} x {m}")
cmtx = coo_matrix((df[key].values, (indx_row, indx_col)), shape=(n, m)).tocsr()


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


qmtx = test_vr.qval.squeeze().T
fmtx = test_vr.log2_fold_change().squeeze().T
tab = pd.DataFrame(qmtx, columns=test_vr.groups)
tab['gene'] = test_vr.gene_ids
qmtx_mask = copy.copy(qmtx)
qmtx_mask[(fmtx < 0)] = 1
tab['qMin'] = qmtx_mask.min(axis = 1)
tab['factor_qMin_Up'] = [test_vr.groups[x] for x in qmtx_mask.argmin(axis=1)]
tab = tab[['gene','qMin','factor_qMin_Up'] + kv_str]
tab.rename(columns = {x:'q_'+x for x in kv_str}, inplace=True)
df=copy.copy(tab)

tab = pd.DataFrame(fmtx, columns=test_vr.groups)
tab['gene'] = test_vr.gene_ids
tab['log2fcMax'] = tab[test_vr.groups].max(axis = 1)
tab['factor_fc_Up'] = [test_vr.groups[x] for x in fmtx.argmax(axis=1)]
df = df.merge(right = tab[['gene','log2fcMax','factor_fc_Up']+kv_str], on = 'gene', how = 'inner')
df.rename(columns = {x:'fc_'+x for x in kv_str}, inplace=True)

df.sort_values(by = ['qMin'], inplace=True)
df = df.merge(right = info, on = 'gene', how = 'left')
for c in ['log2fcMax'] + ['fc_'+x for x in kv_str]:
    df[c] = df[c].map(lambda x : "%.2f" % x)
outf = args.outpref + ".Real_OvR_Wilcoxon.qval.tsv"
df.to_csv(outf,sep='\t',float_format="%.2e",index=False)

# Output pairwise results to table
df = pd.DataFrame()
for g0 in test_pw.groups:
    for g1 in test_pw.groups:
        if g1 == g0:
            continue
        try:
            tab = copy.copy(test_pw.summary_pairs(g0, g1))
        except:
            print(f"Group pair ({g0},{g1}) not in pairwise test result")
            continue
        tab = tab[(tab.qval < qcut) & (tab.log2fc > np.log2(fcut))]
        if tab.shape[0] == 0:
            continue
        tab.sort_values(by = 'qval', inplace=True)
        header =list(tab.columns)
        tab['Factor1'] = g0
        tab['Factor2'] = g1
        tab = tab.loc[:, ['Factor1','Factor2'] + header]
        df = pd.concat([df, tab])
df = df.merge(right = info, on = 'gene', how = 'left')
df['log2fc'] = df.log2fc.map(lambda x : "%.2f" % x)
outf = args.outpref + ".Real_Pairwise_Wilcoxon.summary.tsv"
df.to_csv(outf,sep='\t',float_format="%.2e",index=False)


### Test based on the model
print("Simulate data from posterior model")
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

qmtx = test_vr.qval.squeeze().T
fmtx = test_vr.log2_fold_change().squeeze().T
tab = pd.DataFrame(qmtx, columns=test_vr.groups)
tab['gene'] = test_vr.gene_ids
qmtx_mask = copy.copy(qmtx)
qmtx_mask[(fmtx < 0)] = 1
tab['qMin'] = qmtx_mask.min(axis = 1)
tab['factor_qMin_Up'] = [test_vr.groups[x] for x in qmtx_mask.argmin(axis=1)]
tab = tab[['gene','qMin', 'factor_qMin_Up'] + test_vr.groups]
tab.rename(columns = {x:'q_'+str(x) for x in test_vr.groups}, inplace=True)
df=copy.copy(tab)

tab = pd.DataFrame(fmtx, columns=test_vr.groups)
tab['gene'] = test_vr.gene_ids
tab['log2fcMax'] = tab[test_vr.groups].max(axis = 1)
tab['factor_fc_Up'] = [test_vr.groups[x] for x in fmtx.argmax(axis=1)]
df = df.merge(right = tab[['gene','log2fcMax','factor_fc_Up']+test_vr.groups], on = 'gene', how = 'inner')
df.rename(columns = {x:'fc_'+x for x in kv_str}, inplace=True)

df.sort_values(by = ['qMin'], inplace=True)
df = df.merge(right = info, on = 'gene', how = 'left')
for c in ['log2fcMax'] + ['fc_'+x for x in kv_str]:
    df[c] = df[c].map(lambda x : "%.2f" % x)
outf = args.outpref + ".Simu_OvR_Wilcoxon.qval.tsv"
df.to_csv(outf,sep='\t',float_format="%.2e",index=False)


# Output pairwise results to table
df = pd.DataFrame()
for g0 in test_pw.groups:
    for g1 in test_pw.groups:
        if g1 == g0:
            continue
        tab = copy.copy(test_pw.summary_pairs(g0, g1))
        tab = tab[(tab.qval < qcut) & (tab.log2fc > np.log2(fcut))]
        if tab.shape[0] == 0:
            continue
        tab.sort_values(by = 'qval', inplace=True)
        header =list(tab.columns)
        tab['Factor1'] = g0
        tab['Factor2'] = g1
        tab = tab.loc[:, ['Factor1','Factor2'] + header]
        df = pd.concat([df, tab])
df = df.merge(right = info, on = 'gene', how = 'left')
df['log2fc'] = df.log2fc.map(lambda x : "%.2f" % x)
outf = args.outpref + ".Simu_Pairwise_Wilcoxon.summary.tsv"
df.to_csv(outf,sep='\t',float_format="%.2e",index=False)


outf = args.outpref + ".tests.p"
pickle.dump(test_list, open( outf, "wb" ) )

print("Finish")
