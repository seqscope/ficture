### Simple differential expression tests

import sys, io, os, gzip, copy, re, time, pickle, argparse
import numpy as np
import pandas as pd
import scipy.stats
from scipy.sparse import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--feature', type=str, default='', help='')
parser.add_argument('--feature_label', type=str, default = "gene", help='')
parser.add_argument('--min_ct_per_feature', default=50, type=int, help='')
parser.add_argument('--max_pval_output', default=1e-3, type=float, help='')
parser.add_argument('--min_fold_output', default=1.5, type=float, help='')
args = parser.parse_args()

pcut=args.max_pval_output
fcut=args.min_fold_output
gene_kept = set()
if os.path.exists(args.feature):
    feature = pd.read_csv(args.feature, sep='\t', header=0)
    gene_kept = set(feature[args.feature_label].values )

# Read aggregated count table
info = pd.read_csv(args.input,sep='\t',header=0)
oheader = []
header = []
for x in info.columns:
    y = re.match('^[A-Za-z]*_*(\d+)$', x)
    if y:
        header.append(y.group(1))
        oheader.append(x)
K = len(header)
M = info.shape[0]
reheader = {oheader[k]:header[k] for k in range(K)}
reheader[args.feature_label] = "gene"
info.rename(columns = reheader, inplace=True)
print(f"Read posterior count over {M} genes and {K} factors")

if len(gene_kept) > 0:
    info = info.loc[info.gene.isin(gene_kept), :]
info["gene_tot"] = info.loc[:, header].sum(axis=1)
info = info[info["gene_tot"] > args.min_ct_per_feature]
info.index = info.gene.values
total_umi = info.gene_tot.sum()
total_k = np.array(info.loc[:, [str(k) for k in range(K)]].sum(axis = 0) )
M = info.shape[0]

print(f"Testing {M} genes over {K} factors")

res=[]
for name, v in info.iterrows():
    for k in range(K):
        if total_k[k] <= 0 or v[str(k)] <= 0:
            continue
        tab=np.zeros((2,2))
        tab[0,0]=v[str(k)]
        tab[0,1]=v["gene_tot"]-tab[0,0]
        tab[1,0]=total_k[k]-tab[0,0]
        tab[1,1]=total_umi-total_k[k]-v["gene_tot"]+tab[0,0]
        fd=tab[0,0]/total_k[k]/tab[0,1]*(total_umi-total_k[k])
        tab = np.around(tab, 0).astype(int) + 1
        chi2, p, dof, ex = scipy.stats.chi2_contingency(tab, correction=False)
        res.append([name,k,chi2,p,fd,v["gene_tot"]])

chidf=pd.DataFrame(res,columns=['gene','factor','Chi2','pval','FoldChange','gene_total'])
chidf=chidf.loc[(chidf.pval<pcut)&(chidf.FoldChange>fcut), :].sort_values(by=['factor','Chi2'],ascending=[True,False])
outf = args.output + ".bulk_chisq.tsv"
chidf.to_csv(outf,sep='\t',float_format="%.2e",index=False)
