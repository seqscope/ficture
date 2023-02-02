### Simple differential expression tests

import sys, io, os, gzip, copy, re, time, pickle, argparse
import numpy as np
import pandas as pd
import scipy.stats
from scipy.sparse import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='', help='')
parser.add_argument('--label', type=str, default='', help='')
parser.add_argument('--posterior_count', type=str, default='', help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--feature', type=str, default='', help='')
parser.add_argument('--unit_col_name', default='random_index', type=str, help='')
parser.add_argument('--key', default='gn', type=str, help='')
parser.add_argument('--min_ct_per_feature', default=50, type=int, help='')
parser.add_argument('--max_pval_output', default=1e-3, type=float, help='')
parser.add_argument('--min_fold_output', default=2, type=float, help='')
args = parser.parse_args()

pcut=args.max_pval_output
fcut=args.min_fold_output
key =args.key
gene_kept = set()
if os.path.exists(args.feature):
    feature = pd.read_csv(args.feature, sep='\t', header=0)
    gene_kept = set(feature.gene.values )

if os.path.exists(args.posterior_count):
    info = pd.read_csv(args.posterior_count,sep='\t',header=0)
    oheader = []
    header = []
    for x in info.columns:
        y = re.match('^[A-Za-z]+_(\d+)$', x)
        if y:
            header.append(y.group(1))
            oheader.append(x)
    K = len(header)
    M = info.shape[0]
    info.rename(columns = {oheader[k]:header[k] for k in range(K)}, inplace=True)
    print(f"Read posterior count over {M} genes and {K} factors")

else:
    if not os.path.exists(args.input) or not os.path.exists(args.label):
        sys.exit("Unable to find --input and --label")
    pmtx = pd.read_csv(args.label,sep='\t',header=0)
    pmtx.index =pmtx.j.astype(str)
    oheader = []
    header = []
    for x in pmtx.columns:
        y = re.match('^[A-Za-z]+_(\d+)$', x)
        if y:
            header.append(y.group(1))
            oheader.append(x)
    K = len(header)
    pmtx = pmtx.loc[:, oheader]
    pmtx.rename(columns = {oheader[k]:header[k] for k in range(K)}, inplace=True)
    print(f"Read mixed membership over {K} factors")

    adt = {args.unit_col_name:str, 'gene':str, "gene_tot":int}
    info = pd.DataFrame()
    for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=1000000, header=0, usecols=[args.unit_col_name,"gene",key], dtype=adt):
        chunk.rename(columns = {args.unit_col_name:'j'},inplace=True)
        chunk = chunk.loc[chunk.j.isin(pmtx.index), :]
        ct = pd.DataFrame(np.array(pmtx.loc[chunk.j.values, header]) * chunk[key].values.reshape((-1, 1)), columns = header)
        ct["gene"] = chunk.gene.values
        ct = ct.groupby(by = "gene", as_index=False).agg({x:np.sum for x in header})
        info = pd.concat([info, ct])
        info = info.groupby(by = "gene", as_index=False).agg({x:np.sum for x in header})
        print(info.shape)
    print(f"Aggregated counts")
    info = info.groupby(by = "gene", as_index=False).agg({x:np.sum for x in header})

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
chidf=chidf.loc[(chidf.pval<pcut)&(chidf.FoldChange>fcut), :].sort_values(by=['factor','FoldChange'],ascending=[True,False])
outf = args.output + ".bulk_chisq.tsv"
chidf.to_csv(outf,sep='\t',float_format="%.2e",index=False)
