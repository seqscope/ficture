# Detect highly variable genes treating collapsed data as SC data

import sys, io, os, gzip, glob, copy, re, time, warnings, argparse
from collections import defaultdict,Counter
import subprocess as sp

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--unit_id', default='', type=str, help='')
parser.add_argument('--white_list', default='', type=str, help='')

parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')

parser.add_argument('--n_top_genes', type=int, default=3000, help='')
parser.add_argument('--max_n', type=int, default=100000, help='')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
parser.add_argument('--min_ct_per_feature', type=int, default=50, help='')
args = parser.parse_args()


mu_scale = 1./args.mu_scale
key=args.key
max_n = args.max_n

with gzip.open(args.input,'rt') as rf:
    header=rf.readline()

if os.path.exists(args.unit_id):
    with open(args.unit_id, 'r') as rf:
        unit_id = [x.strip() for x in rf.readlines()]
    print(f"Read {len(unit_id)} hexagon ID from {args.unit_id}")
else:
    indx=str(header.strip().split('\t').index("random_index") + 1)
    cmd="zcat "+args.input+" | grep -vP '^#' | cut -f "+indx+" | uniq "
    print(cmd)
    unit_id=sp.check_output(cmd, shell = True).decode("utf-8").split('\n')
    print(f"Read {len(unit_id)} hexagon ID from {args.input}")
unit_id=np.array(unit_id).astype(str)
np.random.shuffle(unit_id)
if len(unit_id) > max_n:
    unit_id = unit_id[:max_n]
unit_id = set(unit_id)

df = pd.DataFrame()
feature_ct = defaultdict(int)
dty = {'random_index':str, 'gene':str, key:int}
for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=1000000, header=0, usecols=["random_index","gene",key], dtype=dty):
    df = pd.concat((df, chunk[chunk.random_index.isin(unit_id)]))
    feature = chunk.groupby(by = 'gene').agg({key:sum}).reset_index()
    for i,v in feature.iterrows():
        feature_ct[v['gene']] += v[key]
    print(f"Streaming ... {df.shape[0]}")

df.rename(columns = {'random_index':'j'},inplace=True)
print(f"Randomly keep {len(unit_id)}. Read data {df.shape}")

feature = pd.DataFrame([[k,v] for k,v in feature_ct.items()],columns=['gene',key])
feature = feature[feature[key] > args.min_ct_per_feature]
if os.path.exists(args.white_list):
    feature_wl = []
    with open(args.white_list, 'r') as rf:
        for line in rf:
            feature_wl.append(line.strip().split('\t')[0])
    feature=feature[feature.gene.isin(feature_wl)]

feature.index=range(feature.shape[0])
df = df[df.gene.isin(feature.gene.values)]

n_top = min([args.n_top_genes, feature.shape[0]])
print(f"Will keep {n_top} top variable genes out of {feature.shape[0]}")

brc = df.groupby(by = ['j']).agg({key:sum}).reset_index()
brc = brc[brc[key] > args.min_ct_per_unit]
df = df[df.j.isin(brc.j.values)]

flt_dict = {x:i for i,x in enumerate(feature.gene.values)}
drc_dict = {x:i for i,x in enumerate(brc.j.values)}
N,M = len(drc_dict), len(flt_dict)
mtx = coo_matrix((df[key].values, (df.j.map(drc_dict), df.gene.map(flt_dict))), shape=(N,M), dtype=np.int32).tocsr()

adata = ad.AnnData(mtx,dtype=np.int32)
res1 = sc.pp.highly_variable_genes(adata, flavor='seurat_v3', inplace=False,  n_top_genes=n_top)
res1.index = range(res1.shape[0])

res2 = sc.experimental.pp.highly_variable_genes(adata, flavor='pearson_residuals', n_top_genes=n_top, inplace=False)
res2.index = range(res2.shape[0])
hv2 = res2.merge(right = feature, left_index=True,right_index=True, how='inner')
hv1 = res1.merge(right = feature, left_index=True,right_index=True, how='inner')
res = hv2.rename(columns = {'highly_variable_rank':'rank_pr','highly_variable':'hv_pr'}).merge(right =\
                hv1[['gene','highly_variable_rank','highly_variable']].rename(columns ={'highly_variable_rank':'rank_seurat3', 'highly_variable':'hv_seurat3'}),on = 'gene')

hv = copy.copy(res)
gene_list = list(hv.loc[hv.hv_pr.values & hv.hv_seurat3.values, 'gene'].values)
hv = hv[hv.hv_pr.values ^ hv.hv_seurat3.values]
hv['Rank'] = hv.rank_pr.values
hv.loc[pd.isna(hv.Rank),'Rank'] = hv.loc[pd.isna(hv.Rank),'rank_seurat3'].values
hv['Rank'] = hv.Rank.astype(int)
hv.sort_values(by = 'Rank', ascending=True, inplace=True)
gene_list += list(hv.gene.iloc[:(n_top-len(gene_list))].values )
gene_list = set(gene_list)
res['Label'] = 0
res.loc[res.gene.isin(gene_list), 'Label'] = 1

res.to_csv(args.output, sep='\t', index=False, header=True, float_format="%.5f")
