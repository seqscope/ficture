import sys, io, os, copy, re, time, importlib, warnings
import argparse, pickle
import numpy as np
import pandas as pd

# Add parent directory
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hexagon_fn
from hexagon_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--count_key', type=str, help='', default="Count")

parser.add_argument('--mu_scale', type=float, default=80, help='Coordinate to um translate')
parser.add_argument('--gene_type_info', type=str, help='A file containing two columns, gene name and gene type. Used only if specific types of genes are kept.', default = '')
parser.add_argument('--gene_type_keyword', type=str, help='Key words (separated by ,) of gene types to keep, only used is gene_type_info is provided.', default="IG,TR,protein,lnc")
parser.add_argument('--rm_gene_keyword', type=str, help='Key words (separated by ,) of gene names to remove, only used is gene_type_info is provided.', default="")

parser.add_argument('--min_count_per_feature', type=int, default=50, help='')
parser.add_argument('--min_count_per_doc', type=int, default=10, help='')
parser.add_argument('--hex_diam', type=int, default=8, help='')
parser.add_argument('--hex_n_move', type=int, default=4, help='')
args = parser.parse_args()

mu_scale = 1./args.mu_scale
min_count_per_feature=args.min_count_per_feature

### If work on subset of genes
gene_kept = []
if args.gene_type_info != '' and os.path.exists(args.gene_type_info):
    gencode = pd.read_csv(args.gene_type_info, sep='\t', names=['Name','Type'])
    kept_key = args.gene_type_keyword.split(',')
    kept_type = gencode.loc[gencode.Type.str.contains('|'.join(kept_key)),'Type'].unique()
    gencode = gencode.loc[ gencode.Type.isin(kept_type) ]
    if args.rm_gene_keyword != "":
        rm_list = args.rm_gene_keyword.split(",")
        for x in rm_list:
            gencode = gencode.loc[ ~gencode.Name.str.contains(x) ]
    gene_kept = list(gencode.Name)

df = pd.read_csv(args.input, sep='\t')
if len(gene_kept) > 0:
    df = df.loc[df.gene.isin(gene_kept), :]
df.rename(columns = {args.count_key: 'Count'}, inplace=True)

feature = df[['gene', 'Count']].groupby(by = 'gene', as_index=False).agg({'Count':sum}).rename(columns = {'Count':'gene_tot'})
feature = feature[(feature.gene_tot > min_count_per_feature)]
gene_kept = list(feature['gene'])
M = len(gene_kept)
ft_dict = {x:i for i,x in enumerate(gene_kept)}
df = df[df.gene.isin(gene_kept)]
print(f"Read data with {M} genes.")

df = df[["Count","X","Y","gene"]]
df['x'] = df.X.values * mu_scale
df['y'] = df.Y.values * mu_scale
df["hex_x"] = 0
df["hex_y"] = 0

diam = args.hex_diam
n_move = args.hex_n_move
if n_move > diam:
    n_move = diam // 2
radius = diam / np.sqrt(3)

HHt = np.zeros((M ,M))
Dv = np.zeros(M)
nDoc = 0

for i in range(n_move):
    for j in range(n_move):
        df['hex_x'], df['hex_y'] = pixel_to_hex(np.asarray(df[['x','y']]), radius, i/n_move, j/n_move)
        sub = df.groupby(by = ['hex_x','hex_y','gene']).agg({"Count":sum}).reset_index()
        sub['hex_id'] = sub.hex_x.astype(int).astype(str) + '_' + sub.hex_y.astype(int).astype(str)
        tot = sub.groupby(by = 'hex_id').agg({"Count":sum}).reset_index()
        tot.rename(columns = {"Count":"ct_tot"}, inplace=True)
        tot = tot[tot.ct_tot >= args.min_count_per_doc]
        print(i, j, tot.shape[0], sub.shape[0])
        sub = sub.merge(right = tot, on = 'hex_id', how = 'inner')
        barcode_kept = tot.hex_id.values
        bc_dict = {x:y for y,x in enumerate(barcode_kept)}
        indx_row = [ bc_dict[x] for x in sub.hex_id.values]
        indx_col = [ ft_dict[x] for x in sub.gene.values]
        N = len(barcode_kept)
        nDoc += N
        mtx = coo_matrix((sub.Count.values / np.sqrt(sub.ct_tot.values * (sub.ct_tot.values - 1)), (indx_row, indx_col)), shape = (N, M)).tocsr()
        HHt += mtx.T @ mtx
        Dv += np.asarray(coo_matrix((sub.Count.values / (sub.ct_tot.values * (sub.ct_tot.values - 1)), (indx_row, indx_col)), shape = (N, M)).tocsc().sum(axis = 0)).squeeze()

res = {"HHt":HHt, "Diag":Dv, "nDoc":nDoc, "gene_list":gene_kept}
pickle.dump( res, open( args.output, "wb" ) )
