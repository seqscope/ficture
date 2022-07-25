import sys, io, os, copy, gc
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import *

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output_pref', type=str, help='')
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced, velo: velo total')
parser.add_argument('--hex_width', type=int, default=24, help='')
parser.add_argument('--hex_radius', type=int, default=-1, help='')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
parser.add_argument('--min_count_per_feature', type=int, default=50, help='')
parser.add_argument('--n_move', type=int, default=-1, help='')
args = parser.parse_args()

mu_scale = 1./args.mu_scale
radius=args.hex_radius
diam=args.hex_width
if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = int(radius*np.sqrt(3))
out_pref = args.output_pref + ".d_" + str(diam)

### Input and output
if not os.path.exists(args.input):
    print(f"ERROR: cannot find input file \n {args.input}")
    sys.exit()

### Read data
try:
    df = pd.read_csv(args.input, sep='\t', usecols = ['X','Y','gene',args.key])
except:
    df = pd.read_csv(args.input, sep='\t', compression='bz2', usecols = ['X','Y','gene',args.key])

feature = df[['gene', args.key]].groupby(by = 'gene', as_index=False).agg({args.key:sum}).rename(columns = {args.key:'gene_tot'})
feature = feature.loc[feature.gene_tot > args.min_count_per_feature, :]
gene_kept = list(feature['gene'])
df = df[df.gene.isin(gene_kept)]
df['j'] = df.X.astype(str) + '_' + df.Y.astype(str)

brc = df.groupby(by = ['j','X','Y']).agg({args.key: sum}).reset_index()
brc.index = range(brc.shape[0])
pixel_ct = brc[args.key].values
pts = np.asarray(brc[['X','Y']]) * mu_scale
print(f"Read data with {brc.shape[0]} pixels and {len(gene_kept)} genes.")
df.drop(columns = ['X', 'Y'], inplace=True)

# Make DGE
feature_kept = copy.copy(gene_kept)
barcode_kept = list(brc.j.values)
del brc
gc.collect()
bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
ft_dict = {x:i for i,x in enumerate( feature_kept ) }
indx_row = [ bc_dict[x] for x in df['j']]
indx_col = [ ft_dict[x] for x in df['gene']]
N = len(barcode_kept)
M = len(feature_kept)
T = df[args.key].sum()
dge_mtx = coo_matrix((df[args.key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
feature_mf = np.asarray(dge_mtx.sum(axis = 0)).reshape(-1)
feature_mf = feature_mf / feature_mf.sum()
total_molecule=df[args.key].sum()
print(f"Made DGE {dge_mtx.shape}")
del df
gc.collect()


flt = pd.DataFrame({'gene':feature_kept})
flt['i'] = flt.gene.map(ft_dict)
flt = flt.merge(right = feature[['gene', 'gene_tot']], on = 'gene', how = 'left')
f = out_pref + ".features.tsv"
flt.to_csv(f, sep='\t', index=False, header=False)

brc_f = out_pref + ".barcode.tsv"
mtx_f = out_pref + ".matrix.mtx"
with open(mtx_f, 'w') as wf:
    line = "%%MatrixMarket matrix coordinate integer general\n%\n"
    line += " ".join([str(x) for x in [M, 0, T]]) + "\n"
    wf.write(line)

n_move = args.n_move
if n_move > diam or n_move < 0:
    n_move = diam // 4

b_size = 512
last_barcode = 0
offs_x = 0
offs_y = 0
while offs_x < n_move:
    while offs_y < n_move:
        x,y = pixel_to_hex(pts, radius, offs_x/n_move, offs_y/n_move)
        hex_crd = list(zip(x,y))
        ct = pd.DataFrame({'hex_id':hex_crd, 'tot':pixel_ct}).groupby(by = 'hex_id').agg({'tot': sum}).reset_index()
        mid_ct = np.median(ct.loc[ct.tot >= args.min_ct_per_unit, 'tot'].values)
        ct = set(ct.loc[ct.tot >= args.min_ct_per_unit, 'hex_id'].values)
        hex_list = list(ct)
        hex_dict = {x:i for i,x in enumerate(hex_list)}
        sub = pd.DataFrame({'crd':hex_crd,'cCol':range(N), 'X':pts[:, 0], 'Y':pts[:, 1]})
        sub = sub[sub.crd.isin(ct)]
        sub['cRow'] = sub.crd.map(hex_dict).astype(int)

        brc = sub[['cRow','X', 'Y']].groupby(by = 'cRow').agg({'X':np.mean, 'Y':np.mean}).reset_index()
        brc['X'] = np.around(brc.X.values).astype(int)
        brc['Y'] = np.around(brc.Y.values).astype(int)
        brc.sort_values(by = 'cRow', inplace=True)
        brc['cRow'] = last_barcode + brc.cRow.values + 1
        brc.to_csv(brc_f, mode='a', sep='\t', index=False, header=False)

        n_hex = len(hex_dict)
        n_minib = n_hex // b_size
        print(f"{n_minib}, {n_hex}, median count per unit {mid_ct}")
        if n_hex < b_size // 4:
            offs_y += 1
            continue
        grd_minib = list(range(0, n_hex, b_size))
        grd_minib[-1] = n_hex - 1
        st_minib = 0
        n_minib = len(grd_minib) - 1
        
        while st_minib < n_minib:
            indx_minib = (sub.cRow >= grd_minib[st_minib]) & (sub.cRow < grd_minib[st_minib+1])
            npixel_minib = sum(indx_minib)
            nhex_minib = sub.loc[indx_minib, 'cRow'].max() - grd_minib[st_minib] + 1

            mtx = coo_matrix((np.ones(npixel_minib, dtype=bool), (sub.loc[indx_minib, 'cRow'].values-grd_minib[st_minib], sub.loc[indx_minib, 'cCol'].values)), shape=(nhex_minib, N) ).tocsr() @ dge_mtx

            mtx.eliminate_zeros()
            r, c = mtx.nonzero()
            r = sub.cRow.iloc[r].values + last_barcode + 1
            mtx = pd.DataFrame({'i':c, 'j':r, 'v':mtx.data})
            mtx['i'] = mtx.i.astype(int)
            mtx['j'] = mtx.j.astype(int)
            mtx.to_csv(mtx_f, mode='a', sep=' ', index=False, header=False)
            st_minib += 1
            print(f"Minibatch {st_minib}/{n_minib}. Write data matrix {mtx.shape}.")

        last_barcode = brc.cRow.iloc[-1]
        print(f"Sliding offset {offs_x}, {offs_y}. Fit data with {n_hex} units.")
        offs_y += 1
    offs_y = 0
    offs_x += 1
