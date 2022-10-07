import sys, os, gzip, copy, gc
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import *
from io import StringIO
import subprocess as sp

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--feature', type=str, help='')
parser.add_argument('--output_path', type=str, help='')

parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
parser.add_argument('--precision', type=int, default=1, help='Number of digits to store spatial location (in um), 0 for integer.')
parser.add_argument('--hex_width', type=int, default=24, help='')
parser.add_argument('--hex_radius', type=int, default=-1, help='')
parser.add_argument('--overlap', type=float, default=-1, help='')
parser.add_argument('--n_move', type=int, default=1, help='')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
args = parser.parse_args()

if not os.path.exists(args.input):
    print(f"ERROR: cannot find input file \n {args.input}")
    sys.exit()

mu_scale = 1./args.mu_scale
radius=args.hex_radius
b_size = 512

diam=args.hex_width
radius=args.hex_radius
if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = int(radius*np.sqrt(3))
if args.overlap >= 0 and args.overlap < 1:
    n_move = int(1 / (1. - args.overlap) )
else:
    n_move = args.n_move
    if n_move < 0:
        n_move = 1

### Output
if not os.path.exists(args.output_path):
    arg="mkdir -p " + args.output_path
    os.system(arg)

with gzip.open(args.input, 'rt') as rf:
    input_header=rf.readline().strip().split('\t')

feature=pd.read_csv(args.feature,sep='\t',header=0,usecols=['gene', 'gene_id', args.key])
feature.sort_values(by=args.key, ascending=False, inplace=True)
feature.drop_duplicates(subset='gene', inplace=True)
feature['dummy'] = "Gene Expression"
f = args.output_path + "/features.tsv.gz"
feature[['gene_id','gene','dummy']].to_csv(f, sep='\t', index=False, header=False)

feature_kept = list(feature.gene.values)
ft_dict = {x:i for i,x in enumerate( feature_kept ) }
M = len(feature_kept)

brc_f = args.output_path + "/barcodes.tsv"
mtx_f = args.output_path + "/matrix.mtx"
# If exists, delete
if os.path.exists(brc_f):
    _ = os.system("rm " + brc_f)
if os.path.exists(mtx_f):
    _ = os.system("rm " + mtx_f)

n_unit = 0
T = 0

adt = {x:int for x in ['tile','X','Y', args.key]}
adt.update({x:str for x in ['gene', 'gene_id']})
df = pd.DataFrame()
for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=500000, header=0, usecols=['tile','X','Y','gene','gene_id',args.key], dtype=adt):

    ed = chunk.Y.iloc[-1]
    left = copy.copy(chunk[chunk.Y > ed - 5 * args.mu_scale])
    df = pd.concat([df, chunk])
    if chunk.shape[0] == 0:
        break
    df['j'] = df.X.astype(str) + '_' + df.Y.astype(str)
    print(df.shape[0], ed)
    brc_full = df.groupby(by = ['j','tile','X','Y']).agg({args.key: sum}).reset_index()
    brc_full.index = range(brc_full.shape[0])
    pixel_ct = brc_full[args.key].values
    pts = np.asarray(brc_full[['X','Y']]) * mu_scale
    print(f"Read data with {brc_full.shape[0]} pixels and {feature.shape[0]} genes.")
    df.drop(columns = ['X', 'Y'], inplace=True)

    # Make DGE
    barcode_kept = list(brc_full.j.values)
    bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
    indx_row = [ bc_dict[x] for x in df['j']]
    indx_col = [ ft_dict[x] for x in df['gene']]
    N = len(barcode_kept)

    dge_mtx = coo_matrix((df[args.key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
    total_molecule=df[args.key].sum()
    print(f"Made DGE {dge_mtx.shape}")

    offs_x = 0
    offs_y = 0
    while offs_x < n_move:
        while offs_y < n_move:
            x,y = pixel_to_hex(pts, radius, offs_x/n_move, offs_y/n_move)
            hex_crd = list(zip(x,y))
            ct = pd.DataFrame({'hex_id':hex_crd, 'tot':pixel_ct}).groupby(by = 'hex_id').agg({'tot': sum}).reset_index()
            mid_ct = np.median(ct.loc[ct.tot >= args.min_ct_per_unit, 'tot'].values)
            ct = set(ct.loc[ct.tot >= args.min_ct_per_unit, 'hex_id'].values)
            if len(ct) < 2:
                offs_y += 1
                continue
            hex_list = list(ct)
            hex_dict = {x:i for i,x in enumerate(hex_list)}
            sub = pd.DataFrame({'crd':hex_crd,'cCol':range(N), 'X':pts[:, 0], 'Y':pts[:, 1], 'tile':brc_full.tile.values})
            sub = sub[sub.crd.isin(ct)]
            sub['cRow'] = sub.crd.map(hex_dict).astype(int)
            brc = sub[['cRow', 'tile', 'X', 'Y']].groupby(by = 'cRow').agg({'X':np.mean, 'Y':np.mean, 'tile':np.max}).reset_index()
            brc['X'] = [f"{x:.{args.precision}f}" for x in brc.X.values]
            brc['Y'] = [f"{x:.{args.precision}f}" for x in brc.Y.values]
            brc.sort_values(by = 'cRow', inplace=True)
            with open(brc_f, 'a') as wf:
                _ = wf.write('\n'.join((brc.cRow+n_unit+1).astype(str).values + '_' + brc.tile.astype(str) + '_' + brc.X.values + '_' + brc.Y.values)+'\n')
            n_hex = len(hex_dict)
            n_minib = n_hex // b_size
            print(f"{n_minib}, {n_hex} ({sub.cRow.max()}, {sub.shape[0]}), median count per unit {mid_ct}")
            grd_minib = list(range(0, n_hex, b_size))
            grd_minib[-1] = n_hex
            st_minib = 0
            n_minib = len(grd_minib) - 1
            while st_minib < n_minib:
                indx_minib = (sub.cRow >= grd_minib[st_minib]) & (sub.cRow < grd_minib[st_minib+1])
                npixel_minib = sum(indx_minib)
                offset = sub.loc[indx_minib, 'cRow'].min()
                nhex_minib = sub.loc[indx_minib, 'cRow'].max() - offset + 1

                mtx = coo_matrix((np.ones(npixel_minib, dtype=bool), (sub.loc[indx_minib, 'cRow'].values-offset, sub.loc[indx_minib, 'cCol'].values)), shape=(nhex_minib, N) ).tocsr() @ dge_mtx

                mtx.eliminate_zeros()
                r, c = mtx.nonzero()
                r = np.array(r,dtype=int) + offset + n_unit + 1
                c = np.array(c,dtype=int) + 1
                T += mtx.sum()
                mtx = pd.DataFrame({'i':c, 'j':r, 'v':mtx.data})
                mtx['i'] = mtx.i.astype(int)
                mtx['j'] = mtx.j.astype(int)
                mtx.to_csv(mtx_f, mode='a', sep=' ', index=False, header=False)
                st_minib += 1
                print(f"{st_minib}/{n_minib}. Wrote {nhex_minib} units.")
            n_unit += brc.shape[0]
            print(f"Sliding offset {offs_x}, {offs_y}. Wrote {n_unit} units so far.")
            offs_y += 1
        offs_y = 0
        offs_x += 1
    df = copy.copy(left)

_ = os.system("gzip -f " + brc_f)

mtx_header = args.output_path + "/matrix.header"
with open(mtx_header, 'w') as wf:
    line = "%%MatrixMarket matrix coordinate integer general\n%\n"
    line += " ".join([str(x) for x in [M, n_unit, T]]) + "\n"
    wf.write(line)

arg = " ".join(["cat",mtx_header,mtx_f,"|gzip -c > ", mtx_f+".gz"])
if os.system(arg) == 0:
    _ = os.system("rm " + mtx_f)
    _ = os.system("rm " + mtx_header)
