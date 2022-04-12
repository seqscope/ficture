import sys, io, os, copy, re, time, importlib, warnings, subprocess

packages = "numpy,pandas,sklearn,argparse".split(',')
for pkg in packages:
    if not pkg in sys.modules:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pkg])

import argparse
import numpy as np
import pandas as pd
import sklearn.neighbors
import sklearn.mixture

# Add parent directory
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hexagon_fn
from hexagon_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help='')
parser.add_argument('--output_path', type=str, help='')
parser.add_argument('--identifier', type=str, help='')
parser.add_argument("--layout", type=str, help="Layout file of tiles to draw [lane] [tile] [row] [col] format in each line")
parser.add_argument('--filter_criteria_id', type=str, help='Used if filtered and merged data file is to be stored.', default = '')
parser.add_argument('--lane', type=str, help='')
parser.add_argument('--tile', type=str, help='')
parser.add_argument('--mu_scale', type=float, default=80, help='Coordinate to um translate')
parser.add_argument('--gene_type_info', type=str, help='A file containing two columns, gene name and gene type. Used only if specific types of genes are kept.', default = '')
parser.add_argument('--gene_type_keyword', type=str, help='Key words (separated by ,) of gene types to keep, only used is gene_type_info is provided.', default="IG,TR,protein,lnc")

parser.add_argument('--min_count_per_feature', type=int, default=50, help='')
parser.add_argument('--min_mol_density_squm', type=int, default=1, help='')
parser.add_argument('--hex_diam', type=int, default=18, help='')
parser.add_argument('--hex_n_move', type=int, default=3, help='')
parser.add_argument('--auto_rm_background_by_density',dest='auto_rm_dst', action='store_true')
parser.add_argument('--redo_filter', action='store_true')

args = parser.parse_args()

iden=args.identifier
path=args.input_path
outbase=args.output_path
lane=args.lane
tile_list=args.tile.split(',')
mu_scale = 1./args.mu_scale

### Layout
df = pd.read_csv(args.layout,sep="\t")
tiles = [int(x) for x in tile_list]
df = df[df.lane.eq(int(lane)) & df.tile.isin(tiles)]
df.row = df.row - df.row.min()
df.col = df.col - df.col.min()
nrows = df.row.max() + 1
ncols = df.col.max() + 1
lanes = []
tiles = []
for i in range(nrows):
    lanes.append( [None] * ncols )
    tiles.append( [None] * ncols )
for index, row in df.iterrows():
    i = int(row['row'])
    j = int(row['col'])
    lanes[i][j] = str(row['lane'])
    tiles[i][j] = str(row['tile'])

### Output
outpath = '/'.join([outbase,lane])

filter_id = ""
if args.filter_criteria_id != '':
    filter_id += "." + args.filter_criteria_id

flt_f = outpath+"/matrix_merged_info.velo.lane_"+lane+'.'+'_'.join(tile_list)+filter_id+".tsv.gz"

if os.path.exists(flt_f) and not args.redo_filter:
    sys.exit()

if not os.path.exists(outpath):
    arg="mkdir -p "+outpath
    os.system(arg)

### If work on subset of genes
gene_kept = []
if args.gene_type_info != '' and os.path.exists(args.gene_type_info):
    gencode = pd.read_csv(args.gene_type_info, sep='\t', names=['Name','Type'])
    kept_key = args.gene_type_keyword.split(',')
    kept_type = gencode.loc[gencode.Type.str.contains('|'.join(kept_key)),'Type'].unique()
    gencode = gencode.loc[ gencode.Type.isin(kept_type) ]
    gene_kept = list(gencode.Name)

### Basic parameterse
min_count_per_feature=args.min_count_per_feature
min_mol_density_squm=args.min_mol_density_squm

xyr = [[sys.maxsize, 0], [sys.maxsize, 0]]
df = pd.DataFrame()
for itr_r in range(len(lanes)):
    for itr_c in range(len(lanes[0])):
        lane, tile = lanes[itr_r][itr_c], tiles[itr_r][itr_c]
        mrg_f = outpath+"/"+tile+"_matrix_merged_info.velo.tsv.gz"
        if not os.path.exists(mrg_f):
            datapath = "/".join([path,lane,tile])
            f=datapath+"/barcodes.tsv.gz"
            if not os.path.exists(f):
                print(f"WARNING: cannot find input file for {lane}_{tile}, missing tiles are assumed to be empty")
                continue
            brc = pd.read_csv(f, sep='\t|,', names=["barcode","j","v2","brc_tot","lane","tile","X","Y","brc_tot_spl","brc_tot_unspl","brc_tot_ambig"], usecols=["j","X","Y","brc_tot","brc_tot_spl","brc_tot_unspl","brc_tot_ambig"], engine='python')

            f=datapath+"/matrix.mtx.gz"
            mtx = pd.read_csv(f, sep=' ', skiprows=3,names=["i","j","spl","unspl","ambig"])

            f=datapath+"/features.tsv.gz"
            feature = pd.read_csv(f, sep='\t|,', names=["v1","gene","v3","i","gene_tot","gene_tot_spl","gene_tot_unspl","gene_tot_ambig"], usecols=["i","gene","gene_tot","gene_tot_spl","gene_tot_unspl","gene_tot_ambig"],  engine='python')
            feature = feature[feature.gene_tot > 0]

            sub = mtx.merge(right = brc[['j','X','Y','brc_tot',"brc_tot_spl","brc_tot_unspl"]], on = 'j', how = 'inner')
            sub = sub.merge(right = feature[["i","gene","gene_tot","gene_tot_spl","gene_tot_unspl"]], on = 'i', how = 'inner' )
            sub.to_csv(mrg_f,sep='\t',index=False)
        else:
            sub = pd.read_csv(mrg_f,sep='\t')

        sub['row'] = itr_r
        sub['col'] = itr_c
        sub.drop(columns = ['i','j'], inplace=True)
        x,y = sub.X.max(), sub.Y.max()
        if xyr[0][1] < x:
            xyr[0][1] = x
        if xyr[1][1] < y:
            xyr[1][1] = y
        x,y = sub.X.min(), sub.Y.min()
        if xyr[0][0] > x:
            xyr[0][0] = x
        if xyr[1][0] > y:
            xyr[1][0] = y
        df = pd.concat([df, sub])
        print(f"Read data for {lane}_{tile}, {sub.shape}, {mrg_f}")

if xyr[0][1] == 0:
    sys.exit()

print(f"Read data {df.shape}")

xbin_min, xbin_max = xyr[0]
ybin_min, ybin_max = xyr[1]
xr = xbin_max-xbin_min+1
yr = ybin_max-ybin_min+1

if len(gene_kept) > 0:
    df = df.loc[df.gene.isin(gene_kept), :]

df['X'] = (nrows-df.row-1)*xr + df.X.values - xbin_min
df['Y'] = df.col.values*yr + df.Y.values - ybin_min
df['j'] = df.X.astype(str) + '_' + df.Y.astype(str)

feature = df[['gene', 'gene_tot']].drop_duplicates(subset='gene')
feature = feature[(feature.gene_tot > min_count_per_feature)]
gene_kept = list(feature['gene'])
df = df[df.gene.isin(gene_kept)]

brc = copy.copy(df[['j','X','Y','brc_tot']]).drop_duplicates(subset='j')
brc['x'] = brc.X.values * mu_scale
brc['y'] = brc.Y.values * mu_scale
pts = np.asarray(brc[['x','y']])

# ad hoc removal of background only based on density
blur_tot = copy.copy(brc[['brc_tot', 'x', 'y']])
diam = args.hex_diam
n_move = args.hex_n_move
radius = diam / np.sqrt(3)
hex_area = diam*radius*3/2
blur_center = pd.DataFrame()
for i in range(n_move):
    for j in range(n_move):
        blur_tot['hex_x'], blur_tot['hex_y'] = pixel_to_hex(np.asarray(blur_tot[['x','y']]), radius, i/n_move, j/n_move)
        sub = blur_tot.groupby(by = ['hex_x','hex_y']).agg({'brc_tot':sum}).reset_index()
        sub['hex_id'] = [(i,j,v['hex_x'],v['hex_y']) for k,v in sub.iterrows()]
        blur_center = pd.concat([blur_center, sub])
if args.auto_rm_dst:
    gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(np.log10(blur_center.brc_tot.values).reshape(-1,1))
    lab_keep = np.argmax(gm.means_.squeeze())
    lab = gm.predict(np.log10(blur_center.brc_tot.values).reshape(-1,1))
    lab = lab == lab_keep
    blur_center['dense_center'] = lab
    m0=10**gm.means_.squeeze()[lab_keep]/hex_area
    m1=10**gm.means_.squeeze()[1-lab_keep]/hex_area
    print(f"Filter background density, log scale, identified density {m0:.3f} v.s. {m1:.3f}.")
    if m1 > m0 * 0.5:
        gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(blur_center.brc_tot.values.reshape(-1,1))
        lab_keep = np.argmax(gm.means_.squeeze())
        lab = gm.predict(blur_center.brc_tot.values.reshape(-1,1))
        lab = lab == lab_keep
        blur_center['dense_center'] = lab
        m0=gm.means_.squeeze()[lab_keep]/hex_area
        m1=gm.means_.squeeze()[1-lab_keep]/hex_area
        print(f"Filter background density, original scale, identified density {m0:.3f} v.s. {m1:.3f}.")

    if m0 < min_mol_density_squm or m1 > m0 * 0.5:
        blur_center['dense_center'] = blur_center.brc_tot >= min_mol_density_squm * hex_area
else:
    blur_center['dense_center'] = blur_center.brc_tot >= min_mol_density_squm * hex_area

dense_center = set(blur_center.loc[blur_center.dense_center.eq(True), 'hex_id'].values)
keep_pixel = np.zeros(blur_tot.shape[0], dtype=bool)
for i in range(n_move):
    for j in range(n_move):
        x, y = pixel_to_hex(np.asarray(blur_tot[['x','y']]), radius, i/n_move, j/n_move)
        indx = [True if (i,j,x[v],y[v]) in dense_center else False for v in range(len(x))]
        keep_pixel = keep_pixel | np.array(indx)

if keep_pixel.sum() == 0:
    print(f"WARNING: Not enough pixels in this field of view")
    sys.exit()

brc = brc.loc[keep_pixel, :]
df = df[df.j.isin(brc.j)]
df.to_csv(flt_f, sep='\t', index=False)
print("Finish filtering")
