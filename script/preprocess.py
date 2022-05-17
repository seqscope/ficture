import sys, io, os, copy, re, time, importlib, warnings, subprocess

packages = "numpy,pandas,sklearn,argparse,subprocess".split(',')
for pkg in packages:
    if not pkg in sys.modules:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pkg])

import argparse
import numpy as np
import pandas as pd
import subprocess as sp
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
parser.add_argument('--identifier', type=str, help='1stID-2ndZ-Specoes-L')
parser.add_argument("--meta_data", type=str, help="Per tile meta data menifest.tsv")
parser.add_argument("--layout", type=str, help="Layout file of tiles to draw [lane] [tile] [row] [col] format in each line")
parser.add_argument('--lane', type=str, help='')
parser.add_argument('--tile', type=str, help='',default='')
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--filter_based_on', type=str, default='gt', help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced, velo: velo total')

parser.add_argument('--gene_type_info', type=str, help='A file containing two columns, gene name and gene type. Used only if specific types of genes are kept.', default = '')
parser.add_argument('--gene_type_keyword', type=str, help='Key words (separated by ,) of gene types to keep, only used is gene_type_info is provided.', default="IG,TR,protein,lnc")
parser.add_argument('--rm_gene_keyword', type=str, help='Key words (separated by ,) of gene names to remove, only used is gene_type_info is provided.', default="")

parser.add_argument('--min_count_per_feature', type=int, default=20, help='')
parser.add_argument('--min_mol_density_squm', type=float, default=0.1, help='')
parser.add_argument('--hex_diam', type=int, default=12, help='')
parser.add_argument('--hex_n_move', type=int, default=6, help='')
parser.add_argument('--hard_rm_background_by_density',dest='hard_rm_dst', action='store_true')
parser.add_argument('--redo_filter', action='store_true')
parser.add_argument('--save_file_by_tile', action='store_true')

args = parser.parse_args()

path=args.input_path
outbase=args.output_path
lane=args.lane
kept_list=args.tile.split(',')
mu_scale = 1./args.mu_scale

### Output
outpath = '/'.join([outbase,lane])
if not os.path.exists(outpath):
    arg="mkdir -p "+outpath
    os.system(arg)
flt_f = outpath+"/matrix_merged."+args.identifier+".tsv.gz"
if os.path.exists(flt_f) and not args.redo_filter:
    sys.exit("Output file already exists. Do you want to --redo_filter?")

### Menifest
mani=pd.read_csv(args.meta_data, sep='\t')
mani["lane"] = mani["id"].map(lambda x : x.split('_')[0]).astype(int)
mani["tile"] = mani["id"].map(lambda x : x.split('_')[1]).astype(int)
mani = mani[mani.lane.eq(int(lane))]

xbin_min, xbin_max = mani.xmin.min(), mani.xmax.max()
ybin_min, ybin_max = mani.ymin.min(), mani.ymax.max()
xr = xbin_max-xbin_min+1
yr = ybin_max-ybin_min+1
print(f"Read meta data. Xmax, Ymax: {xbin_max}, {ybin_max}")

### Layout
layout = pd.read_csv(args.layout, sep='\t', dtype=int)
layout = layout[layout.lane.eq(int(lane))]
layout.sort_values(by = ['lane', 'row', 'col'], inplace=True)
tile_list = layout.tile.astype(str).values
df = layout.merge(right = mani[["lane", "tile", 'xmin', 'xmax', 'ymin', 'ymax']], on = ["lane", "tile"], how = "left")
if args.tile != '':
    df = df[df.tile.isin(kept_list)]
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
# Code the output as the tile numbers of the lower-left and upper-right corners
tile_ll = tiles[-1][0]
tile_ur = tiles[0][-1]
print(f"Read layout info. lane {lane}, tile {tile_ll}-{tile_ur}")

### Extract list of tiles to process
if args.tile == '':
    cmd="ls -lh "+path+"/"+lane+"/*/barcodes.tsv.gz | sed -r 's/\s+/,/g'"
    tab = sp.check_output(cmd, stderr=sp.STDOUT, shell=True).decode('utf-8')
    tab = tab.split('\n')
    tab = [x.split(',') for x in tab]
    tab = [[x[i] for i in [4, 8]] for x in tab if len(x) > 8]
    tab = pd.DataFrame(tab, columns = ['Size','File'])
    tab["Kb"] = tab.Size.map(lambda x : float(x[:-1]))
    indx = tab.Size.map(lambda x : x[-1].isdigit())
    tab.loc[indx, "Kb"] = tab.loc[indx, "Kb"].values * 1e-3
    indx = tab.Size.map(lambda x : x[-1]=='M')
    tab.loc[indx, "Kb"] = tab.loc[indx, "Kb"].values * 1e3
    indx = tab.Size.map(lambda x : x[-1]=='G')
    tab.loc[indx, "Kb"] = tab.loc[indx, "Kb"].values * 1e6
    if tab.Kb.iloc[-1] < 100:
        # If the barcode.tsv.gz of the largest tile is too small
        sys.exit("Did not locate informative tiles")
    tab["tile"] = tab.File.map(lambda x : os.path.dirname(x).split("/")[-1])
    tab.sort_values(by = "Kb", inplace=True)
    fc = tab.Kb.iloc[1:].values / tab.Kb.iloc[:-1].values
    cut_indx = fc.argmax()
    kept_list = sorted(tab.tile.iloc[(cut_indx+1):].values )

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

df = pd.DataFrame()
ntile = 0
for itr_r in range(len(lanes)):
    for itr_c in range(len(lanes[0])):
        lane, tile = lanes[itr_r][itr_c], tiles[itr_r][itr_c]
        if tile not in kept_list:
            continue
        mrg_f = outpath+"/"+tile+".matrix_merged.tsv.gz"
        if not os.path.exists(mrg_f):
            datapath = "/".join([path,lane,tile])
            f=datapath+"/barcodes.tsv.gz"
            brc = pd.read_csv(f, sep='\t|,', names=["barcode","j","v2","lane","tile","X","Y",\
                   "brc_tot_gn","brc_tot_gt",\
                   "brc_tot_spl","brc_tot_unspl","brc_tot_ambig"],\
                   usecols=["j","X","Y","brc_tot_gn", "brc_tot_gt",\
                            "brc_tot_spl","brc_tot_unspl","brc_tot_ambig"], engine='python')
            f=datapath+"/matrix.mtx.gz"
            mtx = pd.read_csv(f, sep=' ', skiprows=3, names=["i","j","gn","gt","spl","unspl","ambig"])
            f=datapath+"/features.tsv.gz"
            feature = pd.read_csv(f, sep='\t|,', names=["v1","gene","i","gene_tot_gn","gene_tot_gt",\
                   "gene_tot_spl","gene_tot_unspl","gene_tot_ambig"],\
                   usecols=["i","gene","gene_tot_gn","gene_tot_gt","gene_tot_spl",\
                            "gene_tot_unspl","gene_tot_ambig"],  engine='python')
            feature = feature[(feature.gene_tot_gt > 0) | (feature.gene_tot_gn > 0) | (feature.gene_tot_spl + feature.gene_tot_unspl > 0)]
            sub = mtx.merge(right = brc[['j','X','Y']], on = 'j', how = 'inner')
            sub = sub.merge(right = feature[["i","gene"]], on = 'i', how = 'inner' )
            if args.save_file_by_tile:
                sub.to_csv(mrg_f,sep='\t',index=False)
        else:
            sub = pd.read_csv(mrg_f,sep='\t')
        sub.drop(columns = ['i', 'j'], inplace=True)
        sub['X'] = (nrows - itr_r - 1) * xr + sub.X.values - xbin_min
        sub['Y'] = itr_c * yr + sub.Y.values - ybin_min
        sub['lane'] = lane
        sub['tile'] = tile
        if len(gene_kept) > 0:
            sub = sub.loc[sub.gene.isin(gene_kept), :]
        df = pd.concat([df, sub])
        ntile += 1
        print(f"Read data for {lane}_{tile}, {sub.shape}, {mrg_f}")

df['j'] = df.lane.astype(str) + '_' + df.tile.astype(str) + '_' + df.X.astype(str) + '_' + df.Y.astype(str)
print(f"Read data {df.shape}")

ct_header = ['gn', 'gt', 'spl', 'unspl', 'ambig']
feature = df[['gene']+ct_header].groupby(by = 'gene', as_index=False).agg({x:sum for x in ct_header}).rename(columns = {x:'gene_tot_' + x for x in ct_header})
feature['gene_tot_velo'] = feature.gene_tot_spl.values + feature.gene_tot_unspl.values + feature.gene_tot_ambig.values
feature = feature[(feature.gene_tot_gt > args.min_count_per_feature) | (feature.gene_tot_gn > args.min_count_per_feature) | (feature.gene_tot_velo > args.min_count_per_feature)]
gene_kept = list(feature['gene'])

df = df[df.gene.isin(gene_kept)]

if args.filter_based_on == 'gn':
    brc = df[['j','X','Y','gn']].groupby(by = ['j','X','Y'], as_index=False).agg({'gn':sum}).rename(columns = {'gn':'brc_tot'})
elif args.filter_based_on == 'velo':
    brc = df[['j','X','Y','spl','unspl','ambig']].groupby(by = ['j','X','Y'], as_index=False).agg({x:sum for x in ['spl','unspl','ambig']})
    brc['brc_tot'] = brc.spl.values + brc.unspl.values + brc.ambig.values
else:
    brc = df[['j','X','Y','gt']].groupby(by = ['j','X','Y'], as_index=False).agg({'gt':sum}).rename(columns = {'gt':'brc_tot'})

brc['x'] = brc.X.values * mu_scale
brc['y'] = brc.Y.values * mu_scale
brc = brc[['j','brc_tot','x','y']]

# ad hoc removal of background only based on density
diam = args.hex_diam
n_move = args.hex_n_move
radius = diam / np.sqrt(3)
hex_area = diam*radius*3/2
blur_center = pd.DataFrame()
for i in range(n_move):
    for j in range(n_move):
        brc['hex_x'], brc['hex_y'] = pixel_to_hex(np.asarray(brc[['x','y']]), radius, i/n_move, j/n_move)
        sub = brc.groupby(by = ['hex_x','hex_y']).agg({'brc_tot':sum}).reset_index()
        sub['hex_id'] = [(i,j,v['hex_x'],v['hex_y']) for k,v in sub.iterrows()]
        sub['x'], sub['y'] = hex_to_pixel(sub.hex_x.values, sub.hex_y.values, radius, i/n_move, j/n_move)
        sub = sub[sub.brc_tot > hex_area * 1e-2]
        blur_center = pd.concat([blur_center, sub])

if args.hard_rm_dst:
    blur_center['dense_center'] = blur_center.brc_tot >= args.min_mol_density_squm * hex_area
else:
    gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(np.log10(blur_center.brc_tot.values).reshape(-1,1))
    lab_keep = np.argmax(gm.means_.squeeze())
    lab = gm.predict(np.log10(blur_center.brc_tot.values).reshape(-1,1))
    lab = lab == lab_keep
    blur_center['dense_center'] = lab
    m0=(10**gm.means_.squeeze()[lab_keep])/hex_area
    m1=(10**gm.means_.squeeze()[1-lab_keep])/hex_area
    print(f"Filter background density, log scale, identified density {m0:.3f} v.s. {m1:.3f} molecule/um^2.")
    if m1 > m0 * 0.5 or m0 < args.min_mol_density_squm:
        gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(blur_center.brc_tot.values.reshape(-1,1))
        lab_keep = np.argmax(gm.means_.squeeze())
        lab = gm.predict(blur_center.brc_tot.values.reshape(-1,1))
        lab = lab == lab_keep
        blur_center['dense_center'] = lab
        m0=gm.means_.squeeze()[lab_keep]/hex_area
        m1=gm.means_.squeeze()[1-lab_keep]/hex_area
        print(f"Filter background density, original scale, identified density {m0:.3f} v.s. {m1:.3f} molecule/um^2.")
    if m1 > m0 * 0.5 or m0 < args.min_mol_density_squm:
        blur_center['dense_center'] = blur_center.brc_tot >= args.min_mol_density_squm * hex_area

dense_center = set(blur_center.loc[blur_center.dense_center.eq(True), 'hex_id'].values)
keep_pixel = np.zeros(brc.shape[0], dtype=bool)
for i in range(n_move):
    for j in range(n_move):
        x, y = pixel_to_hex(np.asarray(brc[['x','y']]), radius, i/n_move, j/n_move)
        indx = [True if (i,j,x[v],y[v]) in dense_center else False for v in range(len(x))]
        keep_pixel = keep_pixel | np.array(indx)

if keep_pixel.sum() == 0:
    print(f"WARNING: Not enough pixels in this field of view")
    sys.exit()

df = df[df.j.isin( brc.loc[keep_pixel, 'j'] )]
df.to_csv(flt_f, sep='\t', index=False)
print(f"Finish filtering, final size {df.shape[0]}")
