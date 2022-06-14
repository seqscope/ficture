import sys, io, os, gc, copy, re, time, importlib, warnings
import subprocess as sp
import argparse
import numpy as np
import pandas as pd
import sklearn.neighbors
import sklearn.mixture
from random import choices
from collections import defaultdict,Counter

# Add parent directory
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hexagon_fn
from hexagon_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help='')
parser.add_argument('--output_path', type=str, help='')
parser.add_argument('--identifier', type=str, help='1stID-species-lane')
parser.add_argument("--meta_data", type=str, help="Per tile meta data menifest.tsv")
parser.add_argument("--layout", type=str, help="Layout file of tiles to draw [lane] [tile] [row] [col] format in each line")
parser.add_argument('--lane', type=str, help='')
parser.add_argument('--tile', type=str, default='')
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--filter_based_on', type=str, default='gt', help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced, velo: velo total')

parser.add_argument('--gene_type_info', type=str, help='A file containing two columns, gene name and gene type. Used only if specific types of genes are kept.', default = '')
parser.add_argument('--gene_type_keyword', type=str, help='Key words (separated by ,) of gene types to keep, only used is gene_type_info is provided.', default="IG,TR,protein,lnc")
parser.add_argument('--rm_gene_keyword', type=str, help='Key words (separated by ,) of gene names to remove, only used is gene_type_info is provided.', default="")

parser.add_argument('--min_count_per_feature', type=int, default=20, help='')
parser.add_argument('--min_mol_density_squm', type=float, default=0.05, help='')
parser.add_argument('--hex_diam', type=int, default=12, help='')
parser.add_argument('--hex_n_move', type=int, default=6, help='')
parser.add_argument('--hard_rm_background_by_density',dest='hard_rm_dst', action='store_true')
parser.add_argument('--redo_filter', action='store_true')
parser.add_argument('--save_file_by_tile', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

path=args.input_path
outbase=args.output_path
lane=args.lane
tile_list = args.tile.split(',')
tile_list = [x for x in tile_list if x.isnumeric()]
mu_scale = 1./args.mu_scale

### Output
outpath = '/'.join([outbase,lane])
if not os.path.exists(outpath):
    arg="mkdir -p "+outpath
    os.system(arg)
flt_f = outpath+"/matrix_merged."+args.identifier+".tsv"
print(f"Output file:\n{flt_f}")
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
if len(tile_list) > 0:
    tile_list_int = [int(x) for x in tile_list]
    layout = layout[layout.tile.isin(tile_list_int)]
layout.sort_values(by = ['lane', 'row', 'col'], inplace=True)
tile_list = layout.tile.astype(str).values
df = layout.merge(right = mani[["lane", "tile", 'xmin', 'xmax', 'ymin', 'ymax']], on = ["lane", "tile"], how = "left")
df.row = df.row - df.row.min()
df.col = df.col - df.col.min()
nrows = df.row.max() + 1
ncols = df.col.max() + 1
tiles = []
tile_to_position = {}
for i in range(nrows):
    tiles.append( [None] * ncols )
for index, row in df.iterrows():
    i = int(row['row'])
    j = int(row['col'])
    tiles[i][j] = str(row['tile'])
    tile_to_position[row['tile']] = [i, j]
# Code the output as the tile numbers of the lower-left and upper-right corners
tile_ll = tiles[-1][0]
tile_ur = tiles[0][-1]
print(f"Read layout info. lane {lane}, tile {tile_ll}-{tile_ur}")

### If work on subset of genes
gene_kept = set()
if args.gene_type_info != '' and os.path.exists(args.gene_type_info):
    gencode = pd.read_csv(args.gene_type_info, sep='\t', names=['Name','Type'])
    kept_key = args.gene_type_keyword.split(',')
    kept_type = gencode.loc[gencode.Type.str.contains('|'.join(kept_key)),'Type'].unique()
    gencode = gencode.loc[ gencode.Type.isin(kept_type) ]
    if args.rm_gene_keyword != "":
        rm_list = args.rm_gene_keyword.split(",")
        for x in rm_list:
            gencode = gencode.loc[ ~gencode.Name.str.contains(x) ]
    gene_kept = set(gencode.Name.values)

# Read file
diam = args.hex_diam
n_move = args.hex_n_move
radius = diam / np.sqrt(3)
hex_area = diam*radius*3/2
datapath = "/".join([path,lane])
ct_header = ['gn', 'gt', 'spl', 'unspl', 'ambig']

if len(tile_list) != 0: # Read part of the data
    try:
        print(f"Tile list: {','.join(tile_list)}")
        read_sub = 1
        for t in tile_list:
            f = datapath+"/"+t+"/barcodes.tsv.gz"
            if not os.path.exists(f):
                read_sub = 0
        if read_sub:
            df = pd.DataFrame()
            for t in tile_list:
                f=datapath+"/"+t+"/barcodes.tsv.gz"
                brc = pd.read_csv(f, sep='\t', names=["barcode","j","v2","lane","tile","X","Y","brc_tot"],\
                       usecols=["j","tile","X","Y"])
                print(f"Barcode for tile {t}: {brc.shape[0]}")

                f=datapath+"/"+t+"/matrix.mtx.gz"
                mtx = pd.read_csv(f, sep=' ', skiprows=3, names=["i","j","gn","gt","spl","unspl","ambig"])
                print(f"Matrix for tile {t}: {mtx.shape[0]}")

                f=datapath+"/"+t+"/features.tsv.gz"
                feature = pd.read_csv(f, sep='\t', names=["v1","gene","i","gene_tot"],\
                       usecols=["i","gene"])

                sub = mtx.merge(right = brc, on = 'j', how = 'inner')
                sub = sub.merge(right = feature, on = 'i', how = 'inner' )
                sub.drop(columns = ['i'], inplace=True)
                df = pd.concat([df, sub])
                print(f"Data for tile {t}: {sub.shape[0]}")
        else:
            f=datapath+"/barcodes.tsv.gz"
            iter_csv = pd.read_csv(f, sep='\t', names=["barcode","j","v2","lane","tile","X","Y","brc_tot"],\
                   usecols=["j","tile","X","Y"], iterator=True, chunksize=100000)
            brc = pd.concat([chunk[chunk.tile.astype(str).isin(tile_list)] for chunk in iter_csv])
            print(f"Barcode {brc.shape[0]}")

            f=datapath+"/matrix.mtx.gz"
            iter_csv = pd.read_csv(f, sep=' ', skiprows=3, names=["i","j","gn","gt","spl","unspl","ambig"], iterator=True, chunksize=100000)
            mtx = pd.concat([chunk[chunk.j.isin(brc.j.values)] for chunk in iter_csv])
            print(f"Matrix {mtx.shape[0]}")

            f=datapath+"/features.tsv.gz"
            feature = pd.read_csv(f, sep='\t', names=["v1","gene","i","gene_tot"],\
                   usecols=["i","gene"])
            df = mtx.merge(right = brc, on = 'j', how = 'inner')
            del brc
            gc.collect()
            df = df.merge(right = feature, on = 'i', how = 'inner' )
            del feature
            gc.collect()
            df.drop(columns = ['i'], inplace=True)
    except:
        sys.exit("ERROR: cannot read file")
else:
    try:
        f=datapath+"/barcodes.tsv.gz"
        brc = pd.read_csv(f, sep='\t', names=["barcode","j","v2","lane","tile","X","Y","brc_tot"],\
               usecols=["j","tile","X","Y"])
        f=datapath+"/matrix.mtx.gz"
        mtx = pd.read_csv(f, sep=' ', skiprows=3, names=["i","j","gn","gt","spl","unspl","ambig"])
        f=datapath+"/features.tsv.gz"
        feature = pd.read_csv(f, sep='\t', names=["v1","gene","i","gene_tot"], usecols=["i","gene"])
        df = mtx.merge(right = brc, on = 'j', how = 'inner')
        del brc
        gc.collect()
        df = df.merge(right = feature, on = 'i', how = 'inner' )
        del feature
        gc.collect()
        df.drop(columns = ['i'], inplace=True)
    except:
        sys.exit("ERROR: cannot read file")

print(f"Read raw data {df.shape}")


if len(gene_kept) > 0:
    df = df.loc[df.gene.isin(gene_kept), :]

feature = df[['gene']+ct_header].groupby(by = 'gene', as_index=False).agg({x:sum for x in ct_header})
if args.filter_based_on == 'velo':
    feature['gene_tot'] = feature.spl.values + feature.unspl.values
else:
    feature.rename(columns = {args.filter_based_on:'gene_tot'}, inplace=True)
gene_kept = set(feature.loc[ feature.gene_tot > args.min_count_per_feature, 'gene' ].values)
df = df.loc[df.gene.isin(gene_kept), :]
print(f"Keep {len(gene_kept)} genes")

v = df.tile.astype(int).map(lambda x : tile_to_position[x][0])
df['X'] = (nrows - v - 1) * xr + df.X.values - xbin_min
v = df.tile.astype(int).map(lambda x : tile_to_position[x][1])
df['Y'] = v * yr + df.Y.values - ybin_min
print(f"Merge data {df.shape}")

if args.filter_based_on == 'gn':
    brc = df[['j','X','Y','gn']].groupby(by = ['j','X','Y'], as_index=False).agg({'gn':sum}).rename(columns = {'gn':'brc_tot'})
elif args.filter_based_on == 'velo':
    brc = df[['j','X','Y','spl','unspl']].groupby(by = ['j','X','Y'], as_index=False).agg({x:sum for x in ['spl','unspl']})
    brc['brc_tot'] = brc.spl.values + brc.unspl.values
else:
    brc = df[['j','X','Y','gt']].groupby(by = ['j','X','Y'], as_index=False).agg({'gt':sum}).rename(columns = {'gt':'brc_tot'})
brc['x'] = brc.X.values * mu_scale
brc['y'] = brc.Y.values * mu_scale
brc = brc[['j','brc_tot','x','y']]

blur_center = pd.DataFrame()
for i in range(n_move):
    for j in range(n_move):
        brc['hex_x'], brc['hex_y'] = pixel_to_hex(np.asarray(brc[['x','y']]), radius, i/n_move, j/n_move)
        cnt = brc.groupby(by = ['hex_x','hex_y']).agg({'brc_tot':sum}).reset_index()
        cnt.loc[cnt.brc_tot > hex_area * args.min_mol_density_squm, :]
        cnt['hex_id'] = [(i,j,v['hex_x'],v['hex_y']) for k,v in cnt.iterrows()]
        cnt = cnt.loc[cnt.brc_tot > hex_area * args.min_mol_density_squm, ['hex_id',"brc_tot"]]
        blur_center = pd.concat([blur_center, cnt])

if blur_center.shape[0] == 0:
    sys.exist("Did not find enough pixels")
print(f"Record {blur_center.shape[0]} centers")

if not args.hard_rm_dst:
    max_sample = 100000
    x = np.log10(blur_center.brc_tot.values).reshape(-1,1)
    if x.shape[0] > max_sample:
        x = x[choices(range(x.shape[0]), k=max_sample), :]
    gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(x)
    lab_keep = np.argmax(gm.means_.squeeze())
    lab = gm.predict(np.log10(blur_center.brc_tot.values).reshape(-1,1))
    blur_center['dense_center'] = lab == lab_keep
    m0=(10**gm.means_.squeeze()[lab_keep])/hex_area
    m1=(10**gm.means_.squeeze()[1-lab_keep])/hex_area
    print(f"Filter background density, log scale, identified density {m0:.3f} v.s. {m1:.3f} molecule/um^2.")
    if m1 > m0 * 0.5 or m0 < args.min_mol_density_squm:
        x = blur_center.brc_tot.values.reshape(-1,1)
        if x.shape[0] > max_sample:
            x = x[choices(range(x.shape[0]), k=max_sample), :]
        gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(x)
        lab_keep = np.argmax(gm.means_.squeeze())
        lab = gm.predict(blur_center.brc_tot.values.reshape(-1,1))
        blur_center['dense_center'] = lab == lab_keep
        m0=gm.means_.squeeze()[lab_keep]/hex_area
        m1=gm.means_.squeeze()[1-lab_keep]/hex_area
        print(f"Filter background density, original scale, identified density {m0:.3f} v.s. {m1:.3f} molecule/um^2.")
    if m1 > m0 * 0.5 or m0 < args.min_mol_density_squm:
        blur_center['dense_center'] = blur_center.brc_tot >= args.min_mol_density_squm * hex_area
    density_cut = blur_center.loc[ blur_center.dense_center.eq(True), 'brc_tot'].min()
else:
    density_cut = blur_center.brc_tot.min()


output_header = ['X','Y','gene','gn','gt','spl','unspl','ambig','tile']

brc["kept"] = False
for i in range(n_move):
    for j in range(n_move):
        brc['hex_x'], brc['hex_y'] = pixel_to_hex(np.asarray(brc[['x','y']]), radius, i/n_move, j/n_move)
        brc['hex_id'] = [(i,j,v['hex_x'],v['hex_y']) for k,v in brc.iterrows()]
        cnt = brc.groupby(by = 'hex_id').agg({'brc_tot':sum}).reset_index()
        cnt = set(cnt.loc[cnt.brc_tot > density_cut, "hex_id"].values)
        brc.loc[brc.hex_id.isin(cnt), 'kept'] = True
df = df.loc[df.j.isin( brc.loc[brc.kept.eq(True), 'j'] ), output_header]
df.to_csv(flt_f, sep='\t', index=False)
print(f"Write data {df.shape} to \n{flt_f}")
