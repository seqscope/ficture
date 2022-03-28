import sys, io, os, copy, re, time, importlib, warnings, subprocess
from collections import defaultdict, Counter

import pickle, argparse
import numpy as np
import pandas as pd
from random import shuffle

from scipy.sparse import *
import scipy.stats
import sklearn.neighbors
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hexagon_fn
from hexagon_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='')
parser.add_argument('--identifier', type=str, help='')
parser.add_argument('--experiment_id', type=str, help='')
parser.add_argument('--filter_criteria_id', type=str, help='Used if filtered and merged data file is to be stored.', default = '')
parser.add_argument('--lane', type=str, help='')
parser.add_argument('--tile', type=str, help='')
parser.add_argument('--lane_model', default='', type=str, help='')
parser.add_argument('--tile_model', default='', type=str, help='')
parser.add_argument('--output_label', default='', type=str, help='')
parser.add_argument('--mu_scale', type=float, default=80, help='Coordinate to um translate')
parser.add_argument('--hex_width', type=int, default=24, help='')
parser.add_argument('--hex_width_fit', type=int, default=12, help='')
parser.add_argument('--hex_radius_fit', type=float, default=-1, help='')
parser.add_argument('--min_pixel_per_unit', type=int, default=20, help='')
parser.add_argument('--n_move_hex_tile', type=int, default=-1, help='')

args = parser.parse_args()

iden=args.identifier
path=args.path
expr_id=args.experiment_id
lane=args.lane
tile=args.tile
mu_scale = 1./args.mu_scale
pref=path+"/analysis/"+expr_id

lane_model = args.lane_model
if args.lane_model == '':
    lane_model = lane
tile_model = args.tile_model
if args.tile_model == '':
    tile_model = tile

suff = args.output_label
if lane_model != lane or tile != tile_model:
    if suff == '':
        suff = "test"
print(tile, tile_model, suff)

tile_list=tile.split(',')
tile_list_model=tile_model.split(',')

filter_id = ""
if args.filter_criteria_id != '':
    filter_id += "." + args.filter_criteria_id

min_pixel_per_unit_fit=args.min_pixel_per_unit

output_id = "d_"+str(args.hex_width) + ".lane_"+lane_model+'.'+'_'.join(tile_list_model)
m_files = glob.glob(pref+".nFactor_*."+output_id+".*model.p")
if len(m_files) == 0:
    sys.exit()

### Read data
flt_f = '/'.join([path,lane]) + "/matrix_merged_info.lane_"+lane+'.'+'_'.join(tile_list)+filter_id+".tsv.gz"
if not os.path.exists(flt_f):
    print(f"ERROR: cannot find input file, please run preprocessing script first.")
    sys.exit()
try:
    df = pd.read_csv(flt_f, sep='\t')
except:
    df = pd.read_csv(flt_f, sep='\t', compression='bz2')

model_f = m_files[0]
lda_base = pickle.load( open( model_f, "rb" ) )
gene_list = lda_base.feature_names_in_

df=df.loc[df.gene.isin(gene_list), :]
feature = df[['gene', 'gene_tot']].drop_duplicates(subset='gene')
brc = copy.copy(df[['j','X','Y','brc_tot']]).drop_duplicates(subset='j')
brc.index = range(brc.shape[0])
brc['x'] = brc.X.values * mu_scale
brc['y'] = brc.Y.values * mu_scale
pts = np.asarray(brc[['x','y']])
balltree = sklearn.neighbors.BallTree(pts)

# Make DGE
feature_kept = copy.copy(gene_list)
barcode_kept = list(brc['j'])
bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
ft_dict = {x:i for i,x in enumerate( feature_kept ) }
indx_row = [ bc_dict[x] for x in df['j']]
indx_col = [ ft_dict[x] for x in df['gene']]
N = len(barcode_kept)
M = len(feature_kept)

dge_mtx = coo_matrix((df['Count'], (indx_row, indx_col)), shape=(N, M)).tocsr()
print(f"Made DGE {dge_mtx.shape}")

# Apply fitted model
radius=args.hex_radius_fit
diam=args.hex_width_fit
if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = int(radius*np.sqrt(3))

n_move = args.n_move_hex_tile # sliding hexagon
if n_move > diam or n_move < 0:
    n_move = diam // 4

lda_base_result_full = []

for f in m_files:
    lda_base = pickle.load( open(f, "rb") )
    name = os.path.basename(f)
    wd = re.split('\.|_', name)
    K = int(wd[wd.index("nFactor")+1])
    print(K, name)
    offs_x = 0
    offs_y = 0
    while offs_x < n_move:
        while offs_y < n_move:
            x,y = pixel_to_hex(pts, radius, offs_x/n_move, offs_y/n_move)
            hex_crd = list(zip(x,y))
            hex_dict = {x:i for i,x in enumerate(list(set(hex_crd)))}
            c_row = [hex_dict[x] for x in hex_crd]
            c_col = range(pts.shape[0])
            Cmtx = coo_matrix( (np.ones(len(c_col), dtype=bool), (c_row, c_col)), shape=(len(hex_dict), N) ).tocsr()
            ct = np.asarray(Cmtx.sum(axis = 1)).squeeze()
            indx = ct >= min_pixel_per_unit_fit
            mtx = Cmtx[indx, :] @ dge_mtx
            n_unit = mtx.shape[0]

            perp = lda_base.perplexity(mtx)
            logl = lda_base.score(mtx)/n_unit
            print(f"Offsets: {offs_x},{offs_y}; log likelihood {logl:.2E}; perplexity {perp:.2E}")
            lda_base_result_full.append( [K, offs_x, offs_y, n_unit, perp, logl] )
            offs_y += 1
        offs_y = 0
        offs_x += 1

lda_base_result = pd.DataFrame(lda_base_result_full, columns = "nFactor,offs_x,offs_y,nUnit,Perplexity,LogLikelihood".split(','))
for x in ['Perplexity', 'LogLikelihood']:
    lda_base_result[x] = lda_base_result[x].map('{:,.3e}'.format)

for x in "nFactor,offs_x,offs_y,nUnit".split(','):
    lda_base_result[x] = lda_base_result[x].astype(int)

f = ".".join([x for x in [pref,output_id,"fit","d_"+str(diam),suff,"stats.tsv"] if x != ''])
lda_base_result.to_csv(f,sep='\t',index=False)
