import sys, io, os, copy, re, gc, time, importlib, warnings, subprocess
from collections import defaultdict, Counter
import pickle, argparse
import numpy as np
import pandas as pd
from random import shuffle

import matplotlib.pyplot as plt
from plotnine import *
import plotnine
import matplotlib

from scipy.sparse import *
import scipy.stats
import sklearn.neighbors
import sklearn.preprocessing

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from artm_fn import *
from hexagon_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help="")
parser.add_argument('--model', type=str, help="")
parser.add_argument('--output', type=str, help="")

parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced, velo: velo total')
parser.add_argument('--subset_tile', type=str, default = '', help='Lane:tile or just tile subset to work on. l1:t1,...,lk:tk or t1,...,tk')

parser.add_argument('--tau_phi_smooth', type=float, default=0, help="")
parser.add_argument('--tau_phi_sparse', type=float, default=0, help="")
parser.add_argument('--tau_theta_smooth', type=float, default=0, help="")
parser.add_argument('--tau_theta_sparse', type=float, default=0, help="")
parser.add_argument('--tau_decorr', type=float, default=0, help="")
parser.add_argument('--minibatch', type=int, default=256, help="")

parser.add_argument('--hex_width', type=int, default=24, help="")
parser.add_argument('--hex_radius', type=int, default=-1, help="")
parser.add_argument('--min_ct_per_unit', type=int, default=20, help="")
parser.add_argument('--min_count_per_feature', type=int, default=50, help="")
parser.add_argument('--n_move_train', type=int, default=-1, help="")

args = parser.parse_args()

### Input and output
if not os.path.exists(args.input) or not os.path.exists(args.model) :
    print(f"ERROR: cannot find input file \n {args.input}, please run preprocessing script first.")
    sys.exit()

### Basic parameterse
mu_scale = 1./args.mu_scale
b_size = args.minibatch # minibatch size
radius=args.hex_radius
diam=args.hex_width
if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = int(radius*np.sqrt(3))
diam_train = diam
n_move = args.n_move_train # sliding hexagon
if n_move > diam or n_move < 0:
    n_move = diam // 4

### Read data
try:
    df = pd.read_csv(args.input, sep='\t', usecols = ['X','Y','gene',args.key, 'tile'])
except:
    df = pd.read_csv(args.input, sep='\t', compression='bz2', usecols = ['X','Y','gene',args.key, 'tile'])
lda_base = pickle.load(open(args.model, 'rb'))
K = lda_base.components_.shape[0]

### If working on a subset of tiles
if args.subset_tile != '':
    kept_tile = [int(x) for x in args.subset_tile.split(',')]
    df.tile = df.tile.astype(int)
    df = df.loc[df.tile.isin(kept_tile), :]
    df.drop(columns = 'tile', inplace=True)

feature_kept = lda_base.feature_names_in_
ft_dict = {x:i for i,x in enumerate(feature_kept)}

df = df[df.gene.isin(feature_kept)]
feature = df.loc[:, ['gene', args.key]].groupby(\
                by = 'gene', as_index=False).agg({args.key:sum}).rename(columns = {args.key:'gene_tot'})
feature.sort_values(by = 'gene_tot', ascending=False, inplace=True)
feature['ct_rank'] = range(feature.shape[0])
feature.index = feature.gene.map(ft_dict).values

df['j'] = df.X.astype(str) + '_' + df.Y.astype(str)
brc = df.groupby(by = ['j','X','Y']).agg({args.key: sum}).reset_index()
brc['x'] = brc.X.values * mu_scale
brc['y'] = brc.Y.values * mu_scale
brc['indx'] = range(brc.shape[0])
brc.index = brc.indx.values
pixel_ct = brc[args.key].values
pts = np.asarray(brc[['x', 'y']])
balltree = sklearn.neighbors.BallTree(pts)
barcode_kept = list(brc.j.values)
bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
print(f"Read data with {brc.shape[0]} pixels and {len(feature_kept)} genes.")

# Make DGE
indx_row = [ bc_dict[x] for x in df['j']]
indx_col = [ ft_dict[x] for x in df['gene']]
N = len(barcode_kept)
M = len(feature_kept)
dge_mtx = coo_matrix((df[args.key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
total_molecule=df[args.key].sum()
feature_ct = np.asarray(dge_mtx.sum(axis = 0)).reshape(-1)
fw = feature_ct / feature_ct.sum()
print(f"Made DGE {dge_mtx.shape}")
del df
gc.collect()

# Parameters
tau = {'smooth_phi':np.abs(args.tau_phi_smooth),\
       'sparse_phi':-np.abs(args.tau_phi_sparse),\
       'smooth_theta':np.abs(args.tau_theta_smooth),\
       'sparse_theta':-np.abs(args.tau_theta_sparse),\
       'decorr_phi':-np.abs(args.tau_decorr)}
kernel_thres = 0.3
nS = K # Sparse
nB = 1 # Smooth
K = nS + nB
T = set(range(K))
S = set(range(nS))
B = set(range(nS, K))
max_iter = 20
min_iter = 5
rho_offset = 10
kappa = 0.7
tol_theta_max = 1e-4
tol_perp_rel = 1e-4
fw_prt = np.abs(np.random.normal(0, 1, size=(nB, M))) * (fw*0.15).reshape((1, -1))
fw_prt += fw.reshape((1, -1))
fw_prt = np.clip(fw_prt / fw_prt.sum(axis = 1).reshape((-1, 1)), 1e-6, 1)
phi0 = np.concatenate([lda_base.exp_dirichlet_component_, fw_prt], axis = 0) # P(w|t), KxM

### Initialize model
artm = ARTM(K,tau = tau, ker_thres = kernel_thres, B =  B, S = S)
artm.initialize_model(vocab=feature_kept, vocab_freq=fw, phi=phi0,\
                      tol_theta_max=tol_theta_max, tol_perp_rel=tol_perp_rel,\
                      kappa=kappa, verbose=1,\
                      rho_offset=rho_offset,max_iter=max_iter,min_iter=min_iter)

epoch = 0
offs_x = 0
offs_y = 0
while offs_x < n_move:
    while offs_y < n_move:
        N = pts.shape[0]
        x,y = pixel_to_hex(pts, radius, offs_x/n_move, offs_y/n_move)
        hex_crd = list(zip(x,y))
        ct = pd.DataFrame({'hex_id':hex_crd, 'tot':pixel_ct}).groupby(by = 'hex_id').agg({'tot': sum}).reset_index()
        mid_ct = np.median(ct.loc[ct.tot >= args.min_ct_per_unit, 'tot'].values)
        ct = set(ct.loc[ct.tot >= args.min_ct_per_unit, 'hex_id'].values)
        hex_list = list(ct)
        shuffle(hex_list)
        hex_dict = {x:i for i,x in enumerate(hex_list)}
        sub = pd.DataFrame({'crd':hex_crd,'cCol':range(N)})
        sub = sub[sub.crd.isin(ct)]
        sub['cRow'] = sub.crd.map(hex_dict)
        n_hex = len(hex_dict)
        n_minib = n_hex // b_size
        print(f"Epoch {epoch}. Median count per unit {mid_ct}.")
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
            print(f"... ... {st_minib}, {nhex_minib}")
            mtx = coo_matrix((np.ones(npixel_minib, dtype=bool), (sub.loc[indx_minib, 'cRow'].values-grd_minib[st_minib], sub.loc[indx_minib, 'cCol'].values)), shape=(nhex_minib, N) ).tocsr() @ dge_mtx
            st_minib += 1
            theta0 = lda_base.transform(mtx)
            n, M = mtx.shape
            rd_B = np.clip(np.abs( np.random.normal(0, 1,(n, nB))*np.sqrt(np.pi/2/K)), 0.05, 0.8)
            theta0 = np.concatenate([theta0, rd_B], axis = 1) # P(t|d), NxK
            theta0 = theta0 / theta0.sum(axis = 1).reshape((-1, 1))
            artm.fit_stochastic(mtx, theta0)
        print(f"Epoch {epoch}, sliding offset {offs_x}, {offs_y}. Fit data with {n_hex} units.")
        epoch += 1
        offs_y += 1
    offs_y = 0
    offs_x += 1

pickle.dump( artm, open( args.output, "wb" ) )
