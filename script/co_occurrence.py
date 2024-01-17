"""
Compute gene co-occurrence from pixel level input
Gaussian kernel
"""
import sys, io, os, gzip, glob, copy, re, time, logging, warnings, argparse
import numpy as np
import pandas as pd

from scipy.sparse import *
from sklearn.preprocessing import normalize
from sklearn.neighbors import radius_neighbors_graph

import geojson
import shapely.prepared, shapely.geometry
from shapely.geometry import Point

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--feature', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--boundary', type=str, default = '', help='')

parser.add_argument('--major_axis', type=str, default="Y", help='X or Y')
parser.add_argument('--mu_scale', type=float, default=1, help='Coordinate to um translate')
parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced. Otherwise depending on customized ct_header')
parser.add_argument('--radius', type=float, default=15, help='')
parser.add_argument('--resolution', type=int, default=2, help='')
parser.add_argument('--half_life', type=float, default=10, help='')
parser.add_argument('--min_ct', type=float, default=100, help='')
parser.add_argument('--window_size', type=int, default=3000, help='')

parser.add_argument('--thread', type=int, default=1, help='')

args = parser.parse_args()
logging.basicConfig(level= getattr(logging, "INFO", None))

mj = args.major_axis
key = args.key
wsize = args.window_size
resolution = args.resolution
adj_radius = args.radius
half_life = args.half_life
buff = half_life
mi = 'X' if mj == 'Y' else 'Y'
tau = - np.log(.5) / half_life**2

feature = pd.read_csv(args.feature, sep='\t')
feature.drop(index=feature.index[feature[key].lt(args.min_ct)], inplace=True)
feature.sort_values(by = key, ascending = False, inplace = True)
feature.drop_duplicates(subset = 'gene', inplace = True)
M = len(feature)
feature.index = np.arange(M)
ft_dict = {x:i for i,x in enumerate(feature.gene.values )}
# dangeraous
kept_gene_id = set(feature.gene_id.values)
logging.info(f"Load {M} feature from {args.feature}")

mj_max = np.inf
use_boundary = False
if os.path.isfile(args.boundary):
    mpoly = shapely.geometry.shape(geojson.load(open(args.boundary, 'rb')))
    mj_max = mpoly.bounds[2] if mj == 'X' else mpoly.bounds[3]
    mpoly = shapely.prepared.prep(mpoly)
    use_boundary = True
    logging.info(f"Load boundary from {args.boundary}")

reader = pd.read_csv(gzip.open(args.input, 'rt'), sep='\t', usecols=["X","Y","gene_id","gene",key], dtype={x:int for x in ["X","Y",key]}, chunksize = 500000)

Q = np.zeros((M,M), dtype=float)
df = pd.DataFrame()
st = -1
ed = -2
for chunk in reader:
    chunk.X /= args.mu_scale
    chunk.Y /= args.mu_scale
    chunk.drop(index = chunk.index[(~chunk.gene_id.isin(kept_gene_id))|chunk[key].eq(0)], inplace=True)
    chunk.drop(columns = ['gene_id'], inplace=True)
    if len(chunk) == 0:
        continue
    if st > ed:
        st = chunk[mj].iloc[0]
    ed = chunk[mj].iloc[-1]
    df = pd.concat([df, chunk])
    if ed - st > wsize:
        logging.info(f"{st:.1f}, {ed:.1f}, {df.shape[0]}")
        df.X = (df.X / resolution).astype(int) * resolution
        df.Y = (df.Y / resolution).astype(int) * resolution
        brc = df.groupby(by=["X","Y"]).agg({key:sum}).reset_index()
        if use_boundary:
            kept = [mpoly.contains(Point(*p)) for p in brc[['X','Y']].values]
            logging.info(f"Boundary filter: {brc.shape[0]} -> {np.sum(kept)}")
            brc = brc.loc[kept, :]
            if len(brc) < 2:
                df = pd.DataFrame()
                st = -1
                ed = -2
                continue
        brc.sort_values(by = mi, inplace=True)
        N = brc.shape[0]
        brc.index = np.arange(N)
        bc_dict = {x:i for i,x in enumerate(zip(brc.X.values, brc.Y.values)) }
        df['i'] = list(zip(df.X.values, df.Y.values))
        df.drop(index = df.index[~df.i.isin(bc_dict)], inplace=True)
        df['i'] = df.i.map(bc_dict)
        mtx = coo_array((df[key].values, (df.i.values, df.gene.map(ft_dict))), shape=(N,M) ).tocsr()
        logging.info(f"Processing {N} (collapsed) pixels")

        xmax = brc[mi].max()
        xst = brc[mi].min()
        while xst < xmax:
            xed = xst + wsize
            if xed > xmax - wsize:
                xed = xmax
            indx = brc.index[(brc[mi] >= xst - buff) & (brc[mi] <= xed)]
            n = len(indx)
            logging.info(f"{n}, ({xst}, {xed}) x ({st:.0f}, {ed:.0f})")
            xst = xed
            if n < 2:
                continue
            A = radius_neighbors_graph(brc.loc[indx, ['X','Y']].values, radius = adj_radius, mode='distance', include_self=False, n_jobs=args.thread)
            w = copy.copy(A.data)
            A.data = np.exp(- w**2 * tau)
            A += diags(np.ones(n),0,shape=(n,n)).tocsr()

            Q += mtx[indx, :].T @ A @ mtx[indx, :]

        df = df.loc[df[mj] > ed - buff]
        if len(df) == 0:
            st = -1
            ed = -2
        else:
            st = df[mj].iloc[0]
            ed = df[mj].iloc[-1]
    if len(df) > 0 and df[mj].iloc[-1] > mj_max:
        break

Q = normalize(Q, norm = 'l1', axis = 1)
with open (args.output, 'wb') as wf:
    np.save(wf, Q)
    np.save(wf, feature.gene.values)
