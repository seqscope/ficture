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
import sklearn.preprocessing
import sklearn.neighbors
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hexagon_fn
from hexagon_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--model', type=str, help='')
parser.add_argument('--output_figure', type=str, help='')
parser.add_argument('--output_table', type=str, help='')
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')

parser.add_argument('--key', default = 'gt', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced, velo: velo total')
parser.add_argument('--gene_type_info', type=str, help='A file containing two columns, gene name and gene type. Used only if specific types of genes are kept.', default = '')
parser.add_argument('--gene_type_keyword', type=str, help='Key words (separated by ,) of gene types to keep, only used is gene_type_info is provided.', default="IG,TR,protein,lnc")
parser.add_argument('--rm_gene_keyword', type=str, help='Key words (separated by ,) of gene names to remove, only used is gene_type_info is provided.', default="")

parser.add_argument('--hex_width', type=int, default=12, help='')
parser.add_argument('--hex_radius', type=int, default=-1, help='')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
parser.add_argument('--min_count_per_feature', type=int, default=1, help='')
parser.add_argument('--n_move', type=int, default=-1, help='')
parser.add_argument('--thread', type=int, default=1, help='')

parser.add_argument('--figure_width', type=int, default=20, help="Width of the output figure per figure_scale_per_tile um")
parser.add_argument('--figure_scale_per_tile', type=int, default=3000, help="Final figure will have size scaling with figure_width x n_tiles")
parser.add_argument('--cmap_name', type=str, default="nipy_spectral", help="Name of Matplotlib colormap to use")

args = parser.parse_args()

try:
    lda_base = pickle.load( open( args.model, "rb" ) )
except:
    sys.exit("Please provide a proper model object")
L = lda_base.components_.shape[0]
feature_kept = lda_base.feature_names_in_
feature_kept_indx = list(range(len(feature_kept)))
mu_scale = 1./args.mu_scale

radius=args.hex_radius
diam=args.hex_width
if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = int(radius*np.sqrt(3))

### Input and output
if not os.path.exists(args.input):
    print(f"ERROR: cannot find input file \n {args.input}, please run preprocessing script first.")
    sys.exit()

### If work on subset of genes
gene_kept_org = set()
if args.gene_type_info != '' and os.path.exists(args.gene_type_info):
    gencode = pd.read_csv(args.gene_type_info, sep='\t', names=['Name','Type'])
    kept_key = args.gene_type_keyword.split(',')
    kept_type = gencode.loc[gencode.Type.str.contains('|'.join(kept_key)),'Type'].unique()
    gencode = gencode.loc[ gencode.Type.isin(kept_type) ]
    if args.rm_gene_keyword != "":
        rm_list = args.rm_gene_keyword.split(",")
        for x in rm_list:
            gencode = gencode.loc[ ~gencode.Name.str.contains(x) ]
    gene_kept_org = set(list(gencode.Name))
    feature_kept_indx = [i for i,x in enumerate(feature_kept) if x in gene_kept_org]

### Basic parameterse
b_size = 256 # minibatch size
topic_header = ['Topic_'+str(x) for x in range(L)]

### Read data
try:
    df = pd.read_csv(args.input, sep='\t', usecols = ['X','Y','gene',args.key])
except:
    df = pd.read_csv(args.input, sep='\t', compression='bz2', usecols = ['X','Y','gene',args.key])

df = df[df.gene.isin(feature_kept)]
if len(gene_kept_org) > 0:
    df = df[df.gene.isin(gene_kept_org)]
df.drop_duplicates(subset=['X','Y','gene'], inplace=True)
feature = df[['gene', args.key]].groupby(by = 'gene', as_index=False).agg({args.key:sum}).rename(columns = {args.key:'gene_tot'})
feature = feature.loc[feature.gene_tot > args.min_count_per_feature, :]
gene_kept = set(feature['gene'])
df = df[df.gene.isin(gene_kept)]
df['j'] = df.X.astype(str) + '_' + df.Y.astype(str)

feature_kept_indx = [i for i,x in enumerate(feature_kept) if x in gene_kept]
feature_kept = [feature_kept[i] for i in feature_kept_indx]
lda_base.components_ = lda_base.components_[:, feature_kept_indx]
lda_base.exp_dirichlet_component_ = sklearn.preprocessing.normalize(lda_base.exp_dirichlet_component_[:, feature_kept_indx], norm='l1', axis=1)
lda_base.feature_names_in_ = feature_kept
lda_base.n_features_in_ = len(feature_kept)
lda_base.doc_topic_prior_ = 1./L
lda_base.topic_word_prior_= 1./L
print(f"Keep {len(feature_kept)} informative genes")

brc = df.groupby(by = ['j','X','Y']).agg({args.key: sum}).reset_index()
brc.index = range(brc.shape[0])
pixel_ct = brc[args.key].values
pts = np.asarray(brc[['X','Y']]) * mu_scale
balltree = sklearn.neighbors.BallTree(pts)
print(f"Read data with {brc.shape[0]} pixels and {len(gene_kept)} genes.")
df.drop(columns = ['X', 'Y'], inplace=True)

# Make DGE
barcode_kept = list(brc.j.values)
del brc
gc.collect()
bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
ft_dict = {x:i for i,x in enumerate( feature_kept ) }
indx_row = [ bc_dict[x] for x in df['j']]
indx_col = [ ft_dict[x] for x in df['gene']]
N = len(barcode_kept)
M = len(feature_kept)
dge_mtx = coo_matrix((df[args.key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
feature_mf = np.asarray(dge_mtx.sum(axis = 0)).reshape(-1)
feature_mf = feature_mf / feature_mf.sum()
total_molecule=df[args.key].sum()
print(f"Made DGE {dge_mtx.shape}")
del df
gc.collect()


n_move = args.n_move
if n_move > diam or n_move < 0:
    n_move = diam // 4

res_f = args.output_table
wf = open(res_f, 'w')
out_header = "offs_x,offs_y,hex_x,hex_y".split(',')+['Topic_'+str(x) for x in range(L)]
out_header = '\t'.join(out_header)
_ = wf.write(out_header + '\n')

# Apply fitted model
b_size = 512
offs_x = 0
offs_y = 0
while offs_x < n_move:
    while offs_y < n_move:
        x,y = pixel_to_hex(pts, radius, offs_x/n_move, offs_y/n_move)
        hex_crd = list(zip(x,y))
        hex_list = list(set(hex_crd))
        hex_dict = {x:i for i,x in enumerate(hex_list)}
        sub = pd.DataFrame({'cRow':[hex_dict[x] for x in hex_crd], 'cCol':list(range(N))})
        n_hex = len(hex_dict)
        n_minib = n_hex // b_size
        grd_minib = list(range(0, n_hex, b_size))
        grd_minib[-1] = n_hex - 1
        st_minib = 0
        n_minib = len(grd_minib) - 1
        print(f"{n_minib}, {n_hex}")
        while st_minib < n_minib:
            indx_minib = (sub.cRow >= grd_minib[st_minib]) & (sub.cRow < grd_minib[st_minib+1])
            npixel_minib = sum(indx_minib)
            nhex_minib = grd_minib[st_minib+1] - grd_minib[st_minib]
            hex_crd_sub = [hex_list[x+grd_minib[st_minib]] for x in range(nhex_minib)]
            hex_crd[grd_minib[st_minib]:grd_minib[st_minib+1]]

            mtx = coo_matrix((np.ones(npixel_minib, dtype=bool), (sub.loc[indx_minib, 'cRow'].values-grd_minib[st_minib], sub.loc[indx_minib, 'cCol'].values)), shape=(nhex_minib, N) ).tocsr() @ dge_mtx
            ct = np.asarray(mtx.sum(axis = 1)).squeeze()
            indx = ct >= args.min_ct_per_unit
            logl = lda_base.score(mtx)
            theta = lda_base.transform(mtx)
            lines = [ [offs_x, offs_y, hex_crd_sub[i][0],hex_crd_sub[i][1]] + list(np.around(theta[i,], 5)) for i in range(theta.shape[0]) if indx[i] ]
            lines = ['\t'.join([str(x) for x in y])+'\n' for y in lines]
            _ = wf.writelines(lines)

            print(f"Minibatch {st_minib} with {sum(indx)} units, log likelihood {logl:.2E}")
            st_minib += 1
        print(f"{offs_x}, {offs_y}")
        offs_y += 1
    offs_y = 0
    offs_x += 1

wf.close()
del mtx
del dge_mtx
del lda_base
gc.collect()

dtp = {x:int for x in ['off_x','offs_y','hex_x','hex_y']}
dtp.update({"Topic_"+str(x):float for x in range(L)})
lda_base_result = pd.read_csv(res_f, sep='\t', dtype=dtp)

lda_base_result['Top_Topic'] = np.argmax(np.asarray(lda_base_result.loc[:, topic_header ]), axis = 1)
lda_base_result['Top_Prob'] = lda_base_result.loc[:, topic_header].max(axis = 1)
lda_base_result['Top_assigned'] = pd.Categorical(lda_base_result.Top_Topic)

# Transform back to pixel location
x,y = hex_to_pixel(lda_base_result.hex_x.values,\
                   lda_base_result.hex_y.values, radius, lda_base_result.offs_x.values/n_move, lda_base_result.offs_y.values/n_move)

lda_base_result["Hex_center_x"] = x
lda_base_result["Hex_center_y"] = y

lda_base_result.round(5).to_csv(res_f,sep='\t',index=False)

# Plot clustering result
cmap_name = args.cmap_name
if args.cmap_name not in plt.colormaps():
    cmap_name = "nipy_spectral"
cmap = plt.get_cmap(cmap_name, L)
clist = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(L)]

pt_size = 0.01
fig_width = int( (pts[:,1].max()-pts[:,1].min())/args.figure_scale_per_tile * args.figure_width )
plotnine.options.figure_size = (fig_width, fig_width)
with warnings.catch_warnings(record=True):
    ps = (
        ggplot(lda_base_result,
               aes(x='Hex_center_y', y='Hex_center_x',
                                    color='Top_assigned',alpha='Top_Prob'))
        +geom_point(size = pt_size, shape='o')
        +guides(colour = guide_legend(override_aes = {'size':3,'shape':'o'}))
        +xlab("")+ylab("")
        +guides(alpha=None)
        +coord_fixed(ratio = 1)
        +scale_color_manual(values = clist)
        +theme_bw()
        +theme(legend_position='bottom')
    )

ggsave(filename=args.output_figure,plot=ps,device='png',limitsize=False)
