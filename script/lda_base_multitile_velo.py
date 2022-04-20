import sys, io, os, copy, re, time, importlib, warnings, subprocess
from collections import defaultdict, Counter

packages = "numpy,scipy,sklearn,argparse,plotnine,pandas".split(',')
for pkg in packages:
    if not pkg in sys.modules:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pkg])

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
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hexagon_fn
from hexagon_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, help='')
parser.add_argument('--identifier', type=str, help='')
parser.add_argument('--splice', default = 'spl', type=str, help='')
parser.add_argument('--experiment_id', type=str, help='')
parser.add_argument('--filter_criteria_id', type=str, help='Used if filtered and merged data file is to be stored.', default = '')
parser.add_argument('--lane', type=str, help='')
parser.add_argument('--tile', type=str, help='')
parser.add_argument('--mu_scale', type=float, default=80, help='Coordinate to um translate')
parser.add_argument('--gene_type_info', type=str, help='A file containing two columns, gene name and gene type. Used only if specific types of genes are kept.', default = '')
parser.add_argument('--gene_type_keyword', type=str, help='Key words (separated by ,) of gene types to keep, only used is gene_type_info is provided.', default="IG,TR,protein,lnc")
parser.add_argument('--nFactor', type=int, default=10, help='')
parser.add_argument('--hex_width', type=int, default=24, help='')
parser.add_argument('--hex_radius', type=int, default=-1, help='')
parser.add_argument('--min_pixel_per_unit', type=int, default=50, help='')
parser.add_argument('--min_pixel_per_unit_fit', type=int, default=20, help='')
parser.add_argument('--min_count_per_feature', type=int, default=50, help='')
parser.add_argument('--n_move_hex_tile', type=int, default=-1, help='')
parser.add_argument('--hex_width_fit', type=int, default=18, help='')
parser.add_argument('--hex_radius_fit', type=int, default=-1, help='')
parser.add_argument('--figure_width', type=int, default=20, help="Width of the output figure per 1000um")
parser.add_argument('--cmap_name', type=str, default="nipy_spectral", help="Name of Matplotlib colormap to use")
parser.add_argument('--model_only', action='store_true')
parser.add_argument('--use_stored_model', action='store_true')
parser.add_argument('--skip_analysis', action='store_true')

args = parser.parse_args()

iden=args.identifier
outbase=args.output_path
expr_id=args.experiment_id
lane=args.lane
tile_list=args.tile.split(',')
mu_scale = 1./args.mu_scale

filter_id = ""
if args.filter_criteria_id != '':
    filter_id += "." + args.filter_criteria_id

output_id = expr_id + "." + args.splice + ".nFactor_"+str(args.nFactor) + ".d_"+str(args.hex_width) + ".lane_"+lane+'.'+'_'.join(tile_list)

### Input and output
outpath = '/'.join([outbase,lane])
flt_f = outpath+"/matrix_merged_info.velo.lane_"+lane+'.'+'_'.join(tile_list)+filter_id+".tsv.gz"
if not os.path.exists(flt_f):
    print(f"ERROR: cannot find input file, please run preprocessing script first.")
    sys.exit()

figure_path = outbase + "/analysis/figure"
if not os.path.exists(outpath):
    arg="mkdir -p "+outpath
    os.system(arg)

if not os.path.exists(figure_path):
    arg="mkdir -p "+figure_path
    os.system(arg)

### If work on subset of genes
gene_kept_org = set()
if args.gene_type_info != '' and os.path.exists(args.gene_type_info):
    gencode = pd.read_csv(args.gene_type_info, sep='\t', names=['Name','Type'])
    kept_key = args.gene_type_keyword.split(',')
    kept_type = gencode.loc[gencode.Type.str.contains('|'.join(kept_key)),'Type'].unique()
    gencode = gencode.loc[ gencode.Type.isin(kept_type) ]
    if "MT" not in kept_key:
        gencode = gencode[~gencode.Name.str.contains('mt-')]
    gene_kept_org = set(list(gencode.Name))

### Basic parameterse
b_size = 256 # minibatch size
L=args.nFactor
min_pixel_per_unit=args.min_pixel_per_unit
min_pixel_per_unit_fit=args.min_pixel_per_unit_fit
min_count_per_feature=args.min_count_per_feature

### Read data
try:
    df = pd.read_csv(flt_f, sep='\t')
except:
    df = pd.read_csv(flt_f, sep='\t', compression='bz2')

if len(gene_kept_org) > 0:
    df = df[df.gene.isin(gene_kept_org)]

feature = df[['gene', 'gene_tot_'+args.splice]].drop_duplicates(subset='gene')
feature.rename(columns = {'gene_tot_'+args.splice : 'gene_tot'}, inplace=True)
feature = feature.loc[feature.gene_tot > args.min_count_per_feature, :]
gene_kept = list(feature['gene'])
df = df[df.gene.isin(gene_kept)]

brc = copy.copy(df[['j','X','Y','brc_tot_'+args.splice]]).drop_duplicates(subset='j')
brc.rename(columns = {'brc_tot_'+args.splice : 'brc_tot'}, inplace=True)
brc.index = range(brc.shape[0])
brc['x'] = brc.X.values * mu_scale
brc['y'] = brc.Y.values * mu_scale
pts = np.asarray(brc[['x','y']])
balltree = sklearn.neighbors.BallTree(pts)

# Make DGE
feature_kept = copy.copy(gene_kept)
barcode_kept = list(brc['j'].unique())
bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
ft_dict = {x:i for i,x in enumerate( feature_kept ) }
indx_row = [ bc_dict[x] for x in df['j']]
indx_col = [ ft_dict[x] for x in df['gene']]
N = len(barcode_kept)
M = len(feature_kept)

dge_mtx = coo_matrix((df[args.splice], (indx_row, indx_col)), shape=(N, M)).tocsr()
print(f"Made DGE {dge_mtx.shape}")

# Baseline model training
radius=args.hex_radius
diam=args.hex_width
if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = int(radius*np.sqrt(3))

diam_train = diam

n_move = args.n_move_hex_tile # sliding hexagon
if n_move > diam or n_move < 0:
    n_move = diam // 4

topic_header = ['Topic_'+str(x) for x in range(L)]

model_f = outbase + "/analysis/"+output_id+ ".model.p"
if args.use_stored_model and os.path.exists(model_f):
    lda_base = pickle.load( open( model_f, "rb" ) )
else:
    lda_base = LDA(n_components=L, learning_method='online', batch_size=b_size, n_jobs = 1, verbose = 0)
    offs_x = 0
    offs_y = 0
    epoch = 0
    while offs_x < n_move:
        while offs_y < n_move:
            x,y = pixel_to_hex(pts, radius, offs_x/n_move, offs_y/n_move)
            hex_crd = list(zip(x,y))
            ct = Counter(hex_crd)
            ct = {k:v for k,v in ct.items() if v >= min_pixel_per_unit}
            hex_list = list(ct.keys())
            shuffle(hex_list)
            hex_dict = {x:i for i,x in enumerate(hex_list)}
            sub = pd.DataFrame({'crd':hex_crd,  'cCol':range(pts.shape[0])})
            sub = sub[sub.crd.isin(ct)]
            sub['cRow'] = sub.crd.map(hex_dict)

            n_hex = len(hex_dict)
            hex_id= sub.cRow.values
            Cmtx = coo_matrix((np.ones(sub.shape[0], dtype=bool), (hex_id, sub.cCol.values)), shape=(n_hex, N) ).tocsr()
            mtx = Cmtx @ dge_mtx
            _ = lda_base.partial_fit(mtx)
            # Evaluation (todo: use test set?
            logl = lda_base.score(mtx)
            # Compute topic coherence
            topic_pmi = []
            top_gene_n = np.min(100, mtx.shape[1])
            pseudo_ct = 200
            for k in range(L):
                b = lda_base.exp_dirichlet_component_[k,:]
                b = np.clip(b, 1e-6, 1.-1e-6)
                indx = np.argsort(-b)[:top_gene_n]
                w = 1. - np.power(1.-feature_mf[indx], pseudo_ct)
                w = w.reshape((-1, 1)) @ w.reshape((1, -1))
                p0 = 1.-np.power(1-b, pseudo_ct)
                p0 = p0.reshape((-1, 1)) @ p0.reshape((1, -1))
                pmi = np.log(p0) - np.log(w)
                np.fill_diagonal(pmi, 0)
                pmi = np.round(pmi.mean(), 3)
                topic_pmi.append(pmi)
            print(f"Epoch {epoch}, sliding offset {offs_x}, {offs_y}. Fit data matrix {mtx.shape}, log likelihood {logl:.3E}.")
            print(*topic_pmi, sep = ", ")
            epoch += 1
            offs_y += 1

        offs_y = 0
        offs_x += 1

    lda_base.feature_names_in_ = feature_kept
    pickle.dump( lda_base, open( model_f, "wb" ) )
    if args.model_only:
        sys.exit()

### DE gene based on learned factor profiles
if not args.skip_analysis:
    mtx = lda_base.components_ * (df[args.splice].sum()/lda_base.components_.sum()) # L x M
    mtx = np.around(mtx, 0).astype(int)
    gene_sum = mtx.sum(axis = 0)
    fact_sum = mtx.sum(axis = 1)
    tt = mtx.sum()
    res=[]
    tab=np.zeros((2,2))
    for i, name in enumerate(lda_base.feature_names_in_):
        for l in range(L):
            if mtx[l,i] == 0 or fact_sum[l] == 0:
                continue
            tab[0,0]=mtx[l,i]
            tab[0,1]=gene_sum[i]-tab[0,0]
            tab[1,0]=fact_sum[l]-tab[0,0]
            tab[1,1]=tt-fact_sum[l]-gene_sum[i]+tab[0,0]
            fd=tab[0,0]/fact_sum[l]/tab[0,1]*(tt-fact_sum[l])
            chi2, p, dof, ex = scipy.stats.chi2_contingency(tab, correction=False)
            res.append([name,l,chi2,p,fd])

    chidf=pd.DataFrame(res,columns=['gene','factor','Chi2','pval','FoldChange'])
    res=chidf.loc[(chidf.pval<1e-3)*(chidf.FoldChange>1)].sort_values(by='FoldChange',ascending=False)
    res = res.merge(right = feature, on = 'gene', how = 'inner')
    res.sort_values(by=['factor','FoldChange'],ascending=[True,False],inplace=True)
    res['pval'] = res.pval.map('{:,.3e}'.format)

    f = outbase + "/analysis/"+output_id+".DEgene.tsv.gz"
    res.round(5).to_csv(f,sep='\t',index=False)


diam = args.hex_width_fit
radius = args.hex_radius_fit
if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = radius*np.sqrt(3)
n_move = args.n_move_hex_tile
if n_move > diam or n_move < 0:
    n_move = diam // 4

lda_base_result_full = []

# Apply fitted model
offs_x = 0
offs_y = 0
while offs_x < n_move:
    while offs_y < n_move:
        t0=time.time()
        x,y = pixel_to_hex(pts, radius, offs_x/n_move, offs_y/n_move)
        hex_crd = list(zip(x,y))
        hex_dict = {x:i for i,x in enumerate(list(set(hex_crd)))}
        c_row = [hex_dict[x] for x in hex_crd]
        c_col = range(pts.shape[0])
        Cmtx = coo_matrix( (np.ones(len(c_col), dtype=bool), (c_row, c_col)), shape=(len(hex_dict), N) ).tocsr()
        ct = np.asarray(Cmtx.sum(axis = 1)).squeeze()
        indx = ct >= min_pixel_per_unit_fit

        hex_crd = [0]*Cmtx.shape[0]
        for k,v in hex_dict.items():
            hex_crd[v] = k
        mtx = Cmtx @ dge_mtx
        n_unit = mtx.shape[0]
        prep_time = time.time()-t0

        # perp = lda_base.perplexity(mtx)
        logl = lda_base.score(mtx)
        print(f"Offsets: {offs_x},{offs_y}, log likelihood {logl:.2E}")
        theta = lda_base.transform(mtx)
        lda_base_result_full += [ [offs_x, offs_y, hex_crd[i][0],hex_crd[i][1]] + list(theta[i,]) for i in range(len(hex_crd)) if indx[i] ]
        offs_y += 1
    offs_y = 0
    offs_x += 1

lda_base_result = pd.DataFrame(lda_base_result_full, columns = "offs_x,offs_y,hex_x,hex_y".split(',')+['Topic_'+str(x) for x in range(L)])

lda_base_result['Top_Topic'] = np.argmax(np.asarray(lda_base_result.loc[:, topic_header ]), axis = 1)
lda_base_result['Top_Prob'] = lda_base_result.loc[:, topic_header].max(axis = 1)
lda_base_result['Top_assigned'] = pd.Categorical(lda_base_result.Top_Topic)

# Transform back to pixel location
x,y = hex_to_pixel(lda_base_result.hex_x.values,lda_base_result.hex_y.values,
                   radius,lda_base_result.offs_x.values/n_move,lda_base_result.offs_y.values/n_move)

lda_base_result["Hex_center_x"] = x
lda_base_result["Hex_center_y"] = y

f = outbase + "/analysis/"+output_id+".fit_result.tsv.gz"
lda_base_result.round(5).to_csv(f,sep='\t',index=False)

# Plot clustering result
cmap_name = args.cmap_name
if args.cmap_name not in plt.colormaps():
    cmap_name = "nipy_spectral"
cmap = plt.get_cmap(cmap_name, L)
clist = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(L)]

pt_size = 1000/(pts[:,1].max()-pts[:,1].min()) * 0.5
pt_size = np.round(pt_size,2)
fig_width = int( (pts[:,1].max()-pts[:,1].min()) / 1000 * args.figure_width )
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

f = figure_path + "/"+output_id+".png"
ggsave(filename=f,plot=ps,device='png',limitsize=False)
