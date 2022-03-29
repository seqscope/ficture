import sys, io, os, copy, re, time, importlib, warnings, subprocess

# packages = "numpy,scipy,sklearn,argparse,pandas,plotnine,matplotlib".split(',')
# for pkg in packages:
#     if not pkg in sys.modules:
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pkg])

import pickle, argparse
import numpy as np
import pandas as pd
from random import shuffle

import matplotlib.pyplot as plt
from plotnine import *
import plotnine
import matplotlib

from scipy.sparse import *
import sklearn.neighbors
from sklearn.decomposition import LatentDirichletAllocation as LDA
import sklearn

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hexagon_fn
from hexagon_fn import *

import online_slda
import scorpus
from online_slda import *
import utilt

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help='')
parser.add_argument('--identifier', type=str, help='')
parser.add_argument('--model_iden', type=str, help='')
parser.add_argument('--filter_criteria_id', type=str, help='Used to identified input file.', default = '')
parser.add_argument('--lane', type=str, help='')
parser.add_argument('--tile', type=str, help='')
parser.add_argument('--mu_scale', type=float, default=80, help='Coordinate to um translate')
parser.add_argument('--collapse_resolution', type=float, default=0.5, help='Collapse pixels within x um')
parser.add_argument('--k_nn', type=int, default=12, help='')
parser.add_argument('--max_radius', type=float, default=8, help='')
parser.add_argument('--gaussian_sig', type=float, default=5, help='')
parser.add_argument('--figure_width', type=int, default=20, help='')
parser.add_argument('--cmap_name', type=str, default="nipy_spectral", help="Name of Matplotlib colormap to use")
parser.add_argument('--model_f', default='', type=str, help='')
parser.add_argument('--flt_f', default='', type=str, help='')
parser.add_argument('--rectangle', default='', type=str, help='')

args = parser.parse_args()

iden=args.identifier
outbase=args.input_path
lane=args.lane
tile_list=args.tile.split(',')
mu_scale = 1./args.mu_scale
m_res=args.collapse_resolution

tile_iden = "lane_"+lane+'.'+'_'.join(tile_list)
output_id = args.model_iden + "." + tile_iden
filter_id = ""
if args.filter_criteria_id != '':
    filter_id += "." + args.filter_criteria_id

### Basic parameterse
k_nn = args.k_nn # Each pixel sees k grid points
max_radius = args.max_radius
sig = args.gaussian_sig

out_suff = '_'.join([str(x) for x in ['refine', args.k_nn, int(args.max_radius), int(args.gaussian_sig)]])
if out_suff == "refine_12_8_5":
    out_suff = "refine"

### Input and output
outpath = '/'.join([outbase,lane])
if args.flt_f != '':
    flt_f = args.flt_f
else:
    flt_f = outpath+"/matrix_merged_info."+tile_iden+filter_id+".tsv.gz"

if not os.path.exists(flt_f):
    print(f"ERROR: cannot find input file.")
    sys.exit()

figure_path = outbase + "/analysis/figure"
if not os.path.exists(figure_path):
    os.system("mkdir -p "+figure_path)

if args.model_f != '':
    m_f = args.model_f
    r_f = m_f.replace(".model.p", ".fit_result.tsv.gz")
else:
    m_f = outbase + '/analysis/'+args.model_iden+"."+tile_iden+".model.p"
    r_f = outbase + '/analysis/'+args.model_iden+"."+tile_iden+".fit_result.tsv.gz"
if not os.path.exists(m_f) or not os.path.exists(r_f):
    print(m_f)
    print(r_f)
    print(f"ERROR: cannot find input model file.")
    sys.exit()

lda_base = pickle.load( open( m_f, "rb" ) )
gene_kept = lda_base.feature_names_in_
L = lda_base.components_.shape[0]
topic_header = ["Topic_"+str(x) for x in range(L)]
gd = {x:i for i,x in enumerate(lda_base.feature_names_in_)}
m_indx = [gd[x] for x in gene_kept]
_lambda = lda_base.components_[:, m_indx]
_eta = lda_base.components_[:, m_indx].mean(axis = 0)
_eta = _eta / _eta.sum() * 100

lda_base_result = pd.read_csv(r_f,sep='\t')
lda_base_result.Top_assigned = pd.Categorical(lda_base_result.Top_Topic.astype(int))
lda_base_result['x'] = lda_base_result.Hex_center_x.values
lda_base_result['y'] = lda_base_result.Hex_center_y.values
prior_D = lda_base_result.shape[0]//2
print("Read model file")

tpop = lda_base_result[topic_header].sum(axis = 0)
label_sort = np.argsort(tpop)
cmap_name = args.cmap_name
if args.cmap_name not in plt.colormaps():
    cmap_name = "nipy_spectral"
cmap = plt.get_cmap(cmap_name, L)
clist = {}
for i in range(L):
    clist[label_sort[i]] = matplotlib.colors.rgb2hex(cmap(L-1-i))

try:
    df = pd.read_csv(flt_f, sep='\t')
except:
    df = pd.read_csv(flt_f, sep='\t', compression='bz2')

if args.rectangle != '':
    rec = args.rectangle.split(',')
    rmin,rmax,cmin,cmax = [int(x) for x in rec]
    indx = (df.row >= rmin) & (df.row <= rmax) & (df.col >= cmin) & (df.col <= cmax)
    df = df.loc[indx, :]

df = df[df.gene.isin(gene_kept)]
df['x'] = df.X.values * mu_scale
df['y'] = df.Y.values * mu_scale
# Collapse into bigger pixels, so more pixels contain more than one genes
df['x'] = np.around(df.x.values/m_res, 0).astype(int)
df['y'] = np.around(df.y.values/m_res, 0).astype(int)
df = df.groupby(by=['x','y','gene']).agg({'Count':np.sum}).reset_index()
df['j'] = df.x.astype(str)+'_'+df.y.astype(str)
df.x = df.x * m_res
df.y = df.y * m_res

feature = df.groupby(by = 'gene', as_index=False).agg({'Count':np.sum}).rename(columns = {'Count':'gene_tot'})

brc = df.groupby(by='j',as_index=False).agg({'Count':np.sum}).rename(columns = {'Count':'brc_tot'})
brc = brc.merge(right = df[['j','x','y']].drop_duplicates(subset='j'), on='j', how='left')
brc.index = range(brc.shape[0])
pts_full = np.asarray(brc[['x','y']])
balltree = sklearn.neighbors.BallTree(pts_full)

# Make DGE
feature_kept = gene_kept
barcode_kept = list(brc['j'])
bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
ft_dict = {x:i for i,x in enumerate( feature_kept ) }
indx_row = [ bc_dict[x] for x in df['j']]
indx_col = [ ft_dict[x] for x in df['gene']]
N = len(barcode_kept)
M = len(feature_kept)

dge_mtx = coo_matrix((df['Count'], (indx_row, indx_col)), shape=(N, M)).tocsr()
print(f"Made DGE {dge_mtx.shape}")




x_min, y_min = pts_full.min(axis = 0).astype(int)
x_max, y_max = pts_full.max(axis = 0).astype(int)
indx=(lda_base_result.x>=x_min-10)&(lda_base_result.x<=x_max+10)&(lda_base_result.y>=y_min-10)&(lda_base_result.y<=y_max+10)
lda_base_result=lda_base_result.loc[indx, :]
print(lda_base_result.x.max()-lda_base_result.x.min(),lda_base_result.y.max()-lda_base_result.y.min())

# Split into spatially close minibatches (overlapping windows)
batch_size_um = 150 # at most x^2um
batch_ovlp_um = 20
batch_step_um = batch_size_um - batch_ovlp_um
n_batch_x = int((x_max - x_min - batch_ovlp_um)/batch_step_um) + 1
n_batch_y = int((y_max - y_min - batch_ovlp_um)/batch_step_um) + 1
batch_size_x = int((x_max - x_min - batch_ovlp_um)/n_batch_x)+1
batch_size_y = int((y_max - y_min - batch_ovlp_um)/n_batch_y)+1
batch_step_x = batch_size_x - batch_ovlp_um
batch_step_y = batch_size_y - batch_ovlp_um
x_grd = np.arange( x_min, x_max+1, batch_step_x)
y_grd = np.arange( y_min, y_max+1, batch_step_y)
x_grd[-1] = x_max+1
y_grd[-1] = y_max+1
print(x_grd)
print(y_grd)

slda = OnlineLDA(gene_kept, L, prior_D, alpha=1./L, eta=_eta, verbose = 1)
slda.init_global_parameter(_lambda)

pixel_result = pd.DataFrame()
center_result = pd.DataFrame()
for offset in [0, 1]:
    grid_pts_full = copy.copy(lda_base_result[(lda_base_result.offs_x.values%2==offset)&(lda_base_result.offs_y.values%2==offset)])
    grid_pts_full.index = range(grid_pts_full.shape[0])
    for iter_i in range(1, len(x_grd)):
        skip = 0
        x_min = x_grd[iter_i-1] - batch_ovlp_um//2
        x_max = x_grd[iter_i]   + batch_ovlp_um//2
        for iter_j in range(1, len(y_grd)):
            if skip == 0:
                y_min = y_grd[iter_j-1] - batch_ovlp_um//2
            y_max = y_grd[iter_j]   + batch_ovlp_um//2

            grid_indx = (grid_pts_full.x >= x_min) & (grid_pts_full.x <= x_max) & (grid_pts_full.y >= y_min) & (grid_pts_full.y <= y_max)
            pixel_indx= (pts_full[:,0]>=x_min)&(pts_full[:,0]<=x_max)&(pts_full[:,1]>=y_min)&(pts_full[:,1]<=y_max)

            pts = copy.copy(pts_full[pixel_indx, :])
            grid_pts = copy.copy(grid_pts_full.loc[grid_indx, :])

            doc_pts = np.asarray(grid_pts[['x', 'y']])
            if doc_pts.shape[0] < 50:
                skip = 1
                print(["Skip",x_min,x_max,y_min,y_max,doc_pts.shape[0]])
                continue
            skip = 0
            print([iter_i,iter_j,x_min,x_max,y_min,y_max,doc_pts.shape[0]])
            balltree_ref = sklearn.neighbors.BallTree(doc_pts)
            p_mtx = np.asarray(grid_pts[topic_header])

            N = pts.shape[0]
            n = doc_pts.shape[0]
            d_size = N/n

            dist, indx_assign = balltree_ref.query(X=pts, k=k_nn, return_distance=True, sort_results=False)
            indx_row = [i for i in range(len(indx_assign)) for j in range(len(indx_assign[i]))]
            indx_col = [item for sublist in indx_assign for item in sublist]
            dist = np.array([item for sublist in dist for item in sublist])

            psi_org = coo_matrix((dist, (indx_row, indx_col)), shape=[N, n]).tocsr()
            psi_org.data = np.exp( -dist**2 / sig**2 / 2 )
            psi_org.data[psi_org.data > max_radius] = 0
            psi_org.eliminate_zeros()

            # psi_org = sklearn.preprocessing.normalize(psi_org, norm='l1', axis=1)
            psi_org = sklearn.preprocessing.normalize(psi_org, norm='max', axis=1)
            phi_org = psi_org @ p_mtx
            phi_org = sklearn.preprocessing.normalize(phi_org, norm='l1', axis=1)
            # phi_org = sklearn.preprocessing.normalize(phi_org, norm='max', axis=1)

            batch = scorpus.corpus()
            batch.init_from_matrix(dge_mtx[pixel_indx,:], doc_pts, psi_org, m_phi=phi_org, m_gamma=p_mtx*d_size/3)

            scores = slda.update_lambda(batch)
            pixel_result = pd.concat([pixel_result,
                                      pd.concat([copy.copy(brc.loc[pixel_indx, ['j']]).reset_index(),\
                                                 pd.DataFrame(batch.phi, columns =\
                                                            ['Factor_'+str(x) for x in range(slda._K)])],\
                                                 axis = 1)])

            expElog_theta = np.exp(utilt.dirichlet_expectation(batch.gamma))
            expElog_theta /= expElog_theta.sum(axis = 1)[:, np.newaxis]
            tmp = pd.DataFrame({'x':doc_pts[:,0],'y':doc_pts[:,1]})
            tmp['center_id'] = grid_pts_full.loc[grid_indx, :].index.values
            tmp['center_id'] = str(offset) + '_' + tmp.center_id.astype(str)
            tmp['Avg_size'] = np.array(batch.psi.sum(axis = 0)).reshape(-1)
            for v in range(slda._K):
                tmp['Factor_'+str(v)] = expElog_theta[:, v]
            center_result = pd.concat([center_result, tmp])


factor_header = ['Factor_'+str(x) for x in range(slda._K)]
pixel_result = pixel_result.groupby(by=['j'], as_index=False).agg({x:np.mean for x in factor_header})
pixel_result = pixel_result.merge(right = brc, on = 'j', how = 'inner')
pixel_result['Top_Factor'] = np.asarray(pixel_result[factor_header]).argmax(axis = 1)
pixel_result['Top_Prob'] = np.asarray(pixel_result[factor_header]).max(axis = 1)
pixel_result['Top_assigned'] = pd.Categorical(pixel_result.Top_Factor.values, categories=range(slda._K))

f = outbase + "/analysis/"+output_id+"."+out_suff+".pixel.tsv"
pixel_result.to_csv(f,sep='\t',index=False)

plotnine.options.figure_size = (args.figure_width,args.figure_width)
with warnings.catch_warnings(record=True):
    ps = (
        ggplot(pixel_result,
               aes(x='y', y='x', color='Top_assigned',alpha='Top_Prob'))
        +geom_point(size = 0.1, shape='+')
        +guides(colour = guide_legend(override_aes = {'size':4,'shape':'o'}))
        +xlab("")+ylab("")
        +guides(alpha=None)
        +coord_fixed(ratio = 1)
        +scale_color_manual(values = clist)
        +theme_bw()
    )
    f = figure_path + "/"+output_id+"."+out_suff+".pixel.png"
    ggsave(filename=f,plot=ps,device='png')


center_result.sort_values(by = 'Avg_size', ascending=False, inplace=True)
center_info = center_result[['center_id', 'x', 'y']].drop_duplicates(subset=['center_id'])
center_result = center_result.groupby(by=['center_id']).agg({x:np.mean for x in factor_header + ['Avg_size']}).reset_index()
center_result = center_result.merge(right = center_info, on=['center_id'], how='inner')
center_result['Top_Factor'] = np.asarray(center_result[factor_header]).argmax(axis = 1)
center_result['Top_Prob'] = np.asarray(center_result[factor_header]).max(axis = 1)
center_result['Top_assigned'] = pd.Categorical(center_result.Top_Factor.values, categories=range(slda._K))

f = outbase + "/analysis/"+output_id+"."+out_suff+".center.tsv"
center_result.to_csv(f,sep='\t',index=False)

pt_size = 1000/(pts_full[:,1].max()-pts_full[:,1].min()) * 0.5
pt_size = np.round(pt_size,2)
plotnine.options.figure_size = (args.figure_width,args.figure_width)
with warnings.catch_warnings(record=True):
    ps = (
        ggplot(center_result[center_result.Avg_size > 10],
               aes(x='y', y='x',
                   color='Top_assigned',alpha='Top_Prob'))
        +geom_point(size = pt_size, shape='o')
        +guides(colour = guide_legend(override_aes = {'size':4,'shape':'o'}))
        +xlab("")+ylab("")
        +guides(alpha=None)
        +coord_fixed(ratio = 1)
        +scale_color_manual(values = clist)
        +theme_bw()
    )
    f = figure_path + "/"+output_id+"."+out_suff+".center.png"
    ggsave(filename=f,plot=ps,device='png')
