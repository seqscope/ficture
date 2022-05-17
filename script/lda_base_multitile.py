import sys, io, os, copy, re, time, importlib, warnings, subprocess
from collections import defaultdict, Counter

# packages = "networkx,numpy,scipy,sklearn,argparse,plotnine,pandas".split(',')
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

import networkx as nx

from scipy.sparse import *
import scipy.stats
import sklearn.neighbors
import sklearn.cluster
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hexagon_fn
from hexagon_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, help='')
parser.add_argument('--experiment_id', type=str, help='')
parser.add_argument('--filter_criteria_id', type=str, help='Used if filtered and merged data file is to be stored.', default = '')
parser.add_argument('--lane', type=str, help='')
parser.add_argument('--tile', type=str, default='', help='')
parser.add_argument('--tile_id', type=str, default='', help='')
parser.add_argument('--mu_scale', type=float, default=80, help='Coordinate to um translate')
parser.add_argument('--gene_type_info', type=str, help='A file containing two columns, gene name and gene type. Used only if specific types of genes are kept.', default = '')
parser.add_argument('--gene_type_keyword', type=str, help='Key words (separated by ,) of gene types to keep, only used is gene_type_info is provided.', default="IG,TR,protein,lnc")
parser.add_argument('--marker_info', type=str, help='A file containing marker genes, each row is Label;Gene1,Gene2,... Currently each gene can have only one cluster label.', default = '')
parser.add_argument('--white_list', type=str, help='A file containing genes to keep regardless of count', default = '')
parser.add_argument('--upweight_scale', type=int, default=10, help='')
parser.add_argument('--rare_connect_max_radius', type=int, default=8, help='')
parser.add_argument('--nFactor', type=int, default=10, help='')
parser.add_argument('--hex_width', type=int, default=24, help='')
parser.add_argument('--hex_radius', type=int, default=-1, help='')
parser.add_argument('--min_pixel_per_unit', type=int, default=50, help='')
parser.add_argument('--min_pixel_per_unit_fit', type=int, default=20, help='')
parser.add_argument('--min_count_per_feature', type=int, default=50, help='')
parser.add_argument('--n_move_train', type=int, default=-1, help='')
parser.add_argument('--n_move_fit', type=int, default=-1, help='')
parser.add_argument('--hex_width_fit', type=int, default=18, help='')
parser.add_argument('--hex_radius_fit', type=int, default=-1, help='')
parser.add_argument('--figure_width', type=int, default=20, help="Width of the output figure per 1000um")
parser.add_argument('--cmap_name', type=str, default="turbo", help="Name of Matplotlib colormap to use")
parser.add_argument('--identifier', type=str, help='', default='')
parser.add_argument('--model_only', action='store_true')
parser.add_argument('--use_stored_model', action='store_true')
parser.add_argument('--skip_analysis', action='store_true')
parser.add_argument('--upweight_only', action='store_true')
parser.add_argument('--debug',  type=int, default=0)

args = parser.parse_args()

if args.tile == '' and args.tile_id == '':
    print("ERROR: please indicate tiles either use --tile or --tile_id")
    sys.exit()

outbase=args.output_path
expr_id=args.experiment_id
lane=args.lane
tile_id = args.tile_id
mu_scale = 1./args.mu_scale
if tile_id == '':
    tile_id = '_'.join(args.tile.split(','))

reg_iden = "lane_" + lane + "." + tile_id
filter_id = ""
if args.filter_criteria_id != '':
    filter_id += "." + args.filter_criteria_id

radius=args.hex_radius
diam=args.hex_width
if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = int(radius*np.sqrt(3))
area = diam * diam * np.sqrt(3) / 2
diam_train = diam
dilate_r = radius/2
up_weight_fold = args.upweight_scale
max_radius = args.rare_connect_max_radius

output_id = expr_id + ".nFactor_"+str(args.nFactor) + ".d_"+str(int(diam_train)) + "." + reg_iden
print(f"Output id: {output_id}")

use_marker_info = False
marker_dict = {}
marker_list = {}
marker_indx = {}
rare_marker_gene = set()
if args.marker_info != '':
    try:
        with open(args.marker_info, 'r') as rf:
            for line in rf:
                k = line.strip().split(":")[0]
                v = line.strip().split(":")[1].split(",")
                v = [x for x in v if x != '']
                marker_list[k] = []
                for x in v:
                    marker_list[k].append(x)
                    marker_dict[x] = k
    except IOError:
        pass
if len(marker_dict) > 0:
    use_marker_info = True

white_list = set()
if args.white_list != '':
    try:
        with open(args.white_list, 'r') as rf:
            for line in rf:
                white_list.add(line.strip())
        print(f"Read {len(white_list)} genes from white list")
    except IOError:
        pass

### Input and output
outpath = '/'.join([outbase,lane])
flt_f = outpath+"/matrix_merged_info."+reg_iden+filter_id+".tsv.gz"
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
    gene_kept_org = set(list(gencode.Name))

### Basic parameterse
b_size = 256 # minibatch size
L=args.nFactor
min_pixel_per_unit=args.min_pixel_per_unit
min_pixel_per_unit_fit=args.min_pixel_per_unit_fit

### Read data
try:
    df = pd.read_csv(flt_f, sep='\t', usecols = ["Count","X","Y","gene","j"])
except:
    df = pd.read_csv(flt_f, sep='\t', compression='bz2', usecols = ["Count","X","Y","gene","j"])

if len(gene_kept_org) > 0:
    df = df[df.gene.isin(gene_kept_org)]

feature = df[['gene', 'Count']].groupby(by = 'gene', as_index=False).agg({'Count':sum}).rename(columns = {'Count':'gene_tot'})
feature = feature.loc[(feature.gene_tot > args.min_count_per_feature) | feature.gene.isin(marker_dict) | feature.gene.isin(white_list), :]
feature.index = feature.gene.values
feature_kept = feature.gene.values
df = df[df.gene.isin(feature_kept)]
df['x'] = df.X.values * mu_scale
df['y'] = df.Y.values * mu_scale

brc = copy.copy(df[['j','x','y']]).drop_duplicates(subset='j')
brc.index = range(brc.shape[0])
barcode_kept = list(brc['j'])
pts_full = np.asarray(brc[['x','y']])
balltree = sklearn.neighbors.BallTree(pts_full)

# Make DGE
N = len(barcode_kept)
M = len(feature_kept)
bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
ft_dict = {x:i for i,x in enumerate( feature_kept ) }
df['j'] = df.j.map(bc_dict)
df['i'] = df.gene.map(ft_dict)
dge_mtx = coo_matrix((df['Count'], (df['j'], df['i'])), shape=(N, M)).tocsr()
print(f"Made DGE {dge_mtx.shape}")

if use_marker_info:
    for k,v in marker_list.items():
        marker_indx[k] = set([ft_dict[x] for x in v if x in ft_dict])
    marker_count = copy.copy(feature[feature.gene.isin(marker_dict)])
    marker_count_by_type = []
    for k,v in marker_list.items():
        u = [x for x in v if x in feature.index]
        marker_count_by_type.append([k, feature.loc[u, "gene_tot"].sum()])
    marker_count_by_type = pd.DataFrame(marker_count_by_type, columns = ["Type","TotalCount"])
    rare_type = marker_count_by_type.loc[marker_count_by_type.TotalCount < marker_count_by_type.TotalCount.max() * 0.1, "Type"].values
    rare_marker_gene = set([k for k,v in marker_dict.items() if k in feature.index and v in rare_type] )
    rare_marker_indx = [ft_dict[x] for x in rare_marker_gene]
    rare_marker_set = set(rare_marker_indx)
    dge_mtx[:, rare_marker_indx] *= up_weight_fold
    print(f"Read {len(rare_marker_indx)} marker genes from {len(marker_list)} classes")

if args.debug:
    feature_mf = np.asarray(dge_mtx.sum(axis = 0)).reshape(-1)
    feature_mf = feature_mf / feature_mf.sum()

# Baseline model training

n_move = args.n_move_train # sliding hexagon
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
            # Ordinary units
            x,y = pixel_to_hex(pts_full, radius, offs_x/n_move, offs_y/n_move)
            hex_crd = list(zip(x,y))
            ct = Counter(hex_crd)
            ct = {k:v for k,v in ct.items() if v >= min_pixel_per_unit}
            hex_list = list(ct.keys())
            shuffle(hex_list)
            hex_dict = {x:i for i,x in enumerate(hex_list)}
            sub = pd.DataFrame({'crd':hex_crd,  'cCol':range(pts_full.shape[0])})
            sub = sub[sub.crd.isin(ct)]
            sub['cRow'] = sub.crd.map(hex_dict)
            mtx = coo_matrix((np.ones(sub.shape[0], dtype=bool), (sub.cRow.values, sub.cCol.values)), shape=(len(hex_dict), N) ).tocsr() @ dge_mtx
            if use_marker_info and not args.upweight_only:
                # Rare-marker enriched units
                rare_df = copy.copy(df[df.gene.isin(rare_marker_gene)])
                rare_df['Index'] = range(rare_df.shape[0])
                rare_pts = np.asarray(rare_df[['x', 'y']] )
                rmtx = sklearn.neighbors.radius_neighbors_graph(rare_pts,radius=max_radius,mode = "connectivity")
                G = nx.from_scipy_sparse_matrix(rmtx)
                node_rm = set([k for k,v in G.degree() if v == 0])
                G.remove_nodes_from(node_rm)
                rare_df = rare_df[~rare_df["Index"].isin(node_rm)]
                rare_df['Label'] = rare_df.gene.map(marker_dict)
                nx.set_node_attributes(G, { x["Index"]:x["Label"] for i,x in rare_df.iterrows() }, "Label" )
                edge_rm = []
                for e in G.edges:
                    if G.nodes[e[0]]['Label'] != G.nodes[e[1]]['Label']:
                        edge_rm.append(e[:2])
                G.remove_edges_from(edge_rm)
                cc_list = []
                for c in sorted(nx.connected_components(G), key=len, reverse=True):
                    if len(c) == 1:
                        break
                    v = list(c)
                    cc_list.append([v, G.nodes[v[0]]['Label']])
                seed_unit_label = []
                seed_unit_row = []
                seed_unit_col = []
                up_w = []
                if args.debug:
                    print(f"Detected {len(cc_list)} seed units for marker genes")
                for i,c in enumerate(cc_list):
                    xl,yl = rare_pts[c[0], :].min(axis = 0)
                    xu,yu = rare_pts[c[0], :].max(axis = 0)
                    size_r = (xu-xl) * (yu-yl) / area
                    if size_r > 2:
                        k = int(size_r)
                        indx = c[0]
                        kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(rare_pts[c[0], :])
                        sub_indx = []
                        for j in range(k):
                            sub_indx.append([])
                        for j,x in enumerate(kmeans.labels_):
                            sub_indx[x].append(indx[j])
                        cc_list[i][0] = sub_indx[0]
                        for j in range(1, k):
                            cc_list.append([sub_indx[j], c[1]])
                        if args.debug > 1:
                            print("Devide big seed")
                            print(c, sub_indx)
                        c = cc_list[i]
                    indx = balltree.query_radius(rare_pts[c[0], :], r=dilate_r, return_distance=False)
                    indx = list(set( [item for sublist in indx for item in sublist] ))
                    seed_unit_label.append([i, c[1]])
                    seed_unit_row += [i] * len(indx)
                    seed_unit_col += indx
                    up_w += [(v['j'], v['i']) for r,v in df.loc[df.j.isin(indx) & df.i.isin(rare_marker_set - marker_indx[c[1]]), :].iterrows()]
                    if args.debug and i % 100 == 0:
                        print(f"Processed {i} seed units. {len(up_w)} mixed pixels, average {len(seed_unit_col)//(i+1)} pixels per unit")
                seed_unit = pd.DataFrame(seed_unit_label, columns = ["Index","Label"])
                n_seed = seed_unit.shape[0]
                up_w = np.array(up_w, dtype=int)
                w_mtx = coo_matrix(((np.ones(up_w.shape[0])*(up_weight_fold-1)/up_weight_fold).astype(int), (up_w[:, 0], up_w[:, 1])), shape=(N, M) ).tocsr()
                cmtx = coo_matrix((np.ones(len(seed_unit_row), dtype=bool),\
                                       (seed_unit_row, seed_unit_col)), shape=(n_seed, N) ).tocsr()
                mtx_rare = cmtx @ dge_mtx - cmtx @ dge_mtx.multiply(w_mtx)
                mtx_rare.data = np.clip(mtx_rare.data, 0, mtx_rare.data.max())
                if args.debug:
                    size_regular = int(np.median(np.asarray(mtx.sum(axis = 1)).reshape(-1)))
                    size_rare = int(np.median(np.asarray(mtx_rare.sum(axis = 1)).reshape(-1)))
                    print(f"Training data:\n{mtx.shape[0]} regular units with median size {size_regular} v.s.\n{mtx_rare.shape[0]} marker-enhanced units with median size {size_rare}")
                mtx = vstack([mtx, mtx_rare])
                shuffle_row = list(range(mtx.shape[0]))
                shuffle(shuffle_row)
                mtx = mtx[shuffle_row, :]

            _ = lda_base.partial_fit(mtx)

            if args.debug:
                # Compute likelihood
                logl = lda_base.score(mtx)
                # Compute topic coherence
                topic_pmi = []
                top_gene_n = np.min([100, mtx.shape[1]])
                pseudo_ct = 200
                for k in range(L):
                    b = lda_base.exp_dirichlet_component_[k,:]
                    b = np.clip(b, 1e-6, 1.-1e-6)
                    indx = np.argsort(-b)[:top_gene_n]
                    w = 1. - np.power(1.-feature_mf[indx], pseudo_ct)
                    w = w.reshape((-1, 1)) @ w.reshape((1, -1))
                    p0 = 1.-np.power(1-b[indx], pseudo_ct)
                    p0 = p0.reshape((-1, 1)) @ p0.reshape((1, -1))
                    pmi = np.log(p0) - np.log(w)
                    np.fill_diagonal(pmi, 0)
                    pmi = np.round(pmi.mean(), 3)
                    topic_pmi.append(pmi)
                print(f"Epoch {epoch}, sliding offset {offs_x}, {offs_y}. Fit data matrix {mtx.shape}, log likelihood {logl:.3E}.")
                print(*topic_pmi, sep = ", ")
            else:
                print(f"Epoch {epoch}, sliding offset {offs_x}, {offs_y}. Fit data matrix {mtx.shape}.")
            epoch += 1
            offs_y += 1

        offs_y = 0
        offs_x += 1

    lda_base.feature_names_in_ = feature_kept
    pickle.dump( lda_base, open( model_f, "wb" ) )
    if args.model_only:
        sys.exit()

diam = args.hex_width_fit
radius = args.hex_radius_fit
if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = radius*np.sqrt(3)

if args.n_move_fit > 0:
    n_move = args.n_move_fit
if n_move > diam or n_move < 0:
    n_move = diam // 4

lda_base_result_full = []

# Apply fitted model
offs_x = 0
offs_y = 0
while offs_x < n_move:
    while offs_y < n_move:
        t0=time.time()
        x,y = pixel_to_hex(pts_full, radius, offs_x/n_move, offs_y/n_move)
        hex_crd = list(zip(x,y))
        hex_dict = {x:i for i,x in enumerate(list(set(hex_crd)))}
        c_row = [hex_dict[x] for x in hex_crd]
        c_col = range(pts_full.shape[0])
        Cmtx = coo_matrix( (np.ones(len(c_col), dtype=bool), (c_row, c_col)), shape=(len(hex_dict), N) ).tocsr()
        ct = np.asarray(Cmtx.sum(axis = 1)).squeeze()
        indx = ct >= min_pixel_per_unit_fit

        hex_crd = [0]*Cmtx.shape[0]
        for k,v in hex_dict.items():
            hex_crd[v] = k
        mtx = Cmtx @ dge_mtx
        n_unit = mtx.shape[0]
        prep_time = time.time()-t0

        if args.debug:
            # perp = lda_base.perplexity(mtx)
            logl = lda_base.score(mtx)
            print(f"Offsets: {offs_x},{offs_y}, log likelihood {logl:.2E}")
        else:
            print(f"Offsets: {offs_x},{offs_y}")
        theta = lda_base.transform(mtx)
        lda_base_result_full += [ [offs_x, offs_y, hex_crd[i][0],hex_crd[i][1]] + list(theta[i,]) for i in range(len(hex_crd)) if indx[i] ]
        offs_y += 1
    offs_y = 0
    offs_x += 1

lda_base_result = pd.DataFrame(lda_base_result_full, columns = "offs_x,offs_y,hex_x,hex_y".split(',')+['Topic_'+str(x) for x in range(L)])
lda_base_result.index = range(lda_base_result.shape[0])
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
print("Finish applying model")

### DE gene based on learned factor profiles
if not args.skip_analysis:
    dge_mtx = dge_mtx.tocsc()
    if use_marker_info and len(rare_marker_indx) > 0:
        for j in rare_marker_indx:
            dge_mtx[:, j].data = dge_mtx[:, j].data // up_weight_fold
    reftree = sklearn.neighbors.BallTree(np.asarray(lda_base_result[["Hex_center_x","Hex_center_y"]]))
    indx = reftree.query(pts_full, k=1, return_distance=False)
    indx = np.array(indx).reshape(-1)
    mtx = coo_matrix((lda_base_result.Top_Prob.iloc[indx], (lda_base_result.Top_Topic.iloc[indx].values, range(N))), shape=(L,N)).tocsr() @ dge_mtx # L x M
    mtx = np.around(mtx.toarray(), 0).astype(int)
    gene_sum = mtx.sum(axis = 0)
    fact_sum = mtx.sum(axis = 1)
    tt = mtx.sum()
    res=[]
    tab=np.zeros((2,2))
    for i, name in enumerate(lda_base.feature_names_in_):
        for l in range(L):
            if fact_sum[l] == 0:
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
    print("Finish DE")

# Plot clustering result
cmap_name = args.cmap_name
if args.cmap_name not in plt.colormaps():
    cmap_name = "turbo"
cmap = plt.get_cmap(cmap_name, L)
clist = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(L)]

pt_size = 1000/(pts_full[:,1].max()-pts_full[:,1].min()) * 0.5
pt_size = np.round(pt_size,2)
fig_width = int( (pts_full[:,1].max()-pts_full[:,1].min()) / 1000 * args.figure_width )
plotnine.options.figure_size = (fig_width, fig_width)
with warnings.catch_warnings(record=True):
    ps = (
        ggplot(lda_base_result,
               aes(x='Hex_center_y', y='Hex_center_x',
                                    color='Top_assigned',alpha='Top_Prob'))
        +geom_point(size = pt_size, shape='o')
        +guides(colour = guide_legend(override_aes = {'size':3,'shape':'o'}))
        # +guides(color=None,alpha=None)
        +xlab("")+ylab("")
        +guides(alpha=None)
        +coord_fixed(ratio = 1)
        +scale_color_manual(values = clist)
        +theme_bw()
        +theme(legend_position='bottom')
    )

f = figure_path + "/"+output_id+".png"
ggsave(filename=f,plot=ps,device='png',limitsize=False)
