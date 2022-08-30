import sys, os, copy, gc, gzip
import pickle, argparse
import numpy as np
import pandas as pd
from random import shuffle

import matplotlib.pyplot as plt
from PIL import Image

from scipy.sparse import *
import scipy.stats
import sklearn.neighbors
import sklearn.preprocessing
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output_path', type=str, help='')
parser.add_argument('--identifier', type=str, help='')
parser.add_argument('--experiment_id', type=str, help='')
parser.add_argument('--lane', type=str, default = '', help='')
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')

parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced, velo: velo total')
parser.add_argument('--gene_type_info', type=str, help='A file containing two columns, gene name and gene type. Used only if specific types of genes are kept.', default = '')
parser.add_argument('--gene_type_keyword', type=str, help='Key words (separated by ,) of gene types to keep, only used is gene_type_info is provided.', default="IG,TR,protein,lnc")
parser.add_argument('--rm_gene_keyword', type=str, help='Key words (separated by ,) of gene names to remove, only used is gene_type_info is provided.', default="")
parser.add_argument('--subset_tile', type=str, default = '', help='')

parser.add_argument('--nFactor', type=int, default=10, help='')
parser.add_argument('--hex_width', type=int, default=24, help='')
parser.add_argument('--hex_radius', type=int, default=-1, help='')
parser.add_argument('--min_ct_per_unit', type=int, default=50, help='')
parser.add_argument('--min_ct_per_unit_fit', type=int, default=20, help='')
parser.add_argument('--min_count_per_feature', type=int, default=1, help='')
parser.add_argument('--n_move_train', type=int, default=-1, help='')
parser.add_argument('--n_move_fit', type=int, default=-1, help='')
parser.add_argument('--hex_width_fit', type=int, default=18, help='')
parser.add_argument('--hex_radius_fit', type=int, default=-1, help='')
parser.add_argument('--thread', type=int, default=1, help='')

parser.add_argument('--figure_width', type=int, default=20, help="Width of the output figure per figure_scale_per_tile um")
parser.add_argument('--cmap_name', type=str, default="nipy_spectral", help="Name of Matplotlib colormap to use")
parser.add_argument('--plot_um_per_pixel', type=float, default=4, help="Size of the output pixels in um")
parser.add_argument("--plot_top", default=False, action='store_true', help="")
parser.add_argument("--plot_fit", default=False, action='store_true', help="")
parser.add_argument("--tif", default=False, action='store_true', help="Store as 16-bit tif instead of png")

parser.add_argument('--use_specific_model', type=str, default="", help="(Temporary) A file containing pre-trained LDA model object")
parser.add_argument('--model_only', action='store_true')
parser.add_argument('--use_stored_model', action='store_true')
parser.add_argument('--skip_analysis', action='store_true')

args = parser.parse_args()

outbase=args.output_path
mu_scale = 1./args.mu_scale

radius=args.hex_radius
diam=args.hex_width
if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = int(radius*np.sqrt(3))
diam_train = diam
output_suffix = args.key + ".nFactor_"+str(args.nFactor) + ".d_"+str(diam_train)

### Input and output
if not os.path.exists(args.input):
    print(f"ERROR: cannot find input file \n {args.input}, please run preprocessing script first.")
    sys.exit()

output_id = args.identifier + "." + args.experiment_id + "." + output_suffix
print(f"Output id: {output_id}")
figure_path = outbase + "/analysis/figure"
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
    if args.rm_gene_keyword != "":
        rm_list = args.rm_gene_keyword.split(",")
        for x in rm_list:
            gencode = gencode.loc[ ~gencode.Name.str.contains(x) ]
    gene_kept_org = set(list(gencode.Name))

### Basic parameterse
b_size = 256 # minibatch size
L=args.nFactor
topic_header = ['Topic_'+str(x) for x in range(L)]

### Read data
df = pd.read_csv(gzip.open(args.input, 'rb'), sep='\t', usecols = ['X','Y','gene', args.key, 'tile'])


### If working on a subset of tiles
if args.subset_tile != '':
    kept_tile = [int(x) for x in args.subset_tile.split(',')]
    df.tile = df.tile.astype(int)
    df = df.loc[df.tile.isin(kept_tile), :]
    df.drop(columns = 'tile', inplace=True)

if len(gene_kept_org) > 0:
    df = df[df.gene.isin(gene_kept_org)]

# df.drop_duplicates(subset=['X','Y','gene'], inplace=True)
feature = df[['gene', args.key]].groupby(by = 'gene', as_index=False).agg({args.key:sum}).rename(columns = {args.key:'gene_tot'})
feature = feature.loc[feature.gene_tot > args.min_count_per_feature, :]
gene_kept = list(feature['gene'])
df = df[df.gene.isin(gene_kept)]
df['j'] = df.X.astype(str) + '_' + df.Y.astype(str)

brc = df.groupby(by = ['j','X','Y']).agg({args.key: sum}).reset_index()
brc.index = range(brc.shape[0])
pixel_ct = brc[args.key].values
pts = np.asarray(brc[['X','Y']]) * mu_scale
balltree = sklearn.neighbors.BallTree(pts)
print(f"Read data with {brc.shape[0]} pixels and {len(gene_kept)} genes.")
df.drop(columns = ['X', 'Y'], inplace=True)

# Make DGE
feature_kept = copy.copy(gene_kept)
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

# Baseline model training
n_move = args.n_move_train # sliding hexagon
if n_move > diam or n_move < 0:
    n_move = diam // 4

model_f = outbase + "/analysis/"+output_id+ ".model.p"
print(f"Output file {model_f}")
if args.use_specific_model !="" and os.path.exists(args.use_specific_model):
    lda_base = pickle.load( open( args.use_specific_model, "rb" ) )
elif args.use_stored_model and os.path.exists(model_f):
    if args.model_only:
        sys.exit()
    lda_base = pickle.load( open( model_f, "rb" ) )
else:
    lda_base = LDA(n_components=L, learning_method='online', batch_size=b_size, n_jobs = args.thread, verbose = 0)
    epoch = 0
    offs_x = 0
    offs_y = 0
    while offs_x < n_move:
        while offs_y < n_move:
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
            print(f"{n_minib}, {n_hex}, median count per unit {mid_ct}")
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
                _ = lda_base.partial_fit(mtx)

                # Evaluation (todo: use test set?
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
                print(f"Epoch {epoch}, minibatch {st_minib}/{n_minib}. Fit data matrix {mtx.shape}, log likelihood {logl:.3E}.")
                print(*topic_pmi, sep = ", ")
            print(f"Epoch {epoch}, sliding offset {offs_x}, {offs_y}. Fit data with {n_hex} units.")
            epoch += 1
            offs_y += 1
        offs_y = 0
        offs_x += 1

    lda_base.feature_names_in_ = feature_kept
    pickle.dump( lda_base, open( model_f, "wb" ) )
    out_f = model_f.replace("model.p", "model_matrix.tsv.gz")
    pd.concat([pd.DataFrame({'gene': lda_base.feature_names_in_}),\
               pd.DataFrame(sklearn.preprocessing.normalize(lda_base.components_, axis = 1, norm='l1').T,\
               columns = ["Factor_"+str(k) for k in range(L)], dtype='float64')],\
               axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.4e')
    if args.model_only:
        sys.exit()

### DE gene based on learned factor profiles
if not args.skip_analysis:
    mtx = sklearn.preprocessing.normalize(lda_base.components_, axis = 0, norm='l1') # L x M
    mtx = mtx * np.array(feature.gene_tot.values).reshape((1, -1))
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
n_move = args.n_move_fit
if args.n_move_fit < 0 and args.n_move_train > 0:
    n_move = args.n_move_train
if n_move > diam or n_move < 0:
    n_move = diam // 4

res_f = outbase + "/analysis/"+output_id+"_"+str(args.hex_width_fit)+".fit_result.tsv"
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
            indx = ct >= args.min_ct_per_unit_fit
            logl = lda_base.score(mtx)
            theta = lda_base.transform(mtx)
            lines = [ [offs_x, offs_y, hex_crd_sub[i][0],hex_crd_sub[i][1]] + list(np.around(theta[i,], 5)) for i in range(theta.shape[0]) if indx[i] ]
            lines = ['\t'.join([str(x) for x in y])+'\n' for y in lines]
            _ = wf.writelines(lines)

            print(f"Minibatch {st_minib} with {sum(indx)} units, log likelihood {logl:.2E}")
            st_minib += 1
        offs_y += 1
        print(f"{offs_x}, {offs_y}")
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

# Output estimates
lda_base_result.round(5).to_csv(res_f,sep='\t',index=False)



# Plot clustering result
cmap_name = args.cmap_name
if args.cmap_name not in plt.colormaps():
    cmap_name = "turbo"

dt = np.uint16 if args.tif else np.uint8
K = args.nFactor

cmap = plt.get_cmap('turbo', K)
cmtx = np.array([cmap(i) for i in range(K)] )[:, :3]
np.random.shuffle(cmtx)

lda_base_result.rename(columns = {'Hex_center_x':'x', 'Hex_center_y':'y'}, inplace=True)
if args.plot_fit:
    lda_base_result.x -= lda_base_result.x.min()
    lda_base_result.y -= lda_base_result.y.min()
lda_base_result['x_indx'] = np.round(lda_base_result.x.values / args.plot_um_per_pixel, 0).astype(int)
lda_base_result['y_indx'] = np.round(lda_base_result.y.values / args.plot_um_per_pixel, 0).astype(int)

if args.plot_top:
    amax = np.array(lda_base_result[topic_header]).argmax(axis = 1)
    lda_base_result[topic_header] = coo_matrix((np.ones(lda_base_result.shape[0],dtype=np.int8), (range(lda_base_result.shape[0]), amax)), shape=(lda_base_result.shape[0], K)).toarray()

lda_base_result = lda_base_result.groupby(by = ['x_indx', 'y_indx']).agg({ x:np.mean for x in topic_header }).reset_index()
h, w = lda_base_result[['x_indx','y_indx']].max(axis = 0) + 1

rgb_mtx = np.clip(np.around(np.array(lda_base_result[topic_header]) @ cmtx * 255).astype(dt),0,255)
img = np.zeros( (h, w, 3), dtype=dt)
for r in range(3):
    img[:, :, r] = coo_matrix((rgb_mtx[:, r], (lda_base_result.x_indx.values, lda_base_result.y_indx.values)), shape=(h,w), dtype = dt).toarray()

if args.tif:
    img_rgb = Image.fromarray(img, mode="I;16")
else:
    img_rgb = Image.fromarray(img)

outf = figure_path + "/"+output_id+"_"+str(args.hex_width_fit)
outf += ".tif" if args.tif else ".png"
img_rgb.save(outf)
