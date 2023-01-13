import sys, os, argparse, logging, gzip, glob, copy, re, time, importlib, warnings, pickle
import subprocess as sp
import numpy as np
import pandas as pd

from scipy.sparse import *
import sklearn.neighbors
import sklearn.preprocessing
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *
from online_slda import *
import scorpus, utilt

parser = argparse.ArgumentParser()

# Innput and output info
parser.add_argument('--input', type=str, help='')
parser.add_argument('--model', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--anchor', type=str, default='', help='')

# Data realted parameters
parser.add_argument('--anchor_radius', type=float, default=15, help='Radius to initialize anchor points. Only used if --anchor file is not available')
parser.add_argument('--anchor_resolution', type=float, default=3, help='Distance (um) between two neighboring anchor point. Only used if --anchor file is not available')
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--precision', type=float, default=.25, help='If positive, collapse pixels within X um.')
parser.add_argument('--key', type=str, default = 'gn', help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')

# Learning related parameters
parser.add_argument('--total_pixel', type=float, default=1e5, help='(An estimate of) total number of pixels just for calculating the learning rate')
parser.add_argument('--neighbor_radius', type=float, default=25, help='The radius (um) of each anchor point\'s territory')
parser.add_argument('--min_ct_per_unit', type=int, default=10, help='Keep anchor points with at least x reads inside its territory during initialization. Would only be used if --anchor file is not provided')
parser.add_argument('--halflife', type=float, default=0.7, help='Control the decay of distance-based weight')

# Other
parser.add_argument('--log', type=str, default = '', help='files to write log to')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

if args.log != '':
    try:
        logging.basicConfig(filename=args.log, filemode='a', encoding='utf-8', level=logging.INFO)
    except:
        logging.basicConfig(level= getattr(logging, "INFO", None))
else:
    logging.basicConfig(level= getattr(logging, "INFO", None))

if not os.path.exists(args.model):
    sys.exit("ERROR: cannot find model file")
if not os.path.exists(args.input):
    sys.exit("ERROR: cannot find input file")
use_input_anchor = False
grid_info = pd.DataFrame()
if os.path.exists(args.anchor):
    grid_info = pd.read_csv(args.anchor,sep='\t')
    use_input_anchor = True

### Basic parameterse
mu_scale = 1./args.mu_scale
radius = args.anchor_radius
key = args.key
nu = np.log(.5) / np.log(args.halflife)
out_buff = args.neighbor_radius * args.halflife

### Load model
model = pickle.load(open( args.model, "rb" ))
gene_kept = model.feature_names_in_
model.feature_names_in_ = None
ft_dict = {x:i for i,x in enumerate( gene_kept ) }
K, M = model.components_.shape
factor_header = ['Topic_'+str(x) for x in range(K)]
logging.info(f"{M} genes and {K} factors are read from input model")

### Input pixel info (input has to have correct header)
input_header = ["random_index","X","Y","gene",key]
dty = {x:int for x in ['X','Y',key]}
dty.update({x:str for x in ['random_index', 'gene']})

### Model fitting
betaksum = model.components_.sum(axis = 1)
global_scale = np.min([1, 1000 / np.median(betaksum)])
_lambda = model.components_ * global_scale
print(global_scale, np.median(_lambda.sum(axis=1)), sorted(np.around(betaksum*global_scale, 0)) )
slda = OnlineLDA(vocab=gene_kept, K=K, N=args.total_pixel,
                 iter_inner=30, verbose = 1)
slda.init_global_parameter(_lambda)

df_full = pd.DataFrame()
n_batch = 0
chunk_size = 1000000
if args.debug:
    chunk_size = 100000
for chunk in pd.read_csv(args.input, sep='\t', chunksize=chunk_size,\
                        header = 0, usecols=input_header, dtype=dty):
    chunk = chunk[chunk[key] > 0]
    df_full = pd.concat([df_full, chunk])
    last_indx = df_full.random_index.iloc[-1]
    left = copy.copy(df_full.loc[df_full.random_index.eq(last_indx), :])
    df_full = df_full.loc[~df_full.random_index.eq(last_indx), :]
    df_full = df_full.loc[df_full.gene.isin(gene_kept), :]
    if df_full.shape[0] == 0:
        continue

    batch_index = list(df_full.random_index.unique() )
    df_full['j'] = ""
    if args.precision <= 0:
        df_full['j'] = df_full.X.astype(str) + '_' + df_full.Y.astype(str)
    df_full["X"] = df_full.X * mu_scale
    df_full["Y"] = df_full.Y * mu_scale
    if args.precision > 0:
        df_full["X"] = (df_full.X / args.precision).astype(int)
        df_full["Y"] = (df_full.Y / args.precision).astype(int)
        df_full = df_full.groupby(by=["random_index","gene","X","Y"]).agg({key:np.sum}).reset_index()
        df_full['j'] = df_full.X.astype(str) + '_' + df_full.Y.astype(str)
        df_full.X = df_full.X * args.precision
        df_full.Y = df_full.Y * args.precision

    brc = df_full.groupby(by = ['j'], as_index=False).agg({key:np.sum})
    N0 = brc.shape[0]
    brc.index = range(N0)
    brc = brc.merge(right = df_full.loc[:, ["j","X","Y"]].drop_duplicates(subset='j'), on = "j", how = 'left')
    barcode_kept = list(brc['j'])
    bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
    df_full = df_full.groupby(by = ["random_index", "j", "gene"]).agg({key:sum}).reset_index()
    df_full = df_full.merge(right = brc.loc[:, ["j","X","Y"]], on = 'j', how = 'inner')
    indx_row = [ bc_dict[x] for x in df_full['j']]
    indx_col = [ ft_dict[x] for x in df_full['gene']]
    dge_mtx = coo_array((df_full[key], (indx_row, indx_col)), shape=(N0, M)).tocsr()
    logging.info(f"Read pixels in {len(batch_index)} minibatches. Made DGE {dge_mtx.shape}")
    pts = np.asarray(brc[['X','Y']])
    bt = sklearn.neighbors.BallTree(pts)

    if not use_input_anchor:
        # Need to initialize anchor points
        n_move = int(np.round(radius * np.sqrt(3) / args.anchor_resolution, 0) )
        offs_x = 0
        offs_y = 0
        while offs_x < n_move:
            while offs_y < n_move:
                x,y = pixel_to_hex(np.array(brc[['X','Y']]), radius, offs_x/n_move, offs_y/n_move)
                hex_pt  = pd.DataFrame({'hex_x':x,'hex_y':y,'ct':brc[key].values}).groupby(by=['hex_x','hex_y']).agg({'ct':np.sum}).reset_index()
                hex_pt['x'], hex_pt['y'] = hex_to_pixel(hex_pt.hex_x.values, hex_pt.hex_y.values, radius, offs_x/n_move, offs_y/n_move)
                hex_pt = hex_pt.loc[hex_pt.ct >= args.min_ct_per_unit, :]
                if hex_pt.shape[0] < 2:
                    offs_y += 1
                    continue
                hex_list = list(zip(hex_pt.hex_x.values, hex_pt.hex_y.values))
                hex_crd  = list(zip(x,y))
                hex_dict = {x:i for i,x in enumerate(hex_list)}
                indx = [i for i,x in enumerate(hex_crd) if x in hex_dict]
                hex_crd = [hex_crd[i] for i in indx]
                sub = pd.DataFrame({'cRow':[hex_dict[x] for x in hex_crd], 'cCol':indx, 'hexID':hex_crd})
                nunit = len(hex_dict)
                n_pixel = sub.shape[0]
                mtx = coo_matrix((np.ones(n_pixel, dtype=bool),\
                        (sub.cRow.values, sub.cCol.values)),\
                        shape=(nunit,N0) ).tocsr() @ dge_mtx
                theta = model.transform(mtx)
                lines = pd.DataFrame({'offs_x':offs_x,'offs_y':offs_y, 'hex_x':hex_pt.hex_x.values, 'hex_y':hex_pt.hex_y.values})
                lines['x'], lines['y'] = hex_to_pixel(hex_pt.hex_x.values,hex_pt.hex_y.values, radius, offs_x/n_move, offs_y/n_move)
                lines = pd.concat((lines, pd.DataFrame(theta, columns = factor_header)), axis = 1)
                lines['topK'] = np.argmax(theta, axis = 1).astype(int)
                lines['topP'] = np.max(theta, axis = 1)
                grid_info = pd.concat([grid_info, lines])
                offs_y += 1
                logging.info(f"Initializing... {offs_x}, {offs_y}")
            offs_y = 0
            offs_x += 1

    for it_b, b in enumerate(batch_index):
        logging.info(f"i-th batch {it_b}")
        df = df_full.loc[df_full.random_index.eq(b)]
        x_min, x_max = df.X.min(), df.X.max()
        y_min, y_max = df.Y.min(), df.Y.max()
        print(x_min, x_max, y_min, y_max)
        grid_indx = (grid_info.x >= x_min) &\
                    (grid_info.x <= x_max) &\
                    (grid_info.y >= y_min) &\
                    (grid_info.y <= y_max)
        grid_pt = np.array(grid_info.loc[grid_indx, ["x","y"]])
        if grid_pt.shape[0] < 10:
            continue

        theta = np.array(grid_info.loc[grid_indx, factor_header ])
        theta = sklearn.preprocessing.normalize(np.clip(theta, .05, .95), norm='l1', axis=1)
        n = theta.shape[0]

        indx, dist = bt.query_radius(X = grid_pt, r = args.neighbor_radius, return_distance = True)
        r_indx = [i for i,x in enumerate(indx) for y in range(len(x))]
        c_indx = [x for y in indx for x in y]
        psi_org = np.array([x for y in dist for x in y])
        psi_org = 1-(psi_org / args.neighbor_radius)**nu
        psi_org = coo_array((psi_org, (r_indx,c_indx)),shape=(n, brc.shape[0])).tocsc().T
        psi_org.eliminate_zeros()
        b_indx = np.arange(brc.shape[0])[(psi_org != 0).sum(axis = 1) > 0]
        N = len(b_indx)
        mtx = dge_mtx[b_indx, :]
        psi_org = psi_org[b_indx, :]
        psi_org = sklearn.preprocessing.normalize(psi_org, norm='l1', axis=1)
        logodds = copy.copy(psi_org)
        logodds.data = np.clip(logodds.data, .05, .95)
        logodds.data = np.log(logodds.data/(1-logodds.data))

        phi_org = psi_org @ theta
        phi_org = sklearn.preprocessing.normalize(phi_org, norm='l1', axis=1)

        batch = scorpus.corpus()
        batch.init_from_matrix(mtx, grid_pt, logodds, psi = psi_org,
                               phi = phi_org, m_gamma = theta, features = gene_kept)
        scores = slda.update_lambda(batch)

        tmp = pd.concat([brc.loc[b_indx, ['j','X','Y']].reset_index(),\
                        pd.DataFrame(batch.phi, columns =\
                        ['Topic_'+str(x) for x in range(slda._K)])], axis = 1)
        print(n, N, tmp.shape[0])
        tmp["topK_org"] = np.argmax(phi_org, axis = 1)
        tmp["topP_org"] = np.max(phi_org, axis = 1)
        tmp = tmp[(tmp.X > x_min+out_buff) & (tmp.X < x_max-out_buff)]
        tmp = tmp[(tmp.Y > y_min+out_buff) & (tmp.Y < y_max-out_buff)]
        if n_batch == 0:
            tmp.to_csv(args.output+".pixel.tsv.gz", sep='\t', index=False, header=True, mode='w')
        else:
            tmp.to_csv(args.output+".pixel.tsv.gz", sep='\t', index=False, header=False, mode='a')

        expElog_theta = np.exp(utilt.dirichlet_expectation(batch.gamma))
        expElog_theta/= expElog_theta.sum(axis = 1).reshape((-1, 1))
        tmp = pd.DataFrame({'X':grid_pt[:,0],'Y':grid_pt[:,1]})
        tmp['avg_size'] = np.array(batch.psi.sum(axis = 0)).reshape(-1)
        for v in range(slda._K):
            tmp['Topic_'+str(v)] = expElog_theta[:, v]
        tmp["topK_org"] = theta.argmax(axis = 1)
        tmp["topP_org"] = theta.max(axis = 1)
        tmp = tmp[(tmp.X > x_min+out_buff) & (tmp.X < x_max-out_buff)]
        tmp = tmp[(tmp.Y > y_min+out_buff) & (tmp.Y < y_max-out_buff)]
        if n_batch == 0:
            tmp.to_csv(args.output+".anchor.tsv.gz", sep='\t', index=False, header=True, mode='w')
        else:
            tmp.to_csv(args.output+".anchor.tsv.gz", sep='\t', index=False, header=False, mode='a')

        n_batch += 1

    if args.debug:
        break
    df_full = copy.copy(left.loc[:, input_header])
