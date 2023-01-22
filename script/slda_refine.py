import sys, os, argparse, logging, gzip, csv, copy, re, time, importlib, warnings, pickle
import subprocess as sp
import numpy as np
import pandas as pd

from scipy.sparse import *
import sklearn.neighbors
import sklearn.preprocessing

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
precision = args.precision
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

### Input pixel info (input has to contain certain columns with correct header)
with gzip.open(args.input, 'rt') as rf:
    oheader = rf.readline().strip().split('\t')
input_header = ["random_index","X","Y","gene",key]
dty = {x:int for x in ['X','Y',key]}
dty.update({x:str for x in ['random_index', 'gene']})
mheader = [x for x in input_header if x not in oheader]
if len(mheader) > 0:
    mheader = ", ".join(mheader)
    sys.exit(f"Input misses the following column: {mheader}.")
oheader_indx = [input_header.index(x) for x in input_header]

### Model fitting
betaksum = model.components_.sum(axis = 1)
global_scale = np.min([1, 1000 / np.median(betaksum)])
_lambda = model.components_ * global_scale
if args.debug:
    print(global_scale, np.median(_lambda.sum(axis=1)), sorted(np.around(betaksum*global_scale, 0)) )
slda = OnlineLDA(vocab=gene_kept, K=K, N=args.total_pixel,
                 iter_inner=30, verbose = 1)
slda.init_global_parameter(_lambda)

df_full = pd.DataFrame()
n_batch = 0
chunk_size = 1000000
post_count = np.zeros((K, M))

for chunk in pd.read_csv(args.input, sep='\t', chunksize=chunk_size,\
                        header = 0, usecols=input_header, dtype=dty):
    full_chunk = chunk.shape[0] == chunk_size
    chunk = chunk.loc[(chunk[key] > 0)&chunk.gene.isin(gene_kept), :]
    if chunk.shape[0] == 0:
        continue
    df_full = pd.concat([df_full, chunk])
    left    = pd.DataFrame(columns = input_header)

    # Save the incomplete minibatch for later
    if full_chunk: # This is dangerous should be a better way
        if len(df_full.random_index.unique()) == 1:
            continue
        last_indx = df_full.random_index.iloc[-1]
        left = copy.copy(df_full.loc[df_full.random_index.eq(last_indx), :])
        df_full = df_full.loc[~df_full.random_index.eq(last_indx), :]

    ### Process batches in this chunk of data

    batch_index = list(df_full.random_index.unique() )
    random_pref = df_full.random_index.map(lambda x : x[-5:]).values # In case of overlapping batches in the same data chunk
    if precision <= 0:
        df_full['j'] = random_pref + df_full.X.astype(str) + '_' + df_full.Y.astype(str)
    df_full["X"] = df_full.X * mu_scale
    df_full["Y"] = df_full.Y * mu_scale
    if precision > 0: # Collapse pixels if effectively indistinguishabel
        df_full["X"] = (df_full.X / precision).astype(int)
        df_full["Y"] = (df_full.Y / precision).astype(int)
        df_full = df_full.groupby(by=["random_index","gene","X","Y"]).agg({key:np.sum}).reset_index()
        random_pref = df_full.random_index.map(lambda x : x[-5:]).values
        df_full['j'] = random_pref + '_' + df_full.X.astype(str) + '_' + df_full.Y.astype(str)
        df_full.X = df_full.X * precision
        df_full.Y = df_full.Y * precision

    brc = df_full.groupby(by = ["j"], as_index=False).agg({key:np.sum})
    N0 = brc.shape[0]
    brc.index = range(N0)
    barcode_kept = list(brc['j'])
    brc = brc.merge(right = df_full.loc[:, ["j","X","Y"]].drop_duplicates(subset='j'), on = "j", how = 'left')
    bc_dict = {x:i for i,x in enumerate( barcode_kept ) }

    df_full = df_full.groupby(by = ["random_index", "j", "gene"]).agg({key:sum}).reset_index()
    df_full = df_full.merge(right = brc.loc[:, ["j","X","Y"]], on = 'j', how = 'inner')

    indx_row = [ bc_dict[x] for x in df_full['j']]
    indx_col = [ ft_dict[x] for x in df_full['gene']]
    dge_mtx = coo_array((df_full[key], (indx_row, indx_col)), shape=(N0, M)).tocsr()
    pts = np.asarray(brc[['X','Y']])
    bt = sklearn.neighbors.BallTree(pts)

    logging.info(f"Read {N0} pixels in {len(batch_index)} minibatches ({n_batch} so far). Made DGE {dge_mtx.shape}")

    if not use_input_anchor:
        # Need to initialize anchor points
        n_move = int(np.round(radius * np.sqrt(3) / args.anchor_resolution, 0) )
        offs_x = 0
        offs_y = 0
        grid_info = pd.DataFrame()
        tot_unit = 0
        while offs_x < n_move:
            while offs_y < n_move:
                x,y = pixel_to_hex(np.array(brc[['X','Y']]), \
                    radius, offs_x/n_move, offs_y/n_move)
                hex_pt  = pd.DataFrame({'hex_x':x,'hex_y':y,\
                    'ct':brc[key].values}).groupby(\
                        by=['hex_x','hex_y']).agg({'ct':np.sum}).reset_index()
                hex_pt['x'], hex_pt['y'] = hex_to_pixel(hex_pt.hex_x.values, \
                    hex_pt.hex_y.values, radius, offs_x/n_move, offs_y/n_move)
                hex_pt = hex_pt.loc[hex_pt.ct >= args.min_ct_per_unit, :]
                if hex_pt.shape[0] < 2:
                    offs_y += 1
                    continue
                hex_list = list(zip(hex_pt.hex_x.values, hex_pt.hex_y.values))
                hex_crd  = list(zip(x,y))
                hex_dict = {x:i for i,x in enumerate(hex_list)}
                indx = [i for i,x in enumerate(hex_crd) if x in hex_dict]
                hex_crd = [hex_crd[i] for i in indx]
                sub = pd.DataFrame({'cRow':[hex_dict[x] for x in hex_crd], \
                    'cCol':indx, 'hexID':hex_crd})
                nunit = len(hex_dict)
                tot_unit += nunit
                n_pixel = sub.shape[0]
                mtx = coo_matrix((np.ones(n_pixel, dtype=bool),\
                        (sub.cRow.values, sub.cCol.values)),\
                        shape=(nunit,N0) ).tocsr() @ dge_mtx
                theta = model.transform(mtx)
                lines = pd.DataFrame({'offs_x':offs_x,'offs_y':offs_y, \
                    'hex_x':hex_pt.hex_x.values, 'hex_y':hex_pt.hex_y.values})
                lines['x'], lines['y'] = hex_to_pixel(hex_pt.hex_x.values,\
                    hex_pt.hex_y.values, radius, offs_x/n_move, offs_y/n_move)
                lines = pd.concat((lines, pd.DataFrame(theta, \
                    columns = factor_header)), axis = 1)
                lines['topK'] = np.argmax(theta, axis = 1).astype(int)
                lines['topP'] = np.max(theta, axis = 1)
                grid_info = pd.concat([grid_info, lines])
                offs_y += 1
            offs_y = 0
            offs_x += 1
            logging.info(f"Initializing... {offs_x}/{n_move}... {tot_unit}")

    if args.debug:
        print(grid_info.shape)
    if grid_info.shape[0] == 0:
        continue
    for it_b, b in enumerate(batch_index):

        df = df_full.loc[df_full.random_index.eq(b)]
        x_min, x_max = df.X.min(), df.X.max()
        y_min, y_max = df.Y.min(), df.Y.max()
        grid_indx = (grid_info.x >= x_min) & (grid_info.x <= x_max) &\
                    (grid_info.y >= y_min) & (grid_info.y <= y_max)
        grid_pt = np.array(grid_info.loc[grid_indx, ["x","y"]])
        if grid_pt.shape[0] < 10:
            continue

        theta = np.array(grid_info.loc[grid_indx, factor_header ])
        theta = sklearn.preprocessing.normalize(np.clip(theta, .05, .95), norm='l1', axis=1)
        n = theta.shape[0]

        indx, dist = bt.query_radius(X = grid_pt, r = args.neighbor_radius, return_distance = True)
        r_indx = [i for i,x in enumerate(indx) for y in range(len(x))]
        c_indx = [x for y in indx for x in y]
        wij = np.array([x for y in dist for x in y])
        wij = 1-(wij / args.neighbor_radius)**nu
        wij = coo_array((wij, (r_indx,c_indx)), \
                        shape=(n, brc.shape[0])).tocsc().T
        wij.eliminate_zeros()
        b_indx = np.arange(brc.shape[0])[(wij != 0).sum(axis = 1) > 0]
        N = len(b_indx)
        mtx = dge_mtx[b_indx, :]
        logging.info(f"Fitting batch {n_batch} at ({x_max}, {y_max}) with size ({n}, {N})")

        wij = wij[b_indx, :]
        wij.data = np.clip(wij.data, .05, .95)
        psi_org = copy.copy(wij)
        psi_org = sklearn.preprocessing.normalize(psi_org, norm='l1', axis=1)

        phi_org = psi_org @ theta
        phi_org = sklearn.preprocessing.normalize(phi_org, norm='l1', axis=1)

        batch = scorpus.corpus()
        batch.init_from_matrix(mtx, grid_pt, wij, psi = psi_org, phi = phi_org,\
                               m_gamma = theta, features = gene_kept)
        scores = slda.update_lambda(batch)

        tmp = brc.iloc[b_indx, :]
        v = np.arange(N)[(tmp.X > x_min+out_buff) & (tmp.X < x_max-out_buff) &\
                         (tmp.Y > y_min+out_buff) & (tmp.Y < y_max-out_buff)]
        post_count += batch.phi[v, :].T @ batch.mtx[v, :]

        tmp = copy.copy(brc.loc[b_indx, ['j','X','Y']] )
        tmp.index = range(tmp.shape[0])
        tmp = pd.concat([tmp, pd.DataFrame(batch.phi, columns =\
                        ['Topic_'+str(x) for x in range(slda._K)])], axis = 1)
        tmp["topK_org"] = np.argmax(phi_org, axis = 1)
        tmp["topP_org"] = np.max(phi_org, axis = 1)
        tmp = tmp[(tmp.X > x_min+out_buff) & (tmp.X < x_max-out_buff)]
        tmp = tmp[(tmp.Y > y_min+out_buff) & (tmp.Y < y_max-out_buff)]
        tmp.topP_org = tmp.topP_org.map('{:.6f}'.format)
        tmp.X = tmp.X.map('{:.2f}'.format)
        tmp.Y = tmp.Y.map('{:.2f}'.format)
        if n_batch == 0:
            tmp.to_csv(args.output+".pixel.tsv.gz", sep='\t', index=False, header=True, mode='w', float_format="%.2e", compression={"method":"gzip"})
        else:
            tmp.to_csv(args.output+".pixel.tsv.gz", sep='\t', index=False, header=False, mode='a', float_format="%.2e", compression={"method":"gzip"})

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
        tmp.topP_org = tmp.topP_org.map('{:.6f}'.format)
        tmp.X = tmp.X.map('{:.2f}'.format)
        tmp.Y = tmp.Y.map('{:.2f}'.format)
        if n_batch == 0:
            tmp.to_csv(args.output+".anchor.tsv.gz", sep='\t', index=False, header=True, mode='w', float_format="%.2e", compression={"method":"gzip"})
        else:
            tmp.to_csv(args.output+".anchor.tsv.gz", sep='\t', index=False, header=False, mode='a', float_format="%.2e", compression={"method":"gzip"})

        n_batch += 1

    if args.debug and n_batch > 10:
        break
    df_full = copy.copy(left.loc[:, input_header])

nleft = df_full.shape[0]
logging.info(f"Finished {n_batch} batches ({nleft} lines)")

### Output updated parameters

out_f = args.output + ".updated.model.tsv.gz"
pd.concat([pd.DataFrame({'gene': gene_kept}),\
           pd.DataFrame(slda._lambda.T, dtype='float64',\
            columns = ["Factor_"+str(k) for k in range(K)])],\
        axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.4e', compression={"method":"gzip"})

out_f = args.output + ".posterior.count.tsv.gz"
pd.concat([pd.DataFrame({'gene': gene_kept}),\
           pd.DataFrame(post_count.T, dtype='float64',\
            columns = ["Factor_"+str(k) for k in range(K)])],\
        axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})
