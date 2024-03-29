import sys, os, argparse, logging, gzip, csv, copy, re, time, importlib, warnings, pickle
import numpy as np
import pandas as pd

from scipy.sparse import coo_array
import sklearn.neighbors
import sklearn.preprocessing

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from online_slda import OnlineLDA
from slda_minibatch import minibatch
import utilt

parser = argparse.ArgumentParser()

# Innput and output info
parser.add_argument('--input', type=str, help='')
parser.add_argument('--model', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--anchor', type=str, help='')
parser.add_argument('--anchor_in_um', action='store_true')

# Data realted parameters
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--key', type=str, default = 'gn', help='gt: genetotal, gn: gene, spl: velo-spliced')
parser.add_argument('--precision', type=float, default=.25, help='If positive, collapse pixels within X um.')

# Learning related parameters
parser.add_argument('--decode_only', action='store_true')
parser.add_argument('--neighbor_radius', type=float, default=25, help='The radius (um) of each anchor point\'s territory')
parser.add_argument('--halflife', type=float, default=0.7, help='Control the decay of distance-based weight')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='Keep anchor points with at least x reads inside its territory during initialization. Would only be used if --anchor file is not provided')
parser.add_argument('--theta_init_bound_multiplier', type=float, default=.2, help='')
parser.add_argument('--total_pixel', type=float, default=1e5, help='(An estimate of) total number of pixels just for calculating the learning rate')
parser.add_argument('--lambda_init_scale', type=float, default=1e5, help='')
parser.add_argument('--kappa', type=float, default=.7, help='')
parser.add_argument('--tau0', type=int, default=10, help='')
parser.add_argument('--inner_max_iter', type=int, default=30, help='')
parser.add_argument('--epoch', type=int, default=1, help='')
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
if not os.path.exists(args.anchor):
    sys.exit("ERROR: cannot find anchor file")

### Basic parameterse
mu_scale = 1./args.mu_scale
precision = args.precision
key = args.key.lower()
nu = np.log(.5) / np.log(args.halflife)
out_buff = args.neighbor_radius * args.halflife
unspl = 'unspl'

### Load model
model = pickle.load(open( args.model, "rb" ))
gene_kept = model.feature_names_in_
model.feature_names_in_ = None
ft_dict = {x:i for i,x in enumerate( gene_kept ) }
K, M = model.components_.shape
factor_header = [str(x) for x in range(K)]
init_bound = 1./K * args.theta_init_bound_multiplier
if '_unspl' not in ft_dict:
    sys.exit("ERROR: this experimental script is specific to the case where unspliced read count were present in the model as an aggregated feature")
logging.info(f"{M} genes and {K} factors are read from input model")

### Read provided anchor info
grid_info = pd.DataFrame()
grid_info = pd.read_csv(args.anchor,sep='\t')
header_map = {"X":"x", "Y":"y"}
for x in grid_info.columns:
    # Dangerous way to detect which columns to use as factor loadings
    y = re.match('^[A-Za-z]*_*(\d+)$', x)
    if y:
        header_map[y.group(0)] = y.group(1)
grid_info.rename(columns = header_map, inplace=True)
if not args.anchor_in_um:
    grid_info.x *= mu_scale
    grid_info.y *= mu_scale
ref = sklearn.neighbors.BallTree(np.array(grid_info.loc[:, ['x','y']]))
logging.info(f"Read {grid_info.shape[0]} grid points")

### Input pixel info (input has to contain certain columns with correct header)
with gzip.open(args.input, 'rt') as rf:
    oheader = rf.readline().strip().split('\t')
oheader = [x.lower() if len(x) > 1 else x for x in oheader]
input_header = ["random_index","X","Y","gene",key,'unspl']
unit_header = ["random_index","X","Y"]
clps_header = unit_header + ['gene', key]
dty = {x:int for x in ['X','Y',key,'unspl']}
dty.update({x:str for x in ['random_index', 'gene']})
mheader = [x for x in input_header if x not in oheader]
if len(mheader) > 0:
    mheader = ", ".join(mheader)
    sys.exit(f"Input misses the following column: {mheader}.")


### Model fitting
_lambda = model.components_
_tau0 = args.tau0
if not args.decode_only:
    if args.lambda_init_scale > 1:
        _lambda *= args.lambda_init_scale / model.components_.sum()
    if args.tau0 < 0:
        _tau0 = model.n_batch_iter_
slda = OnlineLDA(vocab=gene_kept, K=K, N=args.total_pixel, kappa=args.kappa,
                 tau0 = _tau0, iter_inner=args.inner_max_iter, verbose = 1)
slda.init_global_parameter(_lambda)


chunk_size = 1000000
post_count = np.zeros((K, M))
if args.decode_only:
    args.epoch = 1
for it_epoch in range(args.epoch):
    df_full = pd.DataFrame()
    n_batch = 0
    reader = pd.read_csv(args.input, sep='\t', chunksize=chunk_size,\
                         skiprows=1, names=oheader, usecols=input_header, dtype=dty)
    end_of_file = False
    while True:
        try:
            chunk = next(reader)
        except StopIteration:
            end_of_file = True
        if end_of_file and df_full.shape[0] == 0:
            break

        left = pd.DataFrame(columns = clps_header)
        if not end_of_file:
            full_chunk = len(chunk) == chunk_size
            # Collapse unspliced reads
            clps = pd.DataFrame()
            sub = chunk.groupby(by='random_index',as_index=False).agg({unspl:sum})
            sub = sub.loc[sub[unspl] > 0]
            if sub.shape[0] > 0:
                sub.rename(columns={unspl:key},inplace=True)
                sub['gene'] = '_'+unspl
                clps = pd.concat([clps, sub])
            clps = clps.merge(right = chunk[unit_header].drop_duplicates(subset='random_index'), on='random_index', how='left')
            chunk = chunk.loc[(chunk[key] > 0) & chunk.gene.isin(gene_kept), unit_header + ['gene', key]]
            chunk = pd.concat([clps, chunk])

            df_full = pd.concat([df_full, chunk])
            if df_full.shape[0] == 0:
                continue

            # Save the incomplete minibatch for later
            if full_chunk: # This is dangerous should be a better way
                if len(df_full.random_index.unique()) == 1:
                    continue
                last_indx = df_full.random_index.iloc[-1]
                left = copy.copy(df_full.loc[df_full.random_index.eq(last_indx), :])
                df_full = df_full.loc[~df_full.random_index.eq(last_indx), :]

        ### Process batches in this chunk of data
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
        pts = df_full[["j", "X", "Y"]].drop_duplicates(subset="j")

        dist, indx = ref.query(X = np.array(pts[['X','Y']]), k = 1, return_distance = True)
        dist = dist.squeeze()
        kept_pixel = dist < args.neighbor_radius
        N_org = pts.shape[0]
        N_keep = np.sum(kept_pixel)
        logging.info(f"{N_keep} out of {N_org} pixels are close enough to any anchor.")
        if N_keep == 0:
            df_full = copy.copy(left)
            continue
        pts = pts.loc[kept_pixel, :]
        df_full = df_full.loc[df_full.j.isin(pts.j.values), :]

        batch_index = list(df_full.random_index.unique() )
        brc = df_full.groupby(by = ["j"], as_index=False).agg({key:np.sum})
        N0 = brc.shape[0]
        brc.index = range(N0)
        barcode_kept = list(brc['j'])
        brc = brc.merge(right = pts, on = "j", how = 'left')
        pts = np.asarray(brc[['X','Y']])
        bt = sklearn.neighbors.BallTree(pts)
        bc_dict = {x:i for i,x in enumerate( barcode_kept ) }

        df_full = df_full.groupby(by = ["random_index", "j", "gene"]).agg({key:sum}).reset_index()
        df_full = df_full.merge(right = brc.loc[:, ["j","X","Y"]], on = 'j', how = 'inner')

        indx_row = [ bc_dict[x] for x in df_full['j']]
        indx_col = [ ft_dict[x] for x in df_full['gene']]
        dge_mtx = coo_array((df_full[key], (indx_row, indx_col)), shape=(N0, M)).tocsr()

        logging.info(f"Read {N0} pixels in {len(batch_index)} minibatches ({n_batch} so far). Made DGE {dge_mtx.shape}")

        for it_b, b in enumerate(batch_index):

            df = df_full.loc[df_full.random_index.eq(b)]
            x_min, x_max = df.X.min(), df.X.max()
            y_min, y_max = df.Y.min(), df.Y.max()
            grid_indx = (grid_info.x >= x_min - args.neighbor_radius) &\
                        (grid_info.x <= x_max + args.neighbor_radius) &\
                        (grid_info.y >= y_min - args.neighbor_radius) &\
                        (grid_info.y <= y_max + args.neighbor_radius)
            grid_pt = np.array(grid_info.loc[grid_indx, ["x","y"]])
            if grid_pt.shape[0] < 10:
                continue

            theta = np.array(grid_info.loc[grid_indx, factor_header ])
            theta = sklearn.preprocessing.normalize(np.clip(theta, init_bound, 1.-init_bound), norm='l1', axis=1)
            n = theta.shape[0]

            indx, dist = bt.query_radius(X = grid_pt, r = args.neighbor_radius, return_distance = True)
            r_indx = [i for i,x in enumerate(indx) for y in range(len(x))]
            c_indx = [x for y in indx for x in y]
            wij = np.array([x for y in dist for x in y])
            wij = 1-(wij / args.neighbor_radius)**nu
            wij = coo_array((wij, (r_indx,c_indx)), \
                            shape=(n, brc.shape[0])).tocsc().T
            wij.eliminate_zeros()
            nchoice=(wij != 0).sum(axis = 1)
            mednchoice=np.median(nchoice)
            mednchoicennz=np.median(nchoice[nchoice > 0] )
            b_indx = np.arange(brc.shape[0])[nchoice > 0]
            N = len(b_indx)
            mtx = dge_mtx[b_indx, :]
            logging.info(f"Fitting batch {n_batch} at ({x_max}, {y_max}) with size ({n}, {N}). Median nChoice {mednchoice} ({mednchoicennz})")

            wij = wij[b_indx, :]
            wij.data = np.clip(wij.data, .05, .95)
            psi_org = copy.copy(wij)
            psi_org = sklearn.preprocessing.normalize(psi_org, norm='l1', axis=1)

            batch = minibatch()
            batch.init_from_matrix(mtx, grid_pt, wij, psi = psi_org, phi = None, m_gamma = theta, features = gene_kept)
            if args.decode_only or (it_epoch > 0 and it_epoch == args.epoch-1):
                sstats = slda.do_e_step(batch)
            else:
                scores = slda.update_lambda(batch)

            if it_epoch == args.epoch - 1:

                tmp = brc.iloc[b_indx, :]
                v = np.arange(N)[(tmp.X > x_min+out_buff) & (tmp.X < x_max-out_buff) &\
                                (tmp.Y > y_min+out_buff) & (tmp.Y < y_max-out_buff)]
                post_count += batch.phi[v, :].T @ batch.mtx[v, :]

                tmp = copy.copy(brc.loc[b_indx, ['j','X','Y']] )
                tmp.index = range(tmp.shape[0])
                tmp = pd.concat([tmp, pd.DataFrame(batch.phi, columns =\
                                ['Topic_'+str(x) for x in range(slda._K)])], axis = 1)
                tmp = tmp[(tmp.X > x_min+out_buff) & (tmp.X < x_max-out_buff)]
                tmp = tmp[(tmp.Y > y_min+out_buff) & (tmp.Y < y_max-out_buff)]
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
                tmp = tmp[(tmp.X > x_min+out_buff) & (tmp.X < x_max-out_buff)]
                tmp = tmp[(tmp.Y > y_min+out_buff) & (tmp.Y < y_max-out_buff)]
                tmp.X = tmp.X.map('{:.2f}'.format)
                tmp.Y = tmp.Y.map('{:.2f}'.format)
                if n_batch == 0:
                    tmp.to_csv(args.output+".anchor.tsv.gz", sep='\t', index=False, header=True, mode='w', float_format="%.2e", compression={"method":"gzip"})
                else:
                    tmp.to_csv(args.output+".anchor.tsv.gz", sep='\t', index=False, header=False, mode='a', float_format="%.2e", compression={"method":"gzip"})

            n_batch += 1

        if args.debug and n_batch > 10:
            break
        df_full = copy.copy(left)
        logging.info(f"Left over size {df_full.shape[0]}")

    nleft = df_full.shape[0]
    it_epoch += 1
    logging.info(f"Finished {it_epoch} cover of the data. {n_batch} batches ({nleft} lines)")

### Output posterior summaries

out_f = args.output + ".posterior.count.tsv.gz"
pd.concat([pd.DataFrame({'gene': gene_kept}),\
           pd.DataFrame(post_count.T, dtype='float64',\
            columns = ["Factor_"+str(k) for k in range(K)])],\
        axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})

if not args.decode_only:
    out_f = args.output + ".updated.model.tsv.gz"
    pd.concat([pd.DataFrame({'gene': gene_kept}),\
            pd.DataFrame(slda._lambda.T, dtype='float64',\
                columns = ["Factor_"+str(k) for k in range(K)])],\
            axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.4e', compression={"method":"gzip"})
