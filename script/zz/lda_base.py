import sys, os, copy, gzip, logging
import pickle, argparse
import numpy as np
import pandas as pd
import random as rng

from scipy.sparse import coo_matrix, coo_array
import sklearn.neighbors
import sklearn.preprocessing
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--feature', type=str, default='', help='')
parser.add_argument('--output_path', type=str, default='', help='')
parser.add_argument('--output_pref', type=str, default='', help='')
parser.add_argument('--identifier', type=str, default='', help='')
parser.add_argument('--hvg', type=str, default='', help='')
parser.add_argument('--nFeature', type=int, default=-1, help='If boath nFeature and hvg are provided and nFeature is larger than the number of genes in hvg, top-expressed genes will be added.')
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate, only used if --x_range and --y_range are used')
parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
parser.add_argument('--log', default = '', type=str, help='files to write log to')
parser.add_argument('--seed', type=int, default=-1, help='')

parser.add_argument('--region', type=str, nargs='*', default=[], help="List of lane:tile1-tile2 or lane:tile to work on")
parser.add_argument('--x_range_um', type=float, nargs='*', default=[], help="Lower and upper bound of the x-axis, in um")
parser.add_argument('--y_range_um', type=float, nargs='*', default=[], help="Lower and upper bound of the y-axis, in um")
parser.add_argument('--x_range', type=float, nargs='*', default=[], help="Lower and upper bound of the x-axis, in original barcode coordinates")
parser.add_argument('--y_range', type=float, nargs='*', default=[], help="Lower and upper bound of the y-axis, in original barcode coordinates")

parser.add_argument('--nFactor', type=int, default=10, help='')
parser.add_argument('--minibatch_size', type=int, default=256, help='')
parser.add_argument('--min_count_per_feature', type=int, default=1, help='')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
parser.add_argument('--verbose', type=int, default=0, help='')
parser.add_argument('--thread', type=int, default=1, help='')
parser.add_argument('--epoch', type=int, default=-1, help='How many times to loop through the full data')
parser.add_argument('--epoch_id_length', type=int, default=-1, help='')
parser.add_argument('--transform_epoch_id', type=str, nargs='*', default=[], help="")
parser.add_argument('--use_model', type=str, default='', help="Use provided model to transform input data")
parser.add_argument('--transform_full', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--skip_transform', action='store_true')

args = parser.parse_args()
if args.log != '':
    try:
        logging.basicConfig(filename=args.log, filemode='a', encoding='utf-8', level=logging.INFO)
    except:
        logging.basicConfig(level= getattr(logging, "INFO", None))
else:
    logging.basicConfig(level= getattr(logging, "INFO", None))

if args.use_model != '' and not os.path.exists(args.use_model):
    sys.exit("Invalid model file")

mu_scale = 1./args.mu_scale
key = args.key.lower()

### Basic parameterse
b_size = args.minibatch_size
K = args.nFactor
factor_header = [str(x) for x in range(K)]
seed = args.seed
if seed <= 0:
    rng.seed()
    seed = rng.randrange(1, 2**31)
output_pref = args.output_pref
if output_pref == '':
    if not os.path.exists(args.output_path + "/analysis"):
        os.makedirs(args.output_path + "/analysis")
    output_pref = args.output_path + "/analysis/"+args.identifier
else:
    if not os.path.exists(os.path.dirname(output_pref)):
        os.makedirs(os.path.dirname(output_pref))


### Input
if not os.path.exists(args.input):
    sys.exit("ERROR: cannot find input file.")
with gzip.open(args.input, 'rt') as rf:
    header = rf.readline().strip().split('\t')
header = [x.lower() for x in header]

# If using only subset of input data
tile_list = []
for v in args.region:
    w = v.split(':')
    if len(w) != 2:
        sys.exit("Invalid regions in --region")
    u = [x for x in w[1].split('-') if x != '']
    if len(u) == 0 or len(u) > 2:
        sys.exit("Invalid regions in --region")
    if len(u) == 2:
        u = [str(x) for x in range(int(u[0]), int(u[1])+1)]
    tile_list += [w[0]+":"+x for x in u]
print(tile_list)

xmin = np.array([-1])
xmax = np.array([np.inf])
ymin = np.array([-1])
ymax = np.array([np.inf])
if len(args.x_range_um) > 0:
    xmin = np.array([x for i,x in enumerate(args.x_range_um) if i % 2 == 0])
    xmax = np.array([x for i,x in enumerate(args.x_range_um) if i % 2 == 1])
elif len(args.x_range) > 0:
    xmin = np.array([x * mu_scale for i,x in enumerate(args.x_range) if i % 2 == 0])
    xmax = np.array([x * mu_scale for i,x in enumerate(args.x_range) if i % 2 == 1])
if len(args.y_range_um) > 0:
    ymin = np.array([x for i,x in enumerate(args.y_range_um) if i % 2 == 0])
    ymax = np.array([x for i,x in enumerate(args.y_range_um) if i % 2 == 1])
elif len(args.y_range) > 0:
    ymin = np.array([x * mu_scale for i,x in enumerate(args.y_range) if i % 2 == 0])
    ymax = np.array([x * mu_scale for i,x in enumerate(args.y_range) if i % 2 == 1])


n_region = np.min([len(xmin), len(xmax), len(ymin), len(ymax)])
if n_region < 1:
    sys.exit("Invalid range parameters")

print(xmin, xmax, args.x_range_um, args.x_range)
print(ymin, ymax, args.y_range_um, args.y_range)

model_f = output_pref+".model.p"
read_model = False
# If use existing model
if os.path.exists(args.use_model):
    model_f = args.use_model
if not args.overwrite and os.path.exists(model_f):
    lda = pickle.load( open( model_f, "rb" ) )
    logging.warning(f"Read existing model from\n{model_f}\n use --overwrite to allow the model files to be overwritten")
    feature_kept = lda.feature_names_in_
    lda.feature_names_in_ = None
    ft_dict = {x:i for i,x in enumerate( feature_kept ) }
    M = len(feature_kept)
    read_model = True
else:
    ### Use only the provided list of features
    if not os.path.exists(args.feature):
        sys.exit("ERROR: cannot find input feature file.")
    feature=pd.read_csv(args.feature, sep='\t', header=0)
    feature.columns = [x.lower() for x in feature.columns]
    feature[key] = feature[key].astype(int)
    feature = feature[feature[key] >= args.min_count_per_feature]
    feature.sort_values(by=key,ascending=False,inplace=True)
    feature.drop_duplicates(subset='gene',keep='first',inplace=True)
    if os.path.exists(args.hvg):
        hvg = pd.read_csv(args.hvg, sep='\t', header=0, usecols=['gene',key],dtype={'gene':str,key:int})
        hvg.sort_values(by=key,ascending=False,inplace=True)
        hvg.drop_duplicates(subset='gene',keep='first',inplace=True)
        nhvg = hvg.shape[0]
        logging.info(f"Read {nhvg} highly variable genes from " + args.hvg)
        if args.nFeature > nhvg:
            feature = feature[~feature.gene.isin(hvg.gene.values)]
            feature.sort_values(by = key, ascending=False, inplace=True)
            feature = pd.concat( (hvg, feature.iloc[:(args.nFactor-hvg.shape[0])] ) )
    feature_mf = np.array(feature[key].values).astype(float)
    feature_mf/= feature_mf.sum()
    feature_kept = list(feature.gene.values)
    ft_dict = {x:i for i,x in enumerate( feature_kept ) }
    M = len(feature_kept)
    logging.info(f"{M} genes will be used")

### Stochastic model fitting
adt = {'random_index':str, '#lane':str, 'tile':str, 'x': str, 'y':str, 'gene':str, key:int}
adthat = {'x':float, 'y':float}
if not read_model:
    logging.info(f"Start fitting model ... model will be stored in\n{model_f}")
    lda = LDA(n_components=K, learning_method='online', batch_size=b_size, n_jobs = args.thread, verbose = 0, random_state=seed)
    epoch_id = set()
    df = pd.DataFrame()
    for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=500000, skiprows=1, names=header, usecols=["random_index","#lane","tile","x","y","gene",key], dtype=adt):
        chunk = chunk[chunk.gene.isin(feature_kept)]
        if args.epoch_id_length > 0:
            v = chunk.random_index.map(lambda x: x[:args.epoch_id_length]).values
            v = set(v)
            epoch_id.update(v)
        chunk['j'] = chunk.random_index.values + '_' + chunk.x.values + '_' + chunk.y.values
        chunk['tile'] = chunk["#lane"].values + ':' + chunk.tile.values
        chunk = chunk.astype(adthat)
        i = 0
        indx = (chunk.x >= xmin[i]) & (chunk.x <= xmax[i]) & (chunk.y >= ymin[i]) & (chunk.y <= ymax[i])
        while i < len(xmin):
            indx = indx | ((chunk.x >= xmin[i]) & (chunk.x <= xmax[i]) & (chunk.y >= ymin[i]) & (chunk.y <= ymax[i]))
            i += 1
        if len(tile_list) > 0:
            indx = indx & chunk.tile.isin(tile_list)
        if sum(indx) == 0:
            continue
        chunk = chunk[indx]
        chunk.drop(columns = ['random_index','x','y'], inplace=True)
        last_indx = chunk.j.iloc[-1]
        df = pd.concat([df, chunk[~chunk.j.eq(last_indx)]])
        if len(df.j.unique()) < b_size * 1.5: # Left to next chunk
            df = pd.concat((df, chunk[chunk.j.eq(last_indx)]))
            continue
        # Total mulecule count per unit
        brc = df.groupby(by = ['j']).agg({key: sum}).reset_index()
        brc = brc[brc[key] > args.min_ct_per_unit]
        brc.index = range(brc.shape[0])
        df = df[df.j.isin(brc.j.values)]
        # Make DGE
        barcode_kept = list(brc.j.values)
        bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
        indx_row = [ bc_dict[x] for x in df['j']]
        indx_col = [ ft_dict[x] for x in df['gene']]
        N = len(barcode_kept)
        mtx = coo_matrix((df[key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
        x1 = np.median(brc[key].values)
        x2 = np.mean(brc[key].values)
        logging.info(f"Made DGE {mtx.shape}, median/mean count: {x1:.1f}/{x2:.1f}")
        _ = lda.partial_fit(mtx)

        # Evaluation Training Performance
        if args.verbose > 0:
            logl = lda.score(mtx) / mtx.shape[0]
            logging.info(f"logl: {logl:.4f}")
        if args.verbose > 1:
            # Compute topic coherence
            topic_pmi = []
            top_gene_n = np.min([50, mtx.shape[1]])
            pseudo_ct = 100
            for k in range(K):
                b = lda.exp_dirichlet_component_[k,:]
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
            logging.info("Coherence: "+", ".join([str(x) for x in topic_pmi]))
        df = copy.copy(chunk[chunk.j.eq(last_indx)] )
        logging.info(f"Leftover size {len(df)}")
        if args.epoch > 0 and len(epoch_id) > args.epoch:
            break

    if len(df.j.unique()) > b_size:
        brc = df.groupby(by = ['j']).agg({key: sum}).reset_index()
        brc = brc[brc[key] > args.min_ct_per_unit]
        brc.index = range(brc.shape[0])
        df = df[df.j.isin(brc.j.values)]
        barcode_kept = list(brc.j.values)
        bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
        indx_row = [ bc_dict[x] for x in df['j']]
        indx_col = [ ft_dict[x] for x in df['gene']]
        N = len(barcode_kept)
        mtx = coo_matrix((df[key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
        x1 = np.median(brc[key].values)
        x2 = np.mean(brc[key].values)
        logging.info(f"Made DGE {mtx.shape}, median/mean count: {x1:.1f}/{x2:.1f}")
        _ = lda.partial_fit(mtx)
        logl = lda.score(mtx) / mtx.shape[0]
        logging.info(f"logl: {logl:.4f}")

    lda.feature_names_in_ = feature_kept
    pickle.dump( lda, open( model_f, "wb" ) )
    out_f = model_f.replace("model.p", "model_matrix.tsv.gz")
    pd.concat([pd.DataFrame({'gene': lda.feature_names_in_}),\
                pd.DataFrame(lda.components_.T,\
                columns = [str(k) for k in range(K)], dtype='float64')],\
                axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.4e', compression={"method":"gzip"})


if args.skip_transform:
    sys.exit()

### Rerun all units once and store results
dtp = {'topK':int, key:int,'j':str, 'x':str, 'y':str}
dtp.update({x:float for x in ['topP']+factor_header})
nbatch = 0
res_f = output_pref+".fit_result.tsv.gz"
logging.info(f"Result file {res_f}")

post_count = np.zeros((K, M))
epoch_id = ''
df = pd.DataFrame()
for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=500000, skiprows=1, names=header, usecols=["random_index","#lane","tile","x","y","gene",key], dtype=adt):
    chunk = chunk[chunk.gene.isin(feature_kept)]
    end_of_epoch = False
    if not args.transform_full and args.epoch_id_length > 0:
        v = chunk.random_index.map(lambda x: x[:args.epoch_id_length]).values
        if len(args.transform_epoch_id) > 0:
            indx = [i for i,x in enumerate(v) if x in args.transform_epoch_id]
            chunk = chunk.iloc[indx, :]
        else:
            if epoch_id == '':
                epoch_id = v[0]
            if v[-1] != v[0]:
                end_of_epoch = True
                chunk = chunk.loc[v == epoch_id, :]
        if chunk.shape[0] == 0:
            continue
    chunk['j'] = chunk.random_index.values + '_' + chunk.x.values + '_' + chunk.y.values
    chunk['tile'] = chunk["#lane"].values + ':' + chunk.tile.values
    chunk = chunk.astype(adthat)
    i = 0
    indx = (chunk.x >= xmin[i]) & (chunk.x <= xmax[i]) & (chunk.y >= ymin[i]) & (chunk.y <= ymax[i])
    while i < len(xmin):
        indx = indx | ((chunk.x >= xmin[i]) & (chunk.x <= xmax[i]) & (chunk.y >= ymin[i]) & (chunk.y <= ymax[i]))
        i += 1
    if len(tile_list) > 0:
        indx = indx & chunk.tile.isin(tile_list)
    if sum(indx) == 0:
        continue
    chunk = chunk[indx]
    chunk.drop(columns = ['random_index','x','y'], inplace=True)
    last_indx = chunk.j.iloc[-1]
    left = copy.copy(chunk[chunk.j.eq(last_indx)])
    df = pd.concat([df, chunk[~chunk.j.eq(last_indx)]])
    if len(df.j.unique()) < b_size * 1.5: # Left to next chunk
        df = pd.concat((df, chunk[chunk.j.eq(last_indx)]))
        continue
    # Total mulecule count per unit
    brc = df.groupby(by = ['j']).agg({key: sum}).reset_index()
    brc = brc[brc[key] > args.min_ct_per_unit]
    brc.index = range(brc.shape[0])
    df = df[df.j.isin(brc.j.values)]
    # Make DGE
    barcode_kept = list(brc.j.values)
    bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
    indx_row = [ bc_dict[x] for x in df['j']]
    indx_col = [ ft_dict[x] for x in df['gene']]
    N = len(barcode_kept)
    mtx = coo_array((df[key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
    x1 = np.median(brc[key].values)
    x2 = np.mean(brc[key].values)
    logging.info(f"Made DGE {mtx.shape}, median/mean count: {x1:.1f}/{x2:.1f}")

    theta = lda.transform(mtx)
    post_count += np.array(theta.T @ mtx)
    brc['x'] = brc.j.map(lambda x : x.split('_')[1])
    brc['y'] = brc.j.map(lambda x : x.split('_')[2])
    brc['j'] = brc.j.map(lambda x : x.split('_')[0])
    brc = pd.concat((brc, pd.DataFrame(theta, columns = factor_header)), axis = 1)
    brc['topK'] = np.argmax(theta, axis = 1).astype(int)
    brc['topP'] = np.max(theta, axis = 1)
    brc = brc.astype(dtp)
    print(brc.shape[0])
    if nbatch == 0:
        brc.to_csv(res_f, sep='\t', mode='w', float_format="%.5f", index=False, header=True, compression={"method":"gzip"})
    else:
        brc.to_csv(res_f, sep='\t', mode='a', float_format="%.5f", index=False, header=False, compression={"method":"gzip"})
    nbatch += 1
    df = copy.copy(left)
    if end_of_epoch:
        break

# Leftover
brc = df.groupby(by = ['j']).agg({key: sum}).reset_index()
brc = brc[brc[key] > args.min_ct_per_unit]
brc.index = range(brc.shape[0])
print(brc.shape[0])
if brc.shape[0] > 0:
    df = df[df.j.isin(brc.j.values)]
    # Make DGE
    barcode_kept = list(brc.j.values)
    bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
    indx_row = [ bc_dict[x] for x in df['j']]
    indx_col = [ ft_dict[x] for x in df['gene']]
    N = len(barcode_kept)
    mtx = coo_array((df[key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
    x1 = np.median(brc[key].values)
    x2 = np.mean(brc[key].values)
    logging.info(f"Made DGE {mtx.shape}, median/mean count: {x1:.1f}/{x2:.1f}")

    theta = lda.transform(mtx)
    post_count += np.array(theta.T @ mtx)
    brc['x'] = brc.j.map(lambda x : x.split('_')[1])
    brc['y'] = brc.j.map(lambda x : x.split('_')[2])
    brc['j'] = brc.j.map(lambda x : x.split('_')[0])
    brc = pd.concat((brc, pd.DataFrame(theta, columns = factor_header)), axis = 1)
    brc['topK'] = np.argmax(theta, axis = 1).astype(int)
    brc['topP'] = np.max(theta, axis = 1)
    brc = brc.astype(dtp)
    print(brc.shape)
    if nbatch == 0:
        brc.to_csv(res_f, sep='\t', mode='w', float_format="%.5f", index=False, header=True, compression={"method":"gzip"})
    else:
        brc.to_csv(res_f, sep='\t', mode='a', float_format="%.5f", index=False, header=False, compression={"method":"gzip"})

logging.info(f"Finished ({nbatch})")

out_f = output_pref+".posterior.count.tsv.gz"
pd.concat([pd.DataFrame({'gene': feature_kept}),\
           pd.DataFrame(post_count.T, dtype='float64',\
                        columns = [str(k) for k in range(K)])],\
        axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})
