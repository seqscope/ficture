import sys, os, copy, gzip, logging
import pickle, argparse
import numpy as np
import pandas as pd

from scipy.sparse import *
import sklearn.neighbors
import sklearn.preprocessing
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--feature', type=str, help='')
parser.add_argument('--output_path', type=str, help='')
parser.add_argument('--identifier', type=str, help='')
parser.add_argument('--hvg', type=str, default='', help='')
parser.add_argument('--nFeature', type=int, default=-1, help='If boath nFeature and hvg are provided and nFeature is larger than the number of genes in hvg, top-expressed genes will be added.')
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate, only used if --x_range and --y_range are used')
parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
parser.add_argument('--log', default = '', type=str, help='files to write log to')

parser.add_argument('--region', type=str, nargs='*', default=[], help="List of lane:tile1-tile2 or lane:tile to work on")
parser.add_argument('--x_range_um', type=float, nargs='*', default=[], help="Lower and upper bound of the x-axis, in um")
parser.add_argument('--y_range_um', type=float, nargs='*', default=[], help="Lower and upper bound of the y-axis, in um")
parser.add_argument('--x_range', type=float, nargs='*', default=[], help="Lower and upper bound of the x-axis, in original barcode coordinates")
parser.add_argument('--y_range', type=float, nargs='*', default=[], help="Lower and upper bound of the y-axis, in original barcode coordinates")

parser.add_argument('--nFactor', type=int, default=10, help='')
parser.add_argument('--minibatch_size', type=int, default=256, help='')
parser.add_argument('--min_count_per_feature', type=int, default=1, help='')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
parser.add_argument('--thread', type=int, default=1, help='')
parser.add_argument('--epoch', type=int, default=1, help='How many times to loop through the full data')
parser.add_argument('--use_model', type=str, default='', help="Use provided model to transform input data")
parser.add_argument('--overwrite', action='store_true')

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
factor_header = ['Topic_'+str(x) for x in range(K)]

### Input
if not os.path.exists(args.input) or not os.path.exists(args.feature):
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


### Use only the provided list of features
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

feature_kept = list(feature.gene.values)
ft_dict = {x:i for i,x in enumerate( feature_kept ) }
M = len(feature_kept)
logging.info(f"{M} genes will be used")



### Stochastic model fitting
model_f = args.output_path + "/analysis/"+args.identifier+".model.p"
adt = {'random_index':str, '#lane':str, 'tile':str, 'x': str, 'y':str, 'gene':str, key:int}
adthat = {'x':float, 'y':float}
if os.path.exists(args.use_model):
    model_f = args.use_model
if not args.overwrite and os.path.exists(model_f):
    lda = pickle.load( open( model_f, "rb" ) )
    logging.warning(f"Read existing model from\n{model_f}\n use --overwrite to allow the model files to be overwritten")
    feature_kept = lda.feature_names_in_
    lda.feature_names_in_ = None
    ft_dict = {x:i for i,x in enumerate( feature_kept ) }
    M = len(feature_kept)
else:
    logging.info(f"Start fitting model ... model will be stored in\n{model_f}")
    lda = LDA(n_components=K, learning_method='online', batch_size=b_size, n_jobs = args.thread, verbose = 0)
    feature_mf = np.array(feature[key].values).astype(float)
    feature_mf/= feature_mf.sum()
    epoch = 0
    df = pd.DataFrame()
    while epoch < args.epoch:
        df = pd.DataFrame()
        for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=500000, skiprows=1, names=header, usecols=["random_index","#lane","tile","x","y","gene",key], dtype=adt):
            chunk = chunk[chunk.gene.isin(feature_kept)]
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
            logl = lda.score(mtx) / mtx.shape[0]
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
            df = copy.copy(chunk[chunk.j.eq(last_indx)] )
            logging.info(f"logl: {logl:.4f}")
            logging.info("Coherence: "+", ".join([str(x) for x in topic_pmi]))
        epoch += 1

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
                pd.DataFrame(sklearn.preprocessing.normalize(lda.components_, axis = 1, norm='l1').T,\
                columns = ["Factor_"+str(k) for k in range(K)], dtype='float64')],\
                axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.4e', compression={"method":"gzip"})




### Rerun all units once and store results
dtp = {'topK':int,key:int,'j':str, 'x':str, 'y':str}
dtp.update({x:float for x in ['topP']+factor_header})
res_f = args.output_path+"/analysis/"+args.identifier+".fit_result.tsv.gz"
nbatch = 0
logging.info(f"Result file {res_f}")

post_count = np.zeros((K, M))
df = pd.DataFrame()
for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=500000, skiprows=1, names=header, usecols=["random_index","#lane","tile","x","y","gene",key], dtype=adt):
    chunk = chunk[chunk.gene.isin(feature_kept)]
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
    print(brc.shape)
    if nbatch == 0:
        brc.to_csv(res_f, sep='\t', mode='w', float_format="%.5f", index=False, header=True, compression={"method":"gzip"})
    else:
        brc.to_csv(res_f, sep='\t', mode='a', float_format="%.5f", index=False, header=False, compression={"method":"gzip"})
    nbatch += 1
    df = copy.copy(left)

# Leftover
brc = df.groupby(by = ['j']).agg({key: sum}).reset_index()
brc = brc[brc[key] > args.min_ct_per_unit]
brc.index = range(brc.shape[0])
print(brc.shape)
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

out_f = args.output_path+"/analysis/"+args.identifier+".posterior.count.tsv.gz"
pd.concat([pd.DataFrame({'gene': feature_kept}),\
           pd.DataFrame(post_count.T, dtype='float64',\
                        columns = [str(k) for k in range(K)])],\
        axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})
