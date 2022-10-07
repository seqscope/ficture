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
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
parser.add_argument('--log', default = '', type=str, help='files to write log to')

parser.add_argument('--nFactor', type=int, default=10, help='')
parser.add_argument('--minibatch_size', type=int, default=256, help='')
parser.add_argument('--min_count_per_feature', type=int, default=1, help='')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
parser.add_argument('--thread', type=int, default=1, help='')
parser.add_argument('--epoch', type=int, default=1, help='How many times to loop through the full data')
parser.add_argument('--overwrite', action='store_true')

args = parser.parse_args()
if args.log != '':
    try:
        logging.basicConfig(filename=args.log, filemode='a', encoding='utf-8', level=logging.INFO)
    except:
        logging.basicConfig(level= getattr(logging, "INFO", None))
else:
    logging.basicConfig(level= getattr(logging, "INFO", None))


mu_scale = 1./args.mu_scale
key = args.key

### Basic parameterse
b_size = args.minibatch_size
K = args.nFactor
factor_header = ['Topic_'+str(x) for x in range(K)]

### Input and output
if not os.path.exists(args.input) or not os.path.exists(args.feature):
    sys.exit("ERROR: cannot find input file.")

### Use only the provided list of features
feature = pd.read_csv(args.feature, sep='\t', header=0, usecols=['gene',args.key],dtype={'gene':str,args.key:int})
feature = feature[feature[args.key] >= args.min_count_per_feature]
feature.sort_values(by=args.key,ascending=False,inplace=True)
feature.drop_duplicates(subset='gene',keep='first',inplace=True)
if os.path.exists(args.hvg):
    hvg = pd.read_csv(args.hvg, sep='\t', header=0, usecols=['gene',args.key],dtype={'gene':str,args.key:int})
    hvg.sort_values(by=args.key,ascending=False,inplace=True)
    hvg.drop_duplicates(subset='gene',keep='first',inplace=True)
    nhvg = hvg.shape[0]
    logging.info(f"Read {nhvg} highly variable genes from " + args.hvg)
    if args.nFeature > nhvg:
        feature = feature[~feature.gene.isin(hvg.gene.values)]
        feature.sort_values(by = args.key, ascending=False, inplace=True)
        feature = pd.concat( (hvg, feature.iloc[:(args.nFactor-hvg.shape[0])] ) )

feature_kept = list(feature.gene.values)
ft_dict = {x:i for i,x in enumerate( feature_kept ) }
M = len(feature_kept)
logging.info(f"{M} genes will be used")



### Stochastic model fitting
model_f = args.output_path + "/analysis/"+args.identifier+".model.p"
logging.info(f"Output file {model_f}")
adt = {'random_index':str, 'X': str, 'Y':str, 'gene':str, key:int}
if not args.overwrite and os.path.exists(model_f):
    logging.warning(f"Model already exits, use --overwrite to allow the existing model files to be overwritten\n{model_f}")
    lda = pickle.load( open( model_f, "rb" ) )
    feature_kept = lda.feature_names_in_
    lda.feature_names_in_ = None
    ft_dict = {x:i for i,x in enumerate( feature_kept ) }
    M = len(feature_kept)
else:
    lda = LDA(n_components=K, learning_method='online', batch_size=b_size, n_jobs = args.thread, verbose = 0)
    feature_mf = np.array(feature[args.key].values).astype(float)
    feature_mf/= feature_mf.sum()
    epoch = 0
    while epoch < args.epoch:
        df = pd.DataFrame()
        for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=500000, header=0, usecols=["random_index","X","Y","gene",key], dtype=adt):
            chunk = chunk[chunk.gene.isin(feature_kept)]
            if chunk.shape[0] == 0:
                continue
            chunk['j'] = chunk.random_index.values + '_' + chunk.X.values + '_' + chunk.Y.values
            chunk.drop(columns = ['random_index','X','Y'], inplace=True)
            last_indx = chunk.j.iloc[-1]
            df = pd.concat([df, chunk[~chunk.j.eq(last_indx)]])
            if len(df.j.unique()) < b_size * 1.5: # Left to next chunk
                df = pd.concat((df, chunk[chunk.j.eq(last_indx)]))
                continue
            # Total mulecule count per unit
            brc = df.groupby(by = ['j']).agg({args.key: sum}).reset_index()
            brc = brc[brc[args.key] > args.min_ct_per_unit]
            brc.index = range(brc.shape[0])
            df = df[df.j.isin(brc.j.values)]
            # Make DGE
            barcode_kept = list(brc.j.values)
            bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
            indx_row = [ bc_dict[x] for x in df['j']]
            indx_col = [ ft_dict[x] for x in df['gene']]
            N = len(barcode_kept)
            mtx = coo_matrix((df[args.key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
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
        brc = df.groupby(by = ['j']).agg({args.key: sum}).reset_index()
        brc = brc[brc[args.key] > args.min_ct_per_unit]
        brc.index = range(brc.shape[0])
        df = df[df.j.isin(brc.j.values)]
        barcode_kept = list(brc.j.values)
        bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
        indx_row = [ bc_dict[x] for x in df['j']]
        indx_col = [ ft_dict[x] for x in df['gene']]
        N = len(barcode_kept)
        mtx = coo_matrix((df[args.key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
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
                axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.4e')




### Rerun all units once and store results
dtp = {'topK':int,args.key:int,'j':str, 'x':str, 'y':str}
dtp.update({x:float for x in ['topP']+factor_header})
res_f = args.output_path+"/analysis/"+args.identifier+".fit_result.tsv.gz"
nbatch = 0

df = pd.DataFrame()
for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=100000, header=0, usecols=["random_index","X","Y","gene",key], dtype=adt):
    chunk = chunk[chunk.gene.isin(feature_kept)]
    if chunk.shape[0] == 0:
        continue
    chunk['j'] = chunk.random_index.values + '_' + chunk.X.values + '_' + chunk.Y.values
    chunk.drop(columns = ['random_index','X','Y'], inplace=True)
    last_indx = chunk.j.iloc[-1]
    left = copy.copy(chunk[chunk.j.eq(last_indx)])
    df = pd.concat([df, chunk[~chunk.j.eq(last_indx)]])
    if len(df.j.unique()) < b_size * 1.5: # Left to next chunk
        df = pd.concat((df, chunk[chunk.j.eq(last_indx)]))
        continue
    # Total mulecule count per unit
    brc = df.groupby(by = ['j']).agg({args.key: sum}).reset_index()
    brc = brc[brc[args.key] > args.min_ct_per_unit]
    brc.index = range(brc.shape[0])
    df = df[df.j.isin(brc.j.values)]
    # Make DGE
    barcode_kept = list(brc.j.values)
    bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
    indx_row = [ bc_dict[x] for x in df['j']]
    indx_col = [ ft_dict[x] for x in df['gene']]
    N = len(barcode_kept)
    mtx = coo_matrix((df[args.key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
    x1 = np.median(brc[key].values)
    x2 = np.mean(brc[key].values)
    logging.info(f"Made DGE {mtx.shape}, median/mean count: {x1:.1f}/{x2:.1f}")

    theta = lda.transform(mtx)
    brc['x'] = brc.j.map(lambda x : x.split('_')[1])
    brc['y'] = brc.j.map(lambda x : x.split('_')[2])
    brc['j'] = brc.j.map(lambda x : x.split('_')[0])
    brc = pd.concat((brc, pd.DataFrame(theta, columns = factor_header)), axis = 1)
    brc['topK'] = np.argmax(theta, axis = 1).astype(int)
    brc['topP'] = np.max(theta, axis = 1)
    brc = brc.astype(dtp)
    if nbatch == 0:
        brc.to_csv(res_f, sep='\t', mode='w', float_format="%.5f", index=False, header=True)
    else:
        brc.to_csv(res_f, sep='\t', mode='a', float_format="%.5f", index=False, header=False)
    nbatch += 1
    df = copy.copy(left)

logging.info(f"Finished ({nbatch})")
