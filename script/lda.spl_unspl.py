#### Treating spl and unspl as two separate features per gene ####

import sys, os, copy, gzip, logging
import pickle, argparse
import numpy as np
import pandas as pd
import random as rng

from scipy.sparse import coo_matrix, coo_array
from sklearn.preprocessing import normalize
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--feature', type=str, default='', help='')
parser.add_argument('--output_path', type=str, default='', help='')
parser.add_argument('--output_pref', type=str, default='', help='')
parser.add_argument('--identifier', type=str, default='', help='')
parser.add_argument('--feature_key', default = 'gene', type=str, help='')
parser.add_argument('--spl_key', default = 'spl', type=str, help='')
parser.add_argument('--unspl_key', default = 'unspl', type=str, help='')
parser.add_argument('--unit_id', default = 'random_index', type=str, help='')

parser.add_argument('--nFactor', type=int, default=10, help='')
parser.add_argument('--minibatch_size', type=int, default=256, help='')
parser.add_argument('--min_count_per_feature', type=int, default=50, help='')
parser.add_argument('--min_count_per_feature_unspl', type=int, default=100, help='')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
parser.add_argument('--verbose', type=int, default=1, help='')
parser.add_argument('--epoch', type=int, default=-1, help='How many times to loop through the full data')
parser.add_argument('--epoch_id_length', type=int, default=-1, help='')
parser.add_argument('--transform_epoch_id', type=str, nargs='*', default=[], help="")
parser.add_argument('--use_model', type=str, default='', help="Use provided model to transform input data")
parser.add_argument('--log', default = '', type=str, help='files to write log to')
parser.add_argument('--seed', type=int, default=-1, help='')
parser.add_argument('--thread', type=int, default=1, help='')
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

chunksize=500000
gene = args.feature_key.lower()
spl = args.spl_key.lower()
unspl = args.unspl_key.lower()
unitid = args.unit_id.lower()
key = 'tot'

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
if not os.path.exists(args.feature):
    sys.exit("ERROR: cannot find input feature file.")
with gzip.open(args.input, 'rt') as rf:
    header = rf.readline().strip().split('\t')
header = [x.lower() for x in header] #???


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
    feature=pd.read_csv(args.feature, sep='\t', header=0)
    feature.columns = [x.lower() for x in feature.columns]
    feature[key] = feature[spl] + feature[unspl]
    feature = feature[feature[key] >= args.min_count_per_feature]
    feature.sort_values(by=key,ascending=False,inplace=True)
    feature.drop_duplicates(subset=gene,keep='first',inplace=True)
    gene_kept = list(feature.gene.values)
    spl_kept = list(feature.loc[feature[spl] > args.min_count_per_feature, gene])
    unspl_kept = ['unspl_'+x for x in feature.loc[feature[unspl] > args.min_count_per_feature_unspl, gene].values] + ['_unspl']
    feature_kept = spl_kept + unspl_kept
    ft_dict = {x:i for i,x in enumerate( feature_kept ) }
    M = len(feature_kept)
    logging.info(f"{M} feature will be used, {len(spl_kept)} spl and {len(unspl_kept)} unspl")

### Stochastic model fitting
adt = {unitid:str, 'x': str, 'y':str, gene:str, spl:int, unspl:int}
if not read_model:
    logging.info(f"Start fitting model ... model will be stored in\n{model_f}")
    lda = LDA(n_components=K, learning_method='online', batch_size=b_size, n_jobs = args.thread, verbose = 0, random_state=seed)
    epoch_id = set()
    df = pd.DataFrame()       # Store data with separated spl vs unspl features
    leftover = pd.DataFrame() # Store original input
    for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=chunksize, skiprows=1, names=header, usecols=[unitid,gene,spl,unspl], dtype=adt):

        last_chunk = chunk.shape[0] < chunksize
        last_indx = chunk[unitid].iloc[-1]
        df = pd.concat([leftover, chunk[~chunk[unitid].eq(last_indx)]])
        if len(df[unitid].unique()) < b_size * 1.5 and not last_chunk: # Left to next chunk
            leftover = pd.concat((leftover, chunk))
            continue
        leftover = copy.copy(chunk[chunk[unitid].eq(last_indx)] )

        unspl_df = copy.copy(df.loc[df[unspl] > 0, [unitid,unspl,gene]])
        unspl_df[gene] = 'unspl_' + unspl_df.gene.values
        left_unspl = unspl_df.loc[~unspl_df.gene.isin(ft_dict), :].groupby(by=unitid,as_index=False).agg({unspl:sum})
        left_unspl[gene] = '_unspl'
        unspl_df = pd.concat([unspl_df.loc[unspl_df.gene.isin(ft_dict), :], left_unspl])
        unspl_df.rename(columns = {unspl:key}, inplace=True)
        df = pd.concat([df.loc[(df[spl] > 0) & df[gene].isin(ft_dict), \
                               [unitid,spl,gene]].rename(columns = {spl:key}), unspl_df])

        if args.epoch_id_length > 0:
            v = df[unitid].map(lambda x: x[:args.epoch_id_length]).values
            v = set(v)
            epoch_id.update(v)

        # Total mulecule count per unit (spl + unspl)
        brc = df.groupby(by = [unitid]).agg({key: sum}).reset_index()
        brc = brc[brc[key] > args.min_ct_per_unit]
        brc.index = range(brc.shape[0])
        df = df[df[unitid].isin(brc[unitid].values)]
        # Make DGE
        barcode_kept = list(brc[unitid].values)
        bc_dict = {x:i for i,x in enumerate( barcode_kept )}
        indx_row = [bc_dict[x] for x in df[unitid]]
        indx_col = [ft_dict[x] for x in df[gene]]
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
        logging.info(f"Leftover size {len(leftover)}")
        if args.epoch > 0 and len(epoch_id) > args.epoch:
            break

    lda.feature_names_in_ = feature_kept
    pickle.dump( lda, open( model_f, "wb" ) )
    out_f = model_f.replace("model.p", "model_matrix.tsv.gz")
    pd.concat([pd.DataFrame({gene: lda.feature_names_in_}),\
                pd.DataFrame(lda.components_.T,\
                columns = factor_header, dtype='float64')],\
                axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.4e', compression={"method":"gzip"})


if args.skip_transform:
    sys.exit()

### Rerun all units once and store results
dtp = {'topK':int, key:int,unitid:str, 'x':str, 'y':str}
dtp.update({x:float for x in ['topP']+factor_header})
res_f = output_pref+".fit_result.tsv.gz"
logging.info(f"Result file {res_f}")

post_count = np.zeros((K, M))
epoch_id = '' # default case is to run on non-ovarlapping units
df = pd.DataFrame()       # Store data with separated spl vs unspl features
leftover = pd.DataFrame() # Store original input
nbatch = 0
for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=chunksize, skiprows=1, names=header, usecols=[unitid,'x','y',gene,spl,unspl], dtype=adt):

    end_of_epoch = False
    if not args.transform_full and args.epoch_id_length > 0:
        v = chunk[unitid].map(lambda x: x[:args.epoch_id_length]).values
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

    last_chunk = chunk.shape[0] < chunksize
    last_indx = chunk[unitid].iloc[-1]
    df = pd.concat([leftover, chunk[~chunk[unitid].eq(last_indx)]])
    if len(df[unitid].unique()) < b_size * 1.5 and not last_chunk: # Left to next chunk
        leftover = pd.concat((leftover, chunk))
        continue
    leftover = copy.copy(chunk[chunk[unitid].eq(last_indx)] )

    df[unitid] = df[unitid].values + '_' + df.x.values + '_' + df.y.values
    unspl_df = copy.copy(df.loc[df[unspl] > 0, [unitid,unspl,gene]])
    unspl_df[gene] = 'unspl_' + unspl_df.gene.values
    left_unspl = unspl_df.loc[~unspl_df.gene.isin(ft_dict), :].groupby(by=unitid,as_index=False).agg({unspl:sum})
    left_unspl[gene] = '_unspl'
    unspl_df = pd.concat([unspl_df.loc[unspl_df.gene.isin(ft_dict), :], left_unspl])
    unspl_df.rename(columns = {unspl:key}, inplace=True)
    df = pd.concat([df.loc[(df[spl] > 0) & df[gene].isin(ft_dict), \
                            [unitid,spl,gene]].rename(columns = {spl:key}), unspl_df])

    # Total mulecule count per unit (spl + unspl)
    brc = df.groupby(by = [unitid]).agg({key: sum}).reset_index()
    brc = brc[brc[key] > args.min_ct_per_unit]
    brc.index = range(brc.shape[0])
    df = df[df[unitid].isin(brc[unitid].values)]
    # Make DGE
    barcode_kept = list(brc[unitid].values)
    bc_dict = {x:i for i,x in enumerate( barcode_kept )}
    indx_row = [bc_dict[x] for x in df[unitid]]
    indx_col = [ft_dict[x] for x in df[gene]]
    N = len(barcode_kept)
    mtx = coo_matrix((df[key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
    logging.info(f"Made DGE {mtx.shape}")

    theta = lda.transform(mtx)
    post_count += np.array(theta.T @ mtx)
    brc['x'] = brc[unitid].map(lambda x : x.split('_')[1])
    brc['y'] = brc[unitid].map(lambda x : x.split('_')[2])
    brc[unitid] = brc[unitid].map(lambda x : x.split('_')[0])
    brc = pd.concat((brc, pd.DataFrame(theta, columns = factor_header)), axis = 1)
    brc['topK'] = np.argmax(theta, axis = 1).astype(int)
    brc['topP'] = np.max(theta, axis = 1)
    brc = brc.astype(dtp)
    if nbatch == 0:
        brc.to_csv(res_f, sep='\t', mode='w', float_format="%.5f", index=False, header=True, compression={"method":"gzip"})
    else:
        brc.to_csv(res_f, sep='\t', mode='a', float_format="%.5f", index=False, header=False, compression={"method":"gzip"})
    nbatch += 1
    if end_of_epoch:
        break

logging.info(f"Finished ({nbatch})")

out_f = output_pref+".posterior.count.tsv.gz"
pd.concat([pd.DataFrame({gene: feature_kept}),\
           pd.DataFrame(post_count.T, dtype='float64',\
                        columns = [str(k) for k in range(K)])],\
        axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})
