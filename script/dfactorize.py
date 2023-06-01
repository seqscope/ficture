# Factor analysis with pairwise factor similarity penalty
import sys, os, copy, gzip, time, logging
import pickle, argparse
import numpy as np
import pandas as pd
import random as rng

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lda_minibatch import Minibatch
from unit_loader import UnitLoader
from online_penalized_lda import OnlineLDAPenalized

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output_pref', type=str, help='')
parser.add_argument('--nFactor', type=int, help='')
parser.add_argument('--feature', type=str, default='', help='')

parser.add_argument('--key', type=str, default = 'gn', help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
parser.add_argument('--unit_id', type=str, default = 'random_index', help='')
parser.add_argument('--min_count_per_feature', type=int, default=50, help='')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
parser.add_argument('--epoch_id_length', type=int, default=-1, help='')

# Learning parameters
parser.add_argument('--seed', type=int, default=-1, help='')
parser.add_argument('--verbose', type=int, default=0, help='')
parser.add_argument('--thread', type=int, default=1, help='')
parser.add_argument('--epoch', type=int, default=-1, help='How many times to loop through the full data')
parser.add_argument('--minibatch_size', type=int, default=516, help='')
parser.add_argument('--total_n_unit', type=int, default=int(1e5), help='')
parser.add_argument('--zeta', type=float, default=.1, help='')
parser.add_argument('--kappa', type=float, default=.7, help='')
parser.add_argument('--tau', type=int, default=9, help='')

parser.add_argument('--chunksize', type=int, default=500000, help='')
parser.add_argument('--log', default = '', type=str, help='files to write log to')
parser.add_argument('--transform_full', action='store_true')
parser.add_argument('--skip_transform', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()
if not os.path.exists(args.input):
    sys.exit("ERROR: cannot find input file.")
if args.nFactor <= 0:
    sys.exit("ERROR: --nFactor should be a positive integer")
output_pref = args.output_pref
if not os.path.exists(os.path.dirname(output_pref)):
    os.makedirs(os.path.dirname(output_pref))
if args.log != '':
    try:
        logging.basicConfig(filename=args.log, filemode='a', encoding='utf-8', level=logging.INFO)
    except:
        logging.basicConfig(level= getattr(logging, "INFO", None))
else:
    logging.basicConfig(level= getattr(logging, "INFO", None))

seed = args.seed
if seed <= 0:
    seed = int(time.time())
rng = np.random.default_rng(seed)

key = args.key.lower()
unit_id = args.unit_id.lower()
### Basic parameterse
b_size = args.minibatch_size
K = args.nFactor
factor_header = [str(x) for x in range(K)]
### Input
with gzip.open(args.input, 'rt') as rf:
    header = rf.readline().strip().split('\t')
header = [x.lower() for x in header]
if unit_id not in header or key not in header:
    sys.exit("ERROR: --unit_id or --key is not in the input file")
header[header.index(unit_id)] = "unit"

feature=pd.read_csv(args.feature, sep='\t', header=0)
feature.columns = [x.lower() for x in feature.columns]
feature[key] = feature[key].astype(int)
feature = feature[feature[key] >= args.min_count_per_feature]
feature.sort_values(by=key,ascending=False,inplace=True)
feature.drop_duplicates(subset='gene',keep='first',inplace=True)
feature['freq'] = feature[key].values / feature[key].sum()
feature_kept = list(feature.gene.values)
ft_dict = {x:i for i,x in enumerate( feature_kept ) }
M = len(feature_kept)
logging.info(f"{M} genes will be used")

adt = {'unit':str, 'x': float, 'y':float, 'gene':str, key:int}
reader = pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=args.chunksize,\
            skiprows=1, names=header, usecols=list(adt.keys()), dtype=adt)
# Minibatch reader
batch_obj = UnitLoader(reader, ft_dict, key, \
                       batch_id_prefix = args.epoch_id_length, \
                       min_ct_per_unit = args.min_ct_per_unit)
# Set up model
model = OnlineLDAPenalized(vocab=feature_kept, K = K, N = args.total_n_unit,\
                           alpha = None, eta = feature.freq.values * K,\
                           tau0=args.tau, kappa=args.kappa, zeta=args.zeta,
                           iter_inner = 50, tol = 1e-4, rng=rng,\
                           verbose = args.verbose, thread = args.thread, proximal=False)
model.init_global_parameter()
# Stochastic model fitting
t0 = time.time()
n_batch = 0
while batch_obj.update_batch(b_size):
    batch = Minibatch(batch_obj.mtx)
    scores = model.update_lambda(batch)
    n_batch += 1
    t1 = time.time() - t0
    if args.debug or n_batch % 10 == 0:
        print(f"Model fitting {len(batch_obj.batch_id_list)}-{n_batch}, {t1/60:2f}min")
        post_weight = model._lambda.sum(axis=1)
        post_weight /= post_weight.sum()
        post_weight.sort()
        k = min(model._K // 2, 10)
        print(f"Top {k} topic total weight {post_weight[-k:].sum():.4f}")
        print(np.around(post_weight[-k:], 3))
    if args.epoch_id_length > 0 and len(batch_obj.batch_id_list) > args.epoch:
        break
    if args.debug and n_batch > 2:
        break

if args.debug:
    sys.exit()

# Output model
out_f = args.output_pref + ".model_matrix.tsv.gz"
pd.concat([pd.DataFrame({'gene': feature_kept}),\
            pd.DataFrame(model._lambda.T ,\
            columns = factor_header, dtype='float64')],\
            axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.4e', compression={"method":"gzip"})
if args.skip_transform:
    sys.exit()

# Transform
post_count = np.zeros((K, M))
epoch_id = set()
reader = pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=args.chunksize,\
            skiprows=1, names=header, usecols=list(adt.keys()), dtype=adt)
batch_obj = UnitLoader(reader, ft_dict, key, \
                       batch_id_prefix = args.epoch_id_length, \
                       min_ct_per_unit = args.min_ct_per_unit)
n_batch = 0
oheader = ["unit",key,"x","y","topK","topP"]+factor_header
out_f = args.output_pref + ".fit_result.tsv.gz"
with gzip.open(out_f, 'wt') as wf:
    wf.write('\t'.join(oheader) + '\n')
t0 = time.time()
while batch_obj.update_batch(b_size):
    theta = model.transform(batch_obj.mtx)
    post_count += np.array(theta.T @ batch_obj.mtx)
    n_batch += 1
    t1 = time.time() - t0
    if n_batch % 10 == 0:
        print(f"Transform {len(epoch_id)}-{n_batch}, {t1/60:2f}min")
    if (not args.transform_full) and args.epoch_id_length > 0:
        if len(batch_obj.batch_id_list) > args.epoch:
            break
    batch_obj.brc['topK'] = np.argmax(theta, axis = 1)
    batch_obj.brc['topP'] = theta.max(axis = 1)
    batch_obj.brc = pd.concat([batch_obj.brc, pd.DataFrame(theta, columns=factor_header)], axis=1)
    batch_obj.brc['x'] = batch_obj.brc.x.map('{:.2f}'.format)
    batch_obj.brc['y'] = batch_obj.brc.y.map('{:.2f}'.format)
    batch_obj.brc[oheader].to_csv(out_f, sep='\t', index=False, header=False, float_format='%.4e', mode='a', compression={"method":"gzip"})

out_f = args.output_pref + ".posterior.count.tsv.gz"
pd.concat([pd.DataFrame({'gene': feature_kept}),\
           pd.DataFrame(post_count.T, dtype='float64',\
                        columns = [str(k) for k in range(K)])],\
            axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})
