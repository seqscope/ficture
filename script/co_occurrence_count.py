# Count co-occurrence of gene pairs

import sys, os, gzip, copy, gc, time, argparse, logging, pickle
import numpy as np
import pandas as pd
from scipy.sparse import *

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import StreamUnit

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--feature', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--key', type=str, default = 'gn', help='')
parser.add_argument('--unit_id', type=str, default = 'random_index', help='')
parser.add_argument('--feature_id', type=str, default = 'gene', help='')
parser.add_argument('--min_ct_per_feature', type=int, default=1, help='')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

logging.basicConfig(level= getattr(logging, "INFO", None))

key = args.key
unit_id = args.unit_id
feature_id = args.feature_id
min_ct_per_feature = args.min_ct_per_feature

feature = pd.read_csv(args.feature, sep='\t', header=0, usecols = [feature_id, key])
feature = feature.loc[feature[key] > min_ct_per_feature]
feature.sort_values(by = key, ascending = False, inplace=True)
feature.drop_duplicates(subset=feature_id, inplace=True)
feature_kept = list(feature.gene)
ft_dict = {x:i for i,x in enumerate(feature_kept)}
M = len(ft_dict)
logging.info(f"Calculate co-occurrence for {M} genes")

Q = csr_array(([],([],[])), shape=(M,M), dtype=float)
Qdiag = np.zeros(M)
n_unit = 0
csv_reader = pd.read_csv(args.input, sep='\t', header=0, \
                         usecols=[unit_id,feature_id,key],
                         dtype = {unit_id:str,feature_id:str,key:int},
                         chunksize=1000000)
unit_reader = StreamUnit(reader=csv_reader, unit_id=unit_id, key=key, ft_dict=ft_dict)
for mtx in unit_reader.get_matrix():
    unit_sum = mtx.sum(axis = 1)
    size_scale =  np.sqrt(unit_sum * (unit_sum-1)).reshape((-1, 1))
    mtx = mtx.multiply(1/size_scale)
    Q += mtx.T @ mtx
    Qdiag += mtx.multiply(1/size_scale).sum(axis = 0)
    n_unit += mtx.shape[0]
    logging.info(f"Proceesed {n_unit} units")
    if args.debug:
        break

Q = Q - diags([Qdiag], offsets=[0]).tocsr()
obj = {"Q":Q, "n_unit":n_unit, "feature":feature_kept}
pickle.dump(obj, open(args.output, 'wb'))
