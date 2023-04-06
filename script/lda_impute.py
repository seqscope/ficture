import sys, os, argparse, logging, gzip, csv, copy, re, time, importlib, warnings, pickle
import numpy as np
import pandas as pd

from scipy.sparse import *
from sklearn.preprocessing import normalize

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import factor_space

parser = argparse.ArgumentParser()

# Innput and output info
parser.add_argument('--input', type=str, help='')
parser.add_argument('--model', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--factor_map', type=str, help='')
parser.add_argument('--gene_list', type=str, default='', help='')
parser.add_argument('--gene_list_file', type=str, default='', help='')
parser.add_argument('--impute_resolution', type=float, default=.5, help='')
# Data realted parameters
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--key', type=str, default = 'gn', help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
parser.add_argument('--index_axis', type=str, default = 'Y', help='')
# Control the size of neighborhood to use to guess factor
parser.add_argument('--radius', type=float, default=36, help='Maximum distance to use for guessing factor')
parser.add_argument('--knn', type=int, default=1, help='')
# Other
parser.add_argument('--log', type=str, default = '', help='files to write log to')
parser.add_argument('--debug', type=int, default=0, help='debug mode')
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
if not os.path.exists(args.factor_map):
    sys.exit("ERROR: cannot find factor_map file")

### Basic parameterse
mu_scale = 1./args.mu_scale
key = args.key.lower()
chunk_size = 200000

### Target gene to impute
target_gene_list = []
if args.gene_list != '':
    target_gene_list = args.gene_list.split(',')
else:
    if not os.path.exists(args.gene_list_file):
        sys.exit("ERROR: cannot find gene_list file. One of --gene_list or --gene_list_file has to be provided")
    with open(args.gene_list_file, 'r') as f:
        target_gene_list = f.read().splitlines()

### Load model
model = pd.read_csv(args.model, sep='\t')
gene_kept = model["gene"].tolist()
model_gene_set = set(gene_kept)
target_gene_list = [v for v in target_gene_list if v in model_gene_set]
ft_dict = {x:i for i,x in enumerate( gene_kept ) }
M = len(ft_dict)
model = np.array(model.iloc[:,1:]).T
K = model.shape[0]
target_gene_indx = [ft_dict[x] for x in target_gene_list]
T = len(target_gene_indx)
logging.info(f"{K} factors x {M} genes. {T} target genes are in the provided model.")

### Load factor map
factor_map = factor_space(args.factor_map, debug = args.debug)

def clps(df, resolution, key):
    df['X'] = (df.X / resolution).astype(int)
    df['Y'] = (df.Y / resolution).astype(int)
    df['j'] = list(zip(df.X, df.Y))
    df = df.groupby(by = ['j','gene']).agg({key:sum}).reset_index()
    df['X'] = df.j.map(lambda x : x[0]) * resolution
    df['Y'] = df.j.map(lambda x : x[1]) * resolution
    return df

# Record 1st and 2nd moments
res = pd.DataFrame()
out_full = args.output + '.tsv.gz'
out_summary = args.output + '.summary.tsv.gz'

oheader = ['X','Y','brc_total'] + target_gene_list + [x+"_obs" for x in target_gene_list]
with gzip.open(out_full, 'wt') as wf:
    wf.write('\t'.join(oheader) + '\n')
left_over = pd.DataFrame()
dty = {'X':int,'Y':int,'gene':str,key:int}
for chunk in pd.read_csv(args.input, sep='\t', chunksize=chunk_size,\
                         usecols = ['X','Y','gene',key], dtype=dty):
    chunk.X *= mu_scale
    chunk.Y *= mu_scale
    chunk = clps(chunk, resolution = args.impute_resolution, key = key)
    last_indx = chunk[args.index_axis].max()
    df = pd.concat([left_over, chunk.loc[~chunk[args.index_axis].eq(last_indx), :]])
    left_over = copy.copy(chunk.loc[chunk[args.index_axis].eq(last_indx), :])
    if df.shape[0] == 0:
        continue
    df['brc_total'] = df.groupby(by = 'j')[key].transform(sum)
    brc = df.loc[:, ['j', 'X', 'Y', 'brc_total']].drop_duplicates(subset = 'j')
    # impute factor loading from neighboring anchor points
    theta = factor_map.impute_factor_loading(pos = np.array(brc.loc[:, ['X','Y']]), k = args.knn, radius = args.radius, include_self = True) # N x K
    v = theta.sum(axis = 1)
    if args.debug:
        print(sum(v > 0), len(v))
    if sum(v > 0) == 0:
        continue
    theta = normalize(theta, norm='l1', axis=1, copy=False)
    result = theta @ model[:, target_gene_indx] # N x T
    result = pd.DataFrame(result, columns = target_gene_list)
    result['j'] = brc.j.values
    result = pd.concat([result, result])
    result = result.merge(right = brc[['j','X','Y','brc_total']], on = 'j', how = 'left')
    # observed counts
    for x in target_gene_list:
        obs = df.loc[df.gene.eq(x), ['j', key]].rename(columns = {key:x+'_obs'})
        result = result.merge(right = obs, on = 'j', how = 'left')
    result.fillna(0, inplace=True)
    for x in target_gene_list:
        result[x+'_obs'] = result[x+'_obs'].astype(int)
    result.X.map("{:.2f}".format).astype(str)
    result.Y.map("{:.2f}".format).astype(str)
    result.loc[:, oheader].to_csv(out_full, sep='\t', index=False, header=False, mode='a', float_format="%.3e")
    if args.debug:
        obs_sum = result.loc[:, [x+'_obs' for x in target_gene_list]].sum(axis = 0)
        report_list = [target_gene_list[x] for x in np.argsort(-obs_sum)[:10]]
        for x in report_list:
            t = np.mean(result.loc[result[x+'_obs'] > 0, x] )
            f = np.mean(result.loc[result[x+'_obs'] <= 0, x] )
            print(f"{x}: {t:.2e} v.s. {f:.2e}, {t/f:.2f}")
    for k in target_gene_list:
        adt = {'Count':sum}
        adt[k] = lambda x : np.sum(x**2 )
        result['Count'] = 1
        ct = result.groupby(by = k+'_obs').agg({k:sum, 'Count':sum}).reset_index().rename(columns = {k:'probSum', k+'_obs':'obs'})
        sq = result.groupby(by = k+'_obs').agg(adt).reset_index().rename(columns = {k:'sqSum', k+'_obs':'obs'})
        ct = ct.merge(right = sq[['obs','sqSum']], on = 'obs', how = 'inner')
        ct['gene'] = k
        res = pd.concat([res, ct])

ct = res.groupby(by = ['gene', 'obs']).agg({x:sum for x in ['probSum','sqSum','Count']}).reset_index()
ct['ProbAvg'] = ct.probSum / ct.Count
ct['std'] = np.sqrt( (ct.sqSum / ct.Count).values - ct.ProbAvg.values**2 )
ct.to_csv(out_summary, sep='\t', index=False, header=True)
