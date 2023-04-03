import sys, os, argparse, logging, gzip, csv, copy, re, time, importlib, warnings, pickle
import subprocess as sp
import numpy as np
import pandas as pd

from scipy.sparse import *
import sklearn.neighbors
from sklearn.preprocessing import normalize

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utilt
from data_loader import factor_space_stream
from read_chunk_fn import SlidingPosteriorCount

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
parser.add_argument('--precision', type=float, default=.5, help='If positive, collapse pixels within X um. Used for computing local factor profiles.')
parser.add_argument('--index_axis', type=str, default = 'Y', help='')
# Control the size of neighborhood to use to guess factor loading
parser.add_argument('--radius', type=float, default=5, help='')
parser.add_argument('--halflife', type=float, default=.7, help='')
parser.add_argument('--knn', type=int, default=6, help='')
# Control the size of neighborhood to use to adjust factor profile
parser.add_argument('--factor_adjust_window', type=float, default=800, help='')
parser.add_argument('--factor_adjust_window_slide', type=int, default=4, help='')
# Weight between local and global factor profile
parser.add_argument('--weight_local', type=float, default=.3, help='')
parser.add_argument('--local_count_lower', type=float, default=1000, help='Minimum factor specific read counts to use local factor profile.')
parser.add_argument('--local_count_upper', type=float, default=10000, help='Threshold of factor specific read counts. Above thie threshold the local weight provided in --weight_local will be used.')
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
precision = args.precision
key = args.key.lower()
slide_step = args.factor_adjust_window_slide
chunk_size = 200000
unit_block = args.factor_adjust_window / args.factor_adjust_window_slide

### Target gene to impute
target_gene_unmodeled = []
target_gene_modeled = []
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
for x in target_gene_list:
    if x in model_gene_set:
        target_gene_modeled.append(x)
    else:
        target_gene_unmodeled.append(x)
if args.weight_local > 0:
    gene_kept += target_gene_unmodeled
else:
    target_gene_unmodeled = []
    target_gene_list = copy.copy(target_gene_modeled)
ft_dict = {x:i for i,x in enumerate( gene_kept ) }
M = len(ft_dict)
model = np.array(model.iloc[:,1:]).T
K = model.shape[0]
model = np.hstack((model, np.zeros((K, len(target_gene_unmodeled)))))
model = normalize(model, norm='l1', axis=1, copy=False) # K x M
T1 = len(target_gene_modeled)
T2 = len(target_gene_unmodeled)
target_gene_indx_modeled = [ft_dict[x] for x in target_gene_modeled]
target_gene_indx_unmodeled = [ft_dict[x] for x in target_gene_unmodeled]
logging.info(f"{K} factors x {M} genes. {T1} target genes are in the provided model and {T2} genes are not. Unmodeled genes will only be imputed based on factor-specific local average.")

### Load factor map
factor_map = factor_space_stream(file = args.factor_map, index_axis = args.index_axis, debug = args.debug)

### Local factor profile
dty = {x:int for x in ['X','Y',key]}
dty.update({x:str for x in ['gene']})
if args.weight_local > 0:
    pixel_reader = pd.read_csv(args.input, sep='\t', chunksize=chunk_size,\
                            usecols = ['X','Y','gene',key], dtype=dty)
    local_profile = SlidingPosteriorCount(pixel_reader, index_axis = args.index_axis, key = key, factor_file = args.factor_map, ft_dict = ft_dict, size_um = args.factor_adjust_window, slide_step = slide_step, mu_scale = mu_scale, precision = precision, radius = args.radius, debug = args.debug)


def clps(df, resolution, key):
    df['X'] = (df.X / resolution).astype(int)
    df['Y'] = (df.Y / resolution).astype(int)
    df['j'] = list(zip(df.X, df.Y))
    df = df.groupby(by = ['j','gene']).agg({key:sum}).reset_index()
    df['X'] = df.j.map(lambda x : x[0]) * resolution
    df['Y'] = df.j.map(lambda x : x[1]) * resolution
    return df


oheader = ['X','Y','brc_total'] + target_gene_list + [x+"_obs" for x in target_gene_list]
with gzip.open(args.output, 'wt') as wf:
    wf.write('\t'.join(oheader) + '\n')
left_over = pd.DataFrame()
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
    yst, yed = brc[args.index_axis].min(), brc[args.index_axis].max()
    # impute factor loading from neighboring pixels
    theta = factor_map.impute_factor_loading(pos = np.array(brc.loc[:, ['X','Y']]), k = args.knn, radius = args.radius, halflife = args.halflife, include_self = False) # N x K
    theta = normalize(theta, norm='l1', axis=1, copy=False)
    if args.weight_local > 0:
        # load local factor specific expression profile
        local_profile.update_reference(yst, yed)
    x = (brc.X / unit_block).astype(int)
    y = (brc.Y / unit_block).astype(int)
    brc['block'] = list(zip(x,y))
    result = pd.DataFrame()
    for b in brc.block.unique():
        block_indx = brc.block == b
        x = np.median(brc.loc[block_indx, 'X'] )
        y = np.median(brc.loc[block_indx, 'Y'] )
        if args.weight_local > 0:
            d, beta = local_profile.query(x, y)
            if d > args.factor_adjust_window / 2:
                continue
            beta = normalize(beta, norm = 'l1', axis = 1, copy = False)
            local_count = np.clip(beta.sum(axis = 1), args.local_count_lower, args.local_count_upper) # K
            local_weight = ((local_count - args.local_count_lower)/(args.local_count_upper - args.local_count_lower) * args.weight_local).reshape((K, 1)) # K
            prob_hat_modeled = theta[block_indx, :] @ \
                (np.multiply(model[:, target_gene_indx_modeled], 1-local_weight) +\
                np.multiply(beta[:, target_gene_indx_modeled], local_weight)) # N x T1
            prob_hat_unmodeled = theta[block_indx, :] @ beta[:, target_gene_indx_unmodeled] # N x T2
        else:
            prob_hat_modeled = theta[block_indx, :] @ model[:, target_gene_indx_modeled] # N x T1
            prob_hat_unmodeled = np.empty((prob_hat_modeled.shape[0], 0))
        prob_hat = np.hstack((prob_hat_modeled, prob_hat_unmodeled))
        prob_hat = pd.DataFrame(prob_hat, columns = target_gene_modeled + target_gene_unmodeled)
        prob_hat['j'] = brc.loc[block_indx, 'j'].values
        prob_hat['brc_total'] = brc.loc[block_indx, 'brc_total'].values
        result = pd.concat([result, prob_hat])
    result = result.merge(right = brc[['j','X','Y']], on = 'j', how = 'left')
    for x in target_gene_list:
        obs = df.loc[df.gene.eq(x), ['j', key]].rename(columns = {key:x+'_obs'})
        result = result.merge(right = obs, on = 'j', how = 'left')
    result.fillna(0, inplace=True)
    for x in target_gene_list:
        result[x+'_obs'] = result[x+'_obs'].astype(int)
    result.X.map("{:.2f}".format)
    result.Y.map("{:.2f}".format)
    result.loc[:, oheader].to_csv(args.output, sep='\t', index=False, header=False, mode='a', float_format="%.3e")
    if args.debug:
        obs_sum = result.loc[:, [x+'_obs' for x in target_gene_list]].sum(axis = 0)
        report_list = [target_gene_list[x] for x in np.argsort(-obs_sum)[:10]]
        for x in report_list:
            t = np.mean(result.loc[result[x+'_obs'] > 0, x] )
            f = np.mean(result.loc[result[x+'_obs'] <= 0, x] )
            print(f"{x}: {t:.2e} v.s. {f:.2e}, {t/f:.2f}")
