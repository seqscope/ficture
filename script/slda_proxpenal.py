import sys, os, argparse, logging, gzip, copy, re, time, warnings, pickle
import numpy as np
import pandas as pd

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from online_slda import OnlineLDA
from pixel_loader import PixelMinibatch

parser = argparse.ArgumentParser()

# Innput and output info
parser.add_argument('--input', type=str, help='')
parser.add_argument('--model', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--anchor', type=str, help='')
parser.add_argument('--anchor_in_um', action='store_true')

# Data realted parameters
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--key', type=str, default = 'gn', help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
parser.add_argument('--batch_id', type=str, default = 'random_index', help='Input has to have a column with this name indicating the minibatch id')
parser.add_argument('--precision', type=float, default=.25, help='If positive, collapse pixels within X um.')

# Learning related parameters
parser.add_argument('--zeta', type=float, default=.2, help='')
parser.add_argument('--anchor_penal_radius', type=float, default=-1, help='')
parser.add_argument('--neighbor_radius', type=float, default=25, help='The radius (um) of each anchor point\'s territory')
parser.add_argument('--halflife', type=float, default=0.7, help='Control the decay of distance-based weight')
parser.add_argument('--theta_init_bound_multiplier', type=float, default=.2, help='')
parser.add_argument('--inner_max_iter', type=int, default=30, help='')
parser.add_argument('--gamma_max_iter', type=int, default=15, help='')
parser.add_argument('--verbose', type=int, default=1, help='')

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
if args.zeta <= 0 or args.zeta >= 1:
    sys.exit("ERROR: zeta has to be between 0 and 1")

### Basic parameterse
mu_scale = 1./args.mu_scale
radius = args.neighbor_radius
precision = args.precision
key = args.key.lower()
batch_id = args.batch_id.lower()
chunk_size = 500000
adj_penal = args.anchor_penal_radius
if adj_penal < 0:
    adj_penal = radius

### Load model
model = pickle.load(open( args.model, "rb" ))
gene_kept = model.feature_names_in_
ft_dict = {x:i for i,x in enumerate( gene_kept ) }
K, M = model.components_.shape
factor_header = [str(x) for x in range(K)]
init_bound = 1./K * args.theta_init_bound_multiplier
logging.info(f"{M} genes and {K} factors are read from input model")

### Input pixel info (input has to contain certain columns with correct header)
with gzip.open(args.input, 'rt') as rf:
    oheader = rf.readline().strip().split('\t')
oheader = [x.lower() if len(x) > 1 else x.upper() for x in oheader]
input_header = [batch_id,"X","Y","gene",key]
dty = {x:int for x in ['X','Y',key]}
dty.update({x:str for x in [batch_id, 'gene']})
mheader = [x for x in input_header if x not in oheader]
if len(mheader) > 0:
    mheader = ", ".join(mheader)
    sys.exit(f"Input misses the following column: {mheader}.")

pixel_reader = pd.read_csv(args.input, sep='\t', chunksize=chunk_size, \
            skiprows=1, names=oheader, usecols=input_header, dtype=dty)

pixel_obj = PixelMinibatch(pixel_reader, ft_dict, \
                           batch_id, key, mu_scale, \
                           radius=radius, halflife=args.halflife,\
                           adj_penal = adj_penal,
                           precision=args.precision, thread=1)
### anchor info
pixel_obj.load_anchor(args.anchor, args.anchor_in_um)
logging.info(f"Read {pixel_obj.grid_info.shape[0]} grid points")

slda = OnlineLDA(vocab=gene_kept, K=K, N=1e6, zeta=args.zeta, \
                 iter_gamma = args.gamma_max_iter,\
                 iter_inner=args.inner_max_iter, verbose = args.verbose)
slda.init_global_parameter(model.components_)

post_count = np.zeros((K, M))
n_batch = 0
while True:
    read_n_batch = pixel_obj.read_chunk(1)
    print(f"Read {read_n_batch} batches ({pixel_obj.dge_mtx.shape})")
    pcount, pixel, anchor  = pixel_obj.run_chunk_penalized(slda, init_bound)
    print(f"Output {pixel.shape[0]} pixels and {anchor.shape[0]} anchors")
    pixel.X = pixel.X.map('{:.2f}'.format)
    pixel.Y = pixel.Y.map('{:.2f}'.format)
    if n_batch == 0:
        pixel.to_csv(args.output+".pixel.tsv.gz", sep='\t', index=False, header=True, mode='w', float_format="%.2e", compression={"method":"gzip"})
    else:
        pixel.to_csv(args.output+".pixel.tsv.gz", sep='\t', index=False, header=False, mode='a', float_format="%.2e", compression={"method":"gzip"})
    anchor.X = anchor.X.map('{:.2f}'.format)
    anchor.Y = anchor.Y.map('{:.2f}'.format)
    if n_batch == 0:
        anchor.to_csv(args.output+".anchor.tsv.gz", sep='\t', index=False, header=True, mode='w', float_format="%.2e", compression={"method":"gzip"})
    else:
        anchor.to_csv(args.output+".anchor.tsv.gz", sep='\t', index=False, header=False, mode='a', float_format="%.2e", compression={"method":"gzip"})
    n_batch += read_n_batch
    post_count += pcount
    if not pixel_obj.file_is_open:
        break

### Output posterior summaries
out_f = args.output + ".posterior.count.tsv.gz"
pd.concat([pd.DataFrame({'gene': gene_kept}),\
           pd.DataFrame(post_count.T, dtype='float64',\
                        columns = factor_header)],\
        axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})
