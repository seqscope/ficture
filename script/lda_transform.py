import sys, os, gzip, copy, gc, time, argparse, logging, pickle
import numpy as np
import pandas as pd
from scipy.sparse import *
import subprocess as sp

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help='Output file to store factor membership')
parser.add_argument('--outpost', type=str, default='', help='Output file to store posterior count')
parser.add_argument('--model', type=str, help='')

parser.add_argument('--major_axis', type=str, default="Y", help='X or Y')
parser.add_argument('--regions', type=str, default="", help='A file containing region intervals, same as that used by tabix -R (e.g. tab delimited, lane start end)')
parser.add_argument('--region', nargs='*', type=str, default=[], help='lane:Y_start-Y_end (Y axis in barcode coordinate unit), separate by space if multiple regions')
parser.add_argument('--region_um', nargs='*', type=str, default=[], help='lane:Y_start-Y_end (Y axis in um), separate by space if multiple regions')

parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--key', default = 'gn', type=str, help='')
parser.add_argument('--precision', type=int, default=1, help='Number of digits to store spatial location (in um), 0 for integer.')

parser.add_argument('--thread', type=int, default=-1, help='')
parser.add_argument('--n_move', type=int, default=3, help='')
parser.add_argument('--hex_width', type=int, default=24, help='')
parser.add_argument('--hex_radius', type=int, default=-1, help='')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
args = parser.parse_args()

logging.basicConfig(level= getattr(logging, "INFO", None))


# Load model
lda = pickle.load( open(args.model, "rb") )
feature_kept = lda.feature_names_in_
lda.feature_names_in_ = None
ft_dict = {x:i for i,x in enumerate( feature_kept ) }
K, M = lda.components_.shape
factor_header = [str(k) for k in range(K)]
if args.thread > 0:
    lda.n_jobs = args.thread
logging.info(f"Read existing model with {K} factors and {M} genes. Attempting to use {lda.n_jobs} threads")


# Input
if not os.path.exists(args.input):
    sys.exit(f"ERROR: cannot find input file \n {args.input}")
with gzip.open(args.input, 'rt') as rf:
    input_header=rf.readline().strip().split('\t')
key = args.key
if key not in input_header:
    sys.exit(f"Cannot find --key in the input file")
print(input_header)


# basic parameters
mj = args.major_axis
mu_scale = 1./args.mu_scale
radius=args.hex_radius
diam=args.hex_width
n_move = args.n_move
if n_move > diam // 2:
    n_move = diam // 4
ovlp_buffer = diam * 2
if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = int(radius*np.sqrt(3))

cmd = []
if os.path.exists(args.regions):
    cmd =  ["tabix", args.input, "-R", args.regions]
elif len(args.region) > 0:
    cmd = ["tabix", args.input] + args.region
elif len(args.region_um) > 0:
    reg_list = []
    for v in args.region_um:
        if ":" not in v or "-" not in v:
            continue
        l = v.split(':')[0]
        st, ed = v.split(':')[1].split('-')
        st = str(int(float(st) * args.mu_scale) )
        ed = str(int(float(ed) * args.mu_scale) )
        reg_list.append(l+':'+'-'.join([st,ed]) )
    cmd = ["tabix", args.input] + reg_list
if len(cmd) == 0:
    p0 = sp.Popen(["zcat", args.input], stdout=sp.PIPE)
    process = sp.Popen(["tail", "-n", "+2"], stdin=p0.stdout, stdout=sp.PIPE)
else:
    process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT)

logging.info(cmd)
n_unit = 0
dtp = {'topK':int,key:int,'j':str, 'x':str, 'y':str}
dtp.update({x:float for x in ['topP']+factor_header})
dty = {x:int for x in ['X','Y',key]}
dty.update({x:str for x in ['#lane', 'gene']})
post_count = np.zeros((K, M))
df_full = pd.DataFrame()
chunk_size = 1000000
for chunk in pd.read_csv(process.stdout,sep='\t',chunksize=chunk_size,\
                names=input_header, usecols=["#lane","X","Y","gene",key], dtype=dty):
    last_chunk = chunk.shape[0] < chunk_size
    chunk = chunk[(chunk[key] > 0) & chunk.gene.isin(ft_dict)]
    df_full = pd.concat([df_full, chunk])
    df_full['j'] = df_full.X.astype(str) + '_' + df_full.Y.astype(str)
    l1 = df_full['#lane'].iloc[-1]
    axis_range = df_full[mj].iloc[-1] - df_full[mj].iloc[0]
    l_list = df_full['#lane'].unique()

    if (not last_chunk) and len(l_list) == 1 and axis_range < ovlp_buffer * args.mu_scale:
        logging.info(f"Left over size {df_full.shape[0]}.")
        continue

    for l in l_list:
        df = df_full.loc[df_full['#lane'].eq(l)]
        brc = df.groupby(by = ['j','X','Y']).agg({key:sum}).reset_index()
        brc.index = range(brc.shape[0])
        brc['X'] = brc.X.astype(float).values * mu_scale
        brc['Y'] = brc.Y.astype(float).values * mu_scale
        st = brc[mj].min()
        ed = brc[mj].max()
        pts = np.asarray(brc[['X','Y']])
        logging.info(f"Read {brc.shape[0]} pixels in lane {l}, major axis range {st:.2f} - {ed:.2f}.")

        brc["hex_id"] = ""
        offs_x = 0
        offs_y = 0
        while offs_x < n_move:
            while offs_y < n_move:

                # Translate pixel positions to hexagon units
                x,y = pixel_to_hex(pts, radius, offs_x/n_move, offs_y/n_move)
                hex_x, hex_y = hex_to_pixel(x, y, radius, offs_x/n_move, offs_y/n_move)
                hex_crd = ["_".join([str(x[i]), str(y[i])]) for i in range(len(x))]
                hex_pos = {x:(hex_x[i], hex_y[i]) for i,x in enumerate(hex_crd)}

                # Keep only dense enough hexagons
                brc["hex_id"] = hex_crd
                ct = brc.groupby(by = 'hex_id').agg({key:sum, mj:np.median}).reset_index()
                kept_unit = (ct[key] >= args.min_ct_per_unit) &\
                            (ct[mj] > st + diam/2) & (ct[mj] < ed - diam/2)
                kept_unit = list(ct.loc[kept_unit, "hex_id"].values)

                # Make dge matrix
                sub = brc.loc[brc.hex_id.isin(kept_unit), ["hex_id","j"]].merge(right = df[["j","gene",key]], on='j', how = 'inner')
                sub = sub.groupby(by = ['hex_id', 'gene']).agg({key:sum}).reset_index()
                bc_dict  = { x:i for i,x in enumerate(kept_unit)}
                indx_row = [ bc_dict[x] for x in sub['hex_id']]
                indx_col = [ ft_dict[x] for x in sub['gene']]
                N = len(kept_unit)
                mtx = coo_array((sub[key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()

                theta = lda.transform(mtx)
                post_count += np.array(theta.T @ mtx)

                # Output
                result = pd.DataFrame({"j":kept_unit, key:np.array(mtx.sum(axis = 1)).squeeze()})
                result["x"] = result.j.map(lambda x : f"{hex_pos[x][0]:.{args.precision}f}")
                result["y"] = result.j.map(lambda x : f"{hex_pos[x][1]:.{args.precision}f}")
                result = pd.concat((result, pd.DataFrame(theta, columns = factor_header)), axis = 1)
                result['topK'] = np.argmax(theta, axis = 1).astype(int)
                result['topP'] = np.max(theta, axis = 1)
                result = result.astype(dtp)

                if n_unit == 0:
                    result.to_csv(args.output, sep='\t', mode='w', float_format="%.5f", index=False, header=True, compression={"method":"gzip"})
                else:
                    result.to_csv(args.output, sep='\t', mode='a', float_format="%.5f", index=False, header=False, compression={"method":"gzip"})

                n_unit += result.shape[0]
                logging.info(f"Lane {l}, sliding offset {offs_x}, {offs_y}, {n_unit} units so far.")
                offs_y += 1
            offs_y = 0
            offs_x += 1

    ed = df_full[mj].iloc[-1]
    df_full = df_full.loc[df_full['#lane'].eq(l1) &\
                          (df_full[mj] > ed - ovlp_buffer * args.mu_scale), :]
    logging.info(f"Left over size {df_full.shape[0]}.")


if args.outpost == '':
    args.outpost = args.output.replace(".tsv.gz", ".posterior.count.tsv.gz")
pd.concat([pd.DataFrame({'gene': feature_kept}),\
           pd.DataFrame(post_count.T, dtype='float64',\
                        columns = [str(k) for k in range(K)])],\
        axis = 1).to_csv(args.outpost, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})
