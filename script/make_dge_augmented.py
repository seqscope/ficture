### Pixel to hexagon, augmented with a small neighborhood
import sys, os, gzip, copy, gc, time, argparse, logging
import numpy as np
import pandas as pd
from scipy.sparse import *
import subprocess as sp
import random as rng
import sklearn.neighbors

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import pixel_to_hex, hex_to_pixel

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help='')

parser.add_argument('--major_axis', type=str, default="Y", help='X or Y')
parser.add_argument('--regions', type=str, default="", help='A file containing region intervals, same as that used by tabix -R (e.g. tab delimited, lane start end)')
parser.add_argument('--region', nargs='*', type=str, default=[], help='lane:Y_start-Y_end (Y axis in barcode coordinate unit), separate by space if multiple regions')
parser.add_argument('--region_um', nargs='*', type=str, default=[], help='lane:Y_start-Y_end (Y axis in um), separate by space if multiple regions')
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')

parser.add_argument('--count_header', nargs='*', type=str, default=["gn","gt", "spl","unspl","ambig"], help="Which columns correspond to UMI counts in the input")
parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced. Otherwise depending on customized ct_header')
parser.add_argument('--precision', type=int, default=1, help='Number of digits to store spatial location (in um), 0 for integer.')

parser.add_argument('--n_move', type=int, default=3, help='')
parser.add_argument('--hex_width', type=int, default=24, help='')
parser.add_argument('--hex_radius', type=int, default=-1, help='')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
parser.add_argument('--buffer_radius', type=float, default=-1, help='')
args = parser.parse_args()

r_seed = time.time()
rng.seed(r_seed)
logging.basicConfig(level= getattr(logging, "INFO", None))
logging.info(f"Random seed {r_seed}")
mj = args.major_axis

# Input file and numerical columns to use as counts
ct_header = args.count_header
key = args.key
if key not in ct_header:
    key = ct_header[0]
    logging.warning(f"The designated major key is not one of the specified count columns, --key is ignored the first existing key is chosen")
if not os.path.exists(args.input):
    sys.exit(f"ERROR: cannot find input file \n {args.input}")
with gzip.open(args.input, 'rt') as rf:
    input_header=rf.readline().strip().split('\t')
    ct_header = [v for v in input_header if v in ct_header]
    if len(ct_header) == 0:
        sys.exit("Input header does not contain the specified --count_header")
print(input_header)
ct_header_buff = [x+'_buffer' for x in ct_header]

# basic parameters
random_index_max=sys.maxsize//10000
random_index_length=int(np.log10(random_index_max) ) + 1
mu_scale = 1./args.mu_scale
radius=args.hex_radius
buffer_radius = args.buffer_radius
diam=args.hex_width
n_move = args.n_move
if n_move > diam // 2:
    n_move = diam // 4
if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = int(radius*np.sqrt(3))
if buffer_radius < 0:
    buffer_radius = radius * 2
ovlp_buffer = max(diam * 2, buffer_radius)

adt = {x:np.sum for x in ct_header}
bdt={}
bdt['tile'] = lambda x : int(np.median(x))
n_unit = 0
dty = {x:int for x in ['tile','X','Y']+ct_header}
dty.update({x:str for x in ['#lane', 'gene', 'gene_id']})

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

output_header = copy.copy(input_header)
output_header.insert(1, "random_index")
output_header += ct_header_buff
with open(args.output,'w') as wf:
    _=wf.write('\t'.join(output_header)+'\n')

chunksize = 500000
df_full = pd.DataFrame()
for chunk in pd.read_csv(process.stdout,sep='\t',chunksize=chunksize,\
                names=input_header, dtype=dty):
    if chunk.shape[0] == 0:
        logging.info(f"Empty? Left over size {df_full.shape[0]}.")
        continue
    # ed = chunk.Y.iloc[-1]
    ed = chunk[mj].iloc[-1]
    df_full = pd.concat([df_full, chunk])
    df_full['j'] = df_full.X.astype(str) + '_' + df_full.Y.astype(str)
    l1 = df_full['#lane'].iloc[-1]
    l_list = df_full['#lane'].unique()

    if len(l_list) == 1 and df_full[mj].iloc[-1] -\
        df_full[mj].iloc[0] < ovlp_buffer * args.mu_scale:
        # This chunk is too narrow, leave to process together with neighbors
        r = int(df_full[mj].iloc[-1]*args.mu_scale)
        l = int(df_full[mj].iloc[0] *args.mu_scale)
        logging.info(f"Left over size {df_full.shape[0]} ({l}, {r}).")
        continue

    left = copy.copy(df_full[df_full['#lane'].eq(l1) & (df_full[mj] > ed - ovlp_buffer * args.mu_scale)])

    for l in l_list:
        df = df_full.loc[df_full['#lane'].eq(l)]
        brc = df.groupby(by = ['j','tile','X','Y']).agg(adt).reset_index()
        brc.index = range(brc.shape[0])
        brc['X'] = brc.X.astype(float).values * mu_scale
        brc['Y'] = brc.Y.astype(float).values * mu_scale
        st = brc[mj].min()
        ed = brc[mj].max()
        pts = np.asarray(brc[['X','Y']])
        ref = sklearn.neighbors.BallTree(pts)
        logging.info(f"Read {brc.shape[0]} pixels.")
        brc["hex_id"] = ""
        brc["random_index"] = 0
        offs_x = 0
        offs_y = 0
        while offs_x < n_move:
            while offs_y < n_move:
                prefix  = str(offs_x)+str(offs_y)
                x,y = pixel_to_hex(pts, radius, offs_x/n_move, offs_y/n_move)
                hex_crd = list(zip(x,y))
                ct = pd.DataFrame({'hex_id':hex_crd, 'tot':brc[key].values, 'X':pts[:, 0], 'Y':pts[:,1]}).groupby(by = 'hex_id').agg({'tot': sum, 'X':np.min, 'Y':np.min}).reset_index()
                mid_ct = np.median(ct.loc[ct.tot >= args.min_ct_per_unit, 'tot'].values)
                ct = set(ct.loc[(ct.tot >= args.min_ct_per_unit) & (ct[mj] > st + diam/2) & (ct[mj] < ed - diam/2), 'hex_id'].values)
                if len(ct) < 2:
                    offs_y += 1
                    continue
                hex_list = list(ct)
                suff = [str(x[0])[-1]+str(x[1])[-1] for x in hex_list]
                hex_dict = {x: str(rng.randint(1, random_index_max)).zfill(random_index_length) + suff[i] for i,x in enumerate(hex_list)}
                brc["hex_id"] = hex_crd
                brc["random_index"] = brc.hex_id.map(hex_dict)

                cnt = brc[brc.hex_id.isin(ct)].groupby(by = ['hex_id', 'random_index']).agg(bdt).reset_index()
                hx = cnt.hex_id.map(lambda x : x[0])
                hy = cnt.hex_id.map(lambda x : x[1])
                cnt['X'], cnt['Y'] = hex_to_pixel(hx, hy, radius, offs_x/n_move, offs_y/n_move)

                indx = ref.query_radius(cnt[['X','Y']], r = buffer_radius)
                c_indx = [x for y in indx for x in y]
                r_indx = [x for i,x in enumerate(cnt.random_index) for y in range(len(indx[i])) ]
                buff = pd.DataFrame({'random_index':r_indx,
                                     'j':brc.loc[c_indx, 'j'].values})
                buff = buff.merge(right = df, on = 'j', how = 'inner')
                buff = buff.groupby(by=['random_index', 'gene', 'gene_id']).agg(adt).reset_index()
                buff.rename(columns = {x: f"{x}_buffer" for x in ct_header}, inplace=True)

                sub = brc.loc[brc.hex_id.isin(ct), ['j','random_index']].merge(right = df, on='j', how = 'inner')
                sub = sub.groupby(by = ['random_index','gene','gene_id']).agg(adt).reset_index()

                sub = sub.merge(right = buff, on = ['random_index','gene','gene_id'], how = 'outer')
                sub.fillna(0, inplace=True)
                for k in ct_header:
                    sub[k + '_buffer'] = sub[k + '_buffer'] - sub[k]
                    if sub[k + '_buffer'].min() < 0:
                        logging.warning(f"Negative buffer value for {k}.")

                sub = sub.merge(right = cnt, on = 'random_index', how = 'inner')
                sub['#lane'] = l
                sub['X'] = [f"{x:.{args.precision}f}" for x in sub.X.values]
                sub['Y'] = [f"{x:.{args.precision}f}" for x in sub.Y.values]
                sub = sub.astype({x:int for x in ct_header + ct_header_buff})
                # Add offset combination as prefix to random_index
                sub.random_index = prefix + sub.random_index.values
                sub.loc[:, output_header].to_csv(args.output, mode='a', sep='\t', index=False, header=False, float_format='%.3f')
                n_unit += len(ct)
                logging.info(f"Lane {l}, sliding offset {offs_x}, {offs_y}. Add {len(ct)} units, median count {mid_ct}, {n_unit} units so far.")
                offs_y += 1
            offs_y = 0
            offs_x += 1
    df_full = copy.copy(left)
    logging.info(f"Left over size {df_full.shape[0]}.")
