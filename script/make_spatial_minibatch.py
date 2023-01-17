### Stream in pixel level data, assign minibatch label
### global-randomize-local-contiguous
### Output has the same columns as input with an extra column (1st) being the minibatch index

import sys, io, os, argparse, gzip, logging, glob, copy, re, time, importlib, warnings, pickle
import subprocess as sp
import numpy as np
import pandas as pd
import random as rng

parser = argparse.ArgumentParser()

# Innput and output info
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help='')

# Basic parameter
parser.add_argument('--batch_size', type=float, default=500, help='Length of the side (um) of square minibatches')
parser.add_argument('--batch_buff', type=float, default=50, help='Overlap between minibatches')
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--min_pixel', type=int, default=10, help='Just to avoid non-tissue regions with isolated pixels')
parser.add_argument('--seed', type=float, default=-1, help='')

# Specify a region to work on
parser.add_argument('--regions', type=str, default="", help='A file containing region intervals, same as that used by tabix -R (e.g. tab delimited, lane start end)')
parser.add_argument('--region', nargs='*', type=str, default=[], help='lane:Ystart-Yend (Y axis in barcode coordinate unit), separate by space if multiple regions')
parser.add_argument('--region_um', nargs='*', type=str, default=[], help='lane:Ystart-Yend (Y axis in um), separate by space if multiple regions')

args = parser.parse_args()
logging.basicConfig(level= getattr(logging, "INFO", None))

if not os.path.exists(args.input):
    sys.exit("ERROR: cannot find input file")


r_seed = args.seed
if r_seed <= 0:
    r_seed = time.time()
rng.seed(r_seed)
logging.info(f"Random seed {r_seed}")

### Basic parameterse
batch_size = int(args.batch_size * args.mu_scale)
batch_buff = int(args.batch_buff * args.mu_scale)
batch_step = batch_size - batch_buff

### Input pixel info (input has to have correct header)
with gzip.open(args.input, 'rt') as rf:
    input_header=rf.readline().strip().split('\t')
dty = {x:int for x in ['X', 'Y']}
dty.update({x:str for x in ['#lane', 'tile', 'gene']})

### Streaming input (Is this safe?)
cmd = []
if os.path.exists(args.regions):
    cmd = ["tabix", args.input, "-R", args.regions]
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

### Group pixels into minibatches
output_header = copy.copy(input_header)
output_header.insert(1, "random_index")
with open(args.output,'w') as wf:
    _=wf.write('\t'.join(output_header)+'\n')

df = pd.DataFrame()
for chunk in pd.read_csv(process.stdout,sep='\t',chunksize=1000000,\
                names=input_header, dtype=dty):
    if chunk.shape[0] == 0:
        logging.info(f"Empty? Left over size {df.shape[0]}.")
        continue
    chunk["random_index"] = -1
    chunk.random_index = chunk.random_index.astype(int)
    df = pd.concat([df, chunk])
    df['j'] = df.X.astype(str) + '_' + df.Y.astype(str)
    df["random_index"] = -1
    l_list = df['#lane'].unique()

    for l in l_list:
        lane_indx = df['#lane'].eq(l)
        x_min = df.loc[lane_indx, "X"].min()
        x_max = df.loc[lane_indx, "X"].max()
        y_min = df.loc[lane_indx, "Y"].min()
        y_max = df.loc[lane_indx, "Y"].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range < batch_size or y_range < batch_size:
            continue

        logging.info(f"Read blocks of pixels from lane {l}: {x_range/args.mu_scale:.2f} x {y_range/args.mu_scale:.2f}")
        x_grd_st = np.arange(x_min, x_max-batch_size/2+1, batch_step)
        x_grd_ed = [x + batch_size for x in x_grd_st]
        y_grd_st = np.arange(y_min, y_max-batch_size/2+1, batch_step)
        y_grd_ed = [x + batch_size for x in y_grd_st]
        x_grd_st[0]  = x_min - 1
        y_grd_st[0]  = y_min - 1
        x_grd_ed[-1] = x_max
        y_grd_ed[-1] = y_max

        if len(l_list) > 1 and l == l_list[0]:
            # This is the last bit of the current lane, including everything
            x_grd_ed[-1] = x_max
            y_grd_ed[-1] = y_max

        for it_x in range(len(x_grd_st)):
            for it_y in range(len(y_grd_st)):
                xst = x_grd_st[it_x]
                xed = x_grd_ed[it_x]
                yst = y_grd_st[it_y]
                yed = y_grd_ed[it_y]

                indx = (df.X > xst) & (df.X <= xed) & (df.Y > yst) & (df.Y <= yed) & df['#lane'].eq(l)

                if sum(indx) < args.min_pixel:
                    continue
                df.loc[indx, "random_index"] = rng.randint(1, sys.maxsize//100)
                ### Output
                df.loc[indx, output_header].to_csv(args.output, mode='a', sep='\t', index=False, header=False)

    ### Leftover
    y_max = df.loc[df["#lane"].eq(l_list[-1]), "Y"].max()
    left_indx = (df.random_index < 0) | (df["#lane"].eq(l_list[-1]) & (df.Y > y_max - batch_buff))
    df = df.loc[left_indx, :]
    logging.info(f"Left over size {df.shape[0]}")
