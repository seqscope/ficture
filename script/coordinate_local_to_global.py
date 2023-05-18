from collections import defaultdict
import sys, os, gzip, gc, argparse, warnings, logging
import subprocess as sp
import numpy as np
import pandas as pd
import sklearn.mixture

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *
from layout_fn import layout_map

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--outpref', type=str, help='')
parser.add_argument('--record_range', type=str, default='', help='')
parser.add_argument("--meta_data", type=str, help="Per tile meta data menifest.tsv")
parser.add_argument("--layout", type=str, help="Layout file of tiles to draw [lane] [tile] [row] [col] format in each line")

parser.add_argument('--lanes', nargs='*', type=str, help='One or multiple lanes to work on', default=[])
parser.add_argument('--region', nargs='*', type=str, help="In the form of \"lane1:tile_st1-tile_ed1 lane2:tile_st2-tile_ed2 ... \"", default=[])

parser.add_argument('--usecols', nargs='*', type=int, help="Specify which columns correspond to lane, tile, X, Y (in order) in the input file if it does not contain proper header", default=[])

parser.add_argument('--single_lane', action="store_true", help='Temporary - to be compatible with single lane coordinate')

args = parser.parse_args()

if len(args.lanes) == 0 and len(args.region) == 0:
    sys.exit("At least one of --lanes and --region is required")
if len(args.usecols) != 0 and len(args.usecols) != 4:
    sys.exit("Invalid --usecols")

logging.basicConfig(level= getattr(logging, "INFO", None))

### Parse input region
lane_list = args.lanes
kept_list = defaultdict(list)
for v in args.region:
    w = v.split(':')
    lane_list.append(w[0])
    if len(w) == 1:
        continue
    if len(w) != 2:
        sys.exit("Invalid regions in --region")
    u = [x for x in w[1].split('-') if x != '']
    if len(u) == 0 or len(u) > 2:
        sys.exit("Invalid regions in --region")
    if len(u) == 2:
        u = [str(x) for x in range(int(u[0]), int(u[1])+1)]
    kept_list[w[0]] += u
lane_list = list(set(lane_list))
for k,v in kept_list.items():
    kept_list[k] = sorted(list(set(kept_list[k])))
print(lane_list)
print(kept_list)
kept_layout_index = set([ (k, x) for k,v in kept_list.items() for x in v ] )

if args.single_lane and len(lane_list) > 1:
    sys.exit("Cannot use --single_lane with multiple lanes")

### Layout
lane = lane_list[0] if args.single_lane else None
layout, lanes, tiles = layout_map(args.meta_data, args.layout, lane)
xr = layout.xbin.max()
yr = layout.ybin.max()
logging.info(f"Read meta data. {xr}, {yr}")
nrows = int(layout.row.max() + 1)
ncols = int(layout.col.max() + 1)
# Code the output as the tile numbers of the lower-left and upper-right corners
tile_ll = tiles[-1][0]
tile_ur = tiles[0][-1]
logging.info(f"Read layout info.")

nbatch = 0
adt = {'X':int, 'Y':int, '#lane':str, 'tile':str}

nskip = 0
with gzip.open(args.outpref+".tsv.gz", 'wt') as wf:
    with gzip.open(args.input, 'rt') as rf:
        for line in rf:
            if line[0] == '#':
                # Transfer header to the output
                wf.write(line)
                nskip += 1
            else:
                break

reader = pd.read_csv(args.input, sep='\t', chunksize=500000, dtype=adt)
if len(args.usecols) == 4:
    with gzip.open(args.input, 'rt') as rf:
        header = rf.readline().strip().split('\t')
    ncol = len(header)
    header = ['V'+str(i) for i in range(ncol)]
    named_header = ['#lane', 'tile', 'X', 'Y']
    for i,x in enumerate(args.usecols):
        header[x] = named_header[i]
    reader = pd.read_csv(args.input, sep='\t', chunksize=500000, dtype=adt, skiprows=nskip, names=header)

xmin = np.inf
ymin = np.inf
xmax = -1
ymax = -1
for chunk in reader:
    tile_id = list(zip(chunk['#lane'].values, chunk['tile'].values) )
    kept_rowidx = [i for i,x in enumerate(tile_id) if x in kept_layout_index]
    chunk = chunk.iloc[kept_rowidx, :]
    if chunk.shape[0] == 0:
        continue
    tile_id = list(zip(chunk['#lane'].values, chunk['tile'].values) )

    bxmin = ((nrows - layout.loc[tile_id, 'row'].values - 1) * xr)
    bxmax = ((nrows - layout.loc[tile_id, 'row'].values - 1) * xr + layout.loc[tile_id, 'xbin'] ).max()
    bymin = (layout.loc[tile_id, 'col'].values * yr)
    bymax = (layout.loc[tile_id, 'col'].values * yr + layout.loc[tile_id, 'ybin']).max()

    chunk['X'] = bxmin + chunk.X.values - layout.loc[tile_id, 'xmin'].values
    chunk['Y'] = bymin + chunk.Y.values - layout.loc[tile_id, 'ymin'].values

    bxmin = bxmin.min()
    bymin = bymin.min()
    xmin = min(xmin, bxmin)
    xmax = max(xmax, bxmax)
    ymin = min(ymin, bymin)
    ymax = max(ymax, bymax)
    print(f"{xmin}, {xmax}, {ymin}, {ymax}")

    chunk.to_csv(args.outpref+".tsv.gz", sep='\t', index=False, header=False, mode='a')
    nbatch += 1

log = f"xmin\t{xmin}\nxmax\t{xmax}\nymin\t{ymin}\nymax\t{ymax}\n"
if os.path.exists(os.path.dirname(args.record_range)):
    with open(args.record_range, 'w') as wf:
        _=wf.write(log)
print(log)
