from collections import defaultdict
import sys, os, gzip, gc, argparse, warnings, logging
import subprocess as sp
import numpy as np
import pandas as pd
import sklearn.neighbors
import sklearn.mixture

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *
from layout_fn import layout_map
from filter_fn import filter_by_density_mixture

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help='')

parser.add_argument("--meta_data", type=str, help="Per tile meta data menifest.tsv")
parser.add_argument("--layout", type=str, help="Layout file of tiles to draw [lane] [tile] [row] [col] format in each line")

parser.add_argument('--lanes', nargs='*', type=str, help='One or multiple lanes to work on', default=[])
parser.add_argument('--region', nargs='*', type=str, help="In the form of \"lane1:tile_st1-tile_ed1 lane2:tile_st2-tile_ed2 ... \"", default=[])
args = parser.parse_args()

if len(args.lanes) == 0 and len(args.region) == 0:
    sys.exit("At least one of --lanes and --region is required")

logging.basicConfig(level= getattr(logging, "INFO", None))

### Parse input region
lane_list = args.lanes
kept_list = defaultdict(list)
for v in args.region:
    w = v.split(':')
    if len(w) == 1:
        lane_list.append(w[0])
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
print(kept_list)
kept_layout_index = set([ (k, x) for k,v in kept_list.items() for x in v ] )

### Layout
layout, lanes, tiles = layout_map(args.meta_data, args.layout)
xr = layout.xbin.max()
yr = layout.ybin.max()
logging.info(f"Read meta data. {xr}, {yr}")
nrows = int(layout.row.max() + 1)
ncols = int(layout.col.max() + 1)
# Code the output as the tile numbers of the lower-left and upper-right corners
tile_ll = tiles[-1][0]
tile_ur = tiles[0][-1]
logging.info(f"Read layout info.")


### Detect tissue region
nbatch = 0
adt = {'X':int, 'Y':int, '#lane':str, 'tile':str}
for chunk in pd.read_csv(args.input, sep='\t', chunksize=1000000, dtype=adt):
    tile_id = list(zip(chunk['#lane'].values, chunk['tile'].values) )
    kept_rowidx = [i for i,x in enumerate(tile_id) if x in kept_layout_index]
    chunk = chunk.iloc[kept_rowidx, :]
    if chunk.shape[0] == 0:
        continue
    tile_id = list(zip(chunk['#lane'].values, chunk['tile'].values) )
    chunk['X'] = (nrows - layout.loc[tile_id, 'row'].values - 1) * xr + chunk.X.values - layout.loc[tile_id, 'xmin'].values
    chunk['Y'] = layout.loc[tile_id, 'col'].values * yr + chunk.Y.values - layout.loc[tile_id, 'ymin'].values
    if nbatch == 0:
        chunk.to_csv(args.output, sep='\t', index=False, header=True)
    else:
        chunk.to_csv(args.output, sep='\t', index=False, header=False, mode='a')
    nbatch += 1
