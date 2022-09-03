import sys, os, gzip, copy, gc, argparse
import numpy as np
import pandas as pd
from scipy.sparse import *
from io import StringIO
import subprocess as sp
import random as rng
from datetime import datetime

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--index_code', type=str, help='')

parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
parser.add_argument('--precision', type=int, default=1, help='Number of digits to store spatial location (in um), 0 for integer.')

parser.add_argument('--n_move', type=int, default=3, help='')
parser.add_argument('--hex_width', type=int, default=24, help='')
parser.add_argument('--hex_radius', type=int, default=-1, help='')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
args = parser.parse_args()

rng.seed(datetime.now().timestamp())

mu_scale = 1./args.mu_scale
radius=args.hex_radius
diam=args.hex_width
n_move = args.n_move
if n_move > diam // 2:
    n_move = diam // 4

if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = int(radius*np.sqrt(3))
if not os.path.exists(args.input):
    print(f"ERROR: cannot find input file \n {args.input}")
    sys.exit()

with open(args.index_code, 'r') as rf:
    tile_list = [x.strip() for x in rf.readlines()]

with gzip.open(args.input, 'rt') as rf:
    input_header=rf.readline().strip().split('\t')

print(input_header)
ct_header = ["gn","gt", "spl","unspl","ambig"]
output_header = copy.copy(input_header)
output_header.insert(1, "random_index")
with open(args.output,'w') as wf:
    _=wf.write('\t'.join(output_header)+'\n')

adt = {x:np.sum for x in ct_header}
n_unit = 0
for tile in tile_list:
    cmd = "tabix "+args.input+" " + tile
    df = pd.read_csv(StringIO(sp.check_output(cmd, stderr=sp.STDOUT, shell=True).decode('utf-8')), sep='\t', names=input_header)
    df = df[df[args.key] > 0]
    df['j'] = df.X.astype(str) + '_' + df.Y.astype(str)
    if df.shape[0] < args.min_ct_per_unit:
        continue
    brc = df.groupby(by = ['j','X','Y']).agg({args.key: sum}).reset_index()
    if brc.shape[0] < args.min_ct_per_unit:
        continue
    brc.index = range(brc.shape[0])
    brc['X'] = brc.X.astype(float).values * mu_scale
    brc['Y'] = brc.Y.astype(float).values * mu_scale
    pts = np.asarray(brc[['X','Y']])
    print(f"Read data in {tile} with {brc.shape[0]} pixels.")
    df.drop(columns = ['X', 'Y'], inplace=True)
    brc["hex_id"] = ""
    brc["random_index"] = 0
    offs_x = 0
    offs_y = 0
    while offs_x < n_move:
        while offs_y < n_move:
            x,y = pixel_to_hex(pts, radius, offs_x/n_move, offs_y/n_move)
            hex_crd = list(zip(x,y))
            ct = pd.DataFrame({'hex_id':hex_crd, 'tot':brc[args.key].values}).groupby(by = 'hex_id').agg({'tot': sum}).reset_index()
            mid_ct = np.median(ct.loc[ct.tot >= args.min_ct_per_unit, 'tot'].values)
            ct = set(ct.loc[ct.tot >= args.min_ct_per_unit, 'hex_id'].values)
            if len(ct) < 2:
                offs_y += 1
                continue
            hex_list = list(ct)
            hex_dict = {x:str(rng.randint(1, sys.maxsize)) for i,x in enumerate(hex_list)}
            brc["hex_id"] = hex_crd
            brc["random_index"] = brc.hex_id.map(hex_dict)
            sub = copy.copy(brc[brc.hex_id.isin(ct)] )
            cnt = sub.groupby(by = 'random_index').agg({'X':np.mean,'Y':np.mean}).reset_index()
            sub = sub.loc[:,['j','X','Y','random_index']].merge(right = df, on='j', how = 'inner')
            sub = sub.groupby(by = ['#lane','random_index','gene','gene_id']).agg(adt).reset_index()
            sub = sub.merge(right = cnt, on = 'random_index', how = 'inner')

            sub['X'] = [f"{x:.{args.precision}f}" for x in sub.X.values]
            sub['Y'] = [f"{x:.{args.precision}f}" for x in sub.Y.values]
            sub = sub.astype({x:int for x in ct_header})
            sub['tile'] = tile
            sub.loc[:, output_header].to_csv(args.output, mode='a', sep='\t', index=False, header=False)
            n_unit += len(ct)
            print(f"Sliding offset {offs_x}, {offs_y}. {n_unit} units so far.")
            offs_y += 1
        offs_y = 0
        offs_x += 1
