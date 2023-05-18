### Aggregate pixels by input region
### Temporarily support only rectangulars

# Assume coordinates in the input file are in barcode coordinate unit

import sys, os, gzip, copy, gc, time, argparse, logging
import numpy as np
import pandas as pd
from scipy.sparse import *
import subprocess as sp
import random as rng
from dataclasses import dataclass

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output_path', type=str, help='')

parser.add_argument('--region_list', type=str, default='', help='A tab delimited file containing region_id, lane, x_st-x_ed, y_st-y_ed. write a placeholder "." if no constraint on one axis')
parser.add_argument('--region', nargs='*', type=str, default=[], help='region_id,lane,x_st-x_ed,y_st-y_ed. write a placeholder "." if no constraint on one axis. separate by space for multiple regions')

parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--region_in_um', action="store_true", help="If the input region is in um (instead of barcode coordinate unit)")

parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')

parser.add_argument('--transfer_gene_prefix', action="store_true", help="")


args = parser.parse_args()

if not os.path.exists(args.input):
    sys.exit(f"ERROR: cannot find input file \n {args.input}")
if not os.path.exists(args.region_list) and len(args.region) == 0:
    sys.exit("Either --region_list or --region should be specified")
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

@dataclass
class Rectangle:
    lane: str
    x_st: float = -1
    x_ed: float = np.inf
    y_st: float = -1
    y_ed: float = np.inf

def assign_to_region(df, region_list):
    df['region'] = None
    for k,v in region_list.items():
        df.loc[(df['#lane'] == v.lane) & (df['X'] >= v.x_st) & (df['X'] <= v.x_ed) & (df['Y'] >= v.y_st) & (df['Y'] <= v.y_ed), 'region'] = k

### Parse region list
region_list = {} # region_id: [lane, x_st, x_ed, y_st, y_ed]
if os.path.exists(args.region_list):
    with open(args.region_list, 'r') as rf:
        for line in rf:
            wd = line.strip().split('\t')
            region_list[wd[0]] = wd[1:]
else:
    for region in args.region:
        wd = region.strip().split(',')
        region_list[wd[0]] = wd[1:]

for k,v in region_list.items():
    lane = v[0]
    x_st, x_ed = -1, np.inf
    if v[1] != '.':
        x_st, x_ed = [float(x) for x in v[1].split('-')]
    y_st, y_ed = -1, np.inf
    if v[2] != '.':
        y_st, y_ed = [float(x) for x in v[2].split('-')]
    if not args.region_in_um:
        x_st, x_ed, y_st, y_ed = [np.clip(x/args.mu_scale, -1, np.inf) for x in [x_st, x_ed, y_st, y_ed]]
    region_list[k] = Rectangle(lane, x_st, x_ed, y_st, y_ed)
    print(k, region_list[k])

adt = {x:int for x in ['X','Y', args.key]}
adt.update({x:str for x in ['#lane', 'gene', 'gene_id']})
df = pd.DataFrame()
for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=500000, header=0, usecols=['#lane','X','Y','gene','gene_id',args.key], dtype=adt):
    chunk['X'] = chunk.X / args.mu_scale
    chunk['Y'] = chunk.Y / args.mu_scale
    assign_to_region(chunk, region_list)
    chunk = chunk[chunk['region'].notnull()]
    if len(chunk) == 0:
        continue
    chunk = chunk.groupby(by = ['region', 'gene', 'gene_id']).agg({args.key:sum}).reset_index()
    df = pd.concat([df, chunk], axis=0, ignore_index=True)

print(df.groupby(by='region').agg({args.key:sum}).reset_index().sort_values(by=args.key, ascending=False))

# Write to MatrixMarket and Seurat compatible format

feature = df.groupby(by = ['gene', 'gene_id']).agg({args.key:sum}).reset_index()
feature.sort_values(by=args.key, ascending=False, inplace=True)
feature.drop_duplicates(subset='gene', inplace=True)
if args.transfer_gene_prefix:
    prefix = feature.gene.map(lambda x: x.split('_')[0] + '_' if '_' in x else '').values
    feature.gene_id = prefix + feature.gene_id.values
feature['dummy'] = "Gene Expression"
f = args.output_path + "/features.tsv.gz"
feature[['gene_id','gene','dummy']].to_csv(f, sep='\t', index=False, header=False)
ft_dict = {x:i for i,x in enumerate(feature.gene.values)}
M = len(ft_dict)

brc_f = args.output_path + "/barcodes.tsv.gz"
mtx_f = args.output_path + "/matrix.mtx.gz"
# If exists, delete
if os.path.exists(brc_f):
    _ = os.system("rm " + brc_f)
if os.path.exists(mtx_f):
    _ = os.system("rm " + mtx_f)

brc = list(df.region.unique() )
bc_dict = {x:i for i,x in enumerate(brc)}
N = len(bc_dict)
with gzip.open(brc_f, 'wt') as wf:
    _ = wf.write('\n'.join(brc) + '\n' )

df["i"] = df.region.map(bc_dict)
df["j"] = df.gene.map(ft_dict)
with gzip.open(mtx_f, 'wt') as wf:
    _ = wf.write("%%MatrixMarket matrix coordinate integer general\n%\n")
    _ = wf.write(f"{M} {N} {df.shape[0]}\n")
    for i, j, x in df[['j', 'i', args.key]].values:
        _ = wf.write(f"{i+1} {j+1} {x}\n"   )
