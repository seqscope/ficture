import sys, os, gzip, copy, gc, time, argparse, logging
import numpy as np
import pandas as pd
from scipy.sparse import *
import subprocess as sp
import random as rng
from collections import defaultdict
import geojson
import shapely.prepared, shapely.geometry
from shapely.geometry import Point

from ficture.utils.hexagon_fn import pixel_to_hex, hex_to_pixel

def make_dge(_args):

    parser = argparse.ArgumentParser(prog = "make_dge")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--boundary', type=str, default = '', help='')

    parser.add_argument('--major_axis', type=str, default="Y", help='X or Y')
    parser.add_argument('--mu_scale', type=float, default=1, help='Coordinate to um translate')

    parser.add_argument('--count_header', nargs='*', type=str, default=["gn","gt", "spl","unspl","ambig"], help="Which columns correspond to UMI counts in the input")
    parser.add_argument('--group_within', nargs='*', type=str, default=[], help="Pixels are collapsed within each group. Separate by space if multiple identifier")

    parser.add_argument('--key', default = 'gt', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced. Otherwise depending on customized ct_header')
    parser.add_argument('--feature_id', default = 'gene', type=str)

    parser.add_argument('--precision', type=int, default=1, help='Number of digits to store spatial location (in um), 0 for integer.')

    parser.add_argument('--n_move', type=int, default=3, help='')
    parser.add_argument('--hex_width', type=int, default=24, help='')
    parser.add_argument('--hex_radius', type=int, default=-1, help='')
    parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
    parser.add_argument('--min_density_per_unit', type=float, default=0.2, help='')

    args = parser.parse_args(_args)
    if len(_args) == 0:
        parser.print_help()
        return

    r_seed = time.time()
    rng.seed(r_seed)
    logging.basicConfig(level= getattr(logging, "INFO", None))
    logging.info(f"Random seed {r_seed}")
    mj = args.major_axis

    # Input file and numerical columns to use as counts
    ct_header = args.count_header
    key = args.key
    if key not in ct_header:
        ct_header = [key]
        logging.warning(f"The designated major key is not one of the specified count columns, --count_header is ignored")
    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: cannot find input file \n {args.input}")
    with gzip.open(args.input, 'rt') as rf:
        input_header=rf.readline().strip().split('\t')
        ct_header = [v for v in input_header if v in ct_header]
        if len(ct_header) == 0:
            sys.exit("Input header does not contain the specified --count_header")
    print(input_header)
    keep_header=['X','Y','j','Group'] + ct_header
    output_header = ["random_index",'X','Y','gene'] + ct_header + args.group_within
    with open(args.output,'w') as wf:
        _=wf.write('\t'.join(output_header)+'\n')

    # basic parameters
    random_index_max=sys.maxsize//100000
    random_index_length=int(np.log10(random_index_max) ) + 1
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
    area = radius * diam * 3 / 2
    min_ct_per_unit = max(args.min_ct_per_unit, args.min_density_per_unit * area)

    adt = {x:"sum" for x in ct_header}
    dty = {x:int for x in ct_header}
    dty.update({x:str for x in ['X','Y','gene'] + args.group_within})

    use_boundary = False
    if os.path.isfile(args.boundary):
        mpoly = shapely.geometry.shape(geojson.load(open(args.boundary, 'rb')))
        mpoly = shapely.prepared.prep(mpoly)
        use_boundary = True
        logging.info(f"Load boundary from {args.boundary}")

    n_unit = 0
    df_full = pd.DataFrame()
    last_batch = defaultdict(set)
    for chunk in pd.read_csv(args.input, sep='\t', chunksize=1000000, dtype=dty):
        if chunk.shape[0] == 0:
            logging.info(f"Empty? Left over size {df_full.shape[0]}.")
            continue
        chunk.rename(columns={args.feature_id:'gene'}, inplace=True)
        chunk['j'] = chunk.X + '_' + chunk.Y
        chunk.X = chunk.X.astype(float)
        chunk.Y = chunk.Y.astype(float)
        ed = chunk[mj].iloc[-1]
        chunk['Group'] = ''
        if len(args.group_within) > 0:
            chunk['Group'] = chunk[args.group_within].apply(lambda x: '_'.join(x), axis=1)
        df_full = pd.concat([df_full, chunk])

        if df_full[mj].iloc[-1] - df_full[mj].iloc[0] < ovlp_buffer * args.mu_scale:
            # This chunk is too narrow, leave to process together with neighbors
            r = int(df_full[mj].iloc[-1]*mu_scale)
            l = int(df_full[mj].iloc[0] *mu_scale)
            logging.info(f"Not enough pixels, left over size {df_full.shape[0]} ({l}, {r}).")
            continue

        left = copy.copy(df_full.loc[df_full[mj] > ed - ovlp_buffer * args.mu_scale, keep_header])

        for l in df_full.Group.unique():
            df = df_full.loc[df_full.Group.eq(l)]
            brc = df.groupby(by = ['j']).agg(adt).reset_index()
            brc = brc.merge(right = df[['j','X','Y']].drop_duplicates(subset='j'), on = 'j', how='inner')
            brc.index = range(brc.shape[0])
            brc['X'] *= mu_scale
            brc['Y'] *= mu_scale
            st = brc[mj].min()
            ed = brc[mj].max()
            pts = np.asarray(brc[['X','Y']])
            logging.info(f"Processing {brc.shape[0]} pixels ({len(df)} {st}, {ed}).")
            brc["hex_id"] = ""
            brc["random_index"] = 0
            offs_x = 0
            offs_y = 0
            while offs_x < n_move:
                while offs_y < n_move:
                    prefix  = str(offs_x)+str(offs_y)
                    x,y = pixel_to_hex(pts, radius, offs_x/n_move, offs_y/n_move)
                    hex_crd = list(zip(x,y))
                    ct = pd.DataFrame({'hex_id':hex_crd, 'tot':brc[key].values, 'X':pts[:, 0], 'Y':pts[:,1]}).groupby(by = 'hex_id').agg({'tot': "sum", 'X':"min", 'Y':"min"}).reset_index()
                    mid_ct = np.median(ct.loc[ct.tot >= min_ct_per_unit, 'tot'].values)
                    ct = set(ct.loc[(ct.tot >= min_ct_per_unit) & (ct[mj] > st + diam/2) & (ct[mj] < ed - diam/2), 'hex_id'].values)
                    ct = ct - last_batch[(offs_x,offs_y,l)]
                    if len(ct) < 2:
                        offs_y += 1
                        continue
                    last_batch[(offs_x,offs_y,l)] = ct
                    hex_list = list(ct)
                    suff = [str(x[0])[-1]+str(x[1])[-1] for x in hex_list]
                    hex_dict = {x: str(rng.randint(1, random_index_max)).zfill(random_index_length) + suff[i] for i,x in enumerate(hex_list)}
                    brc["hex_id"] = hex_crd
                    brc["random_index"] = brc.hex_id.map(hex_dict)
                    sub = copy.copy(brc[brc.hex_id.isin(ct)] )

                    cnt = sub[['hex_id', 'random_index']].drop_duplicates()
                    hx = cnt.hex_id.map(lambda x : x[0])
                    hy = cnt.hex_id.map(lambda x : x[1])
                    cnt['X'], cnt['Y'] = hex_to_pixel(hx, hy, radius, offs_x/n_move, offs_y/n_move)
                    if use_boundary:
                        kept = [mpoly.contains(Point(*p)) for p in cnt[['X','Y']].values]
                        logging.info(f"Keep {sum(kept)}/{len(cnt)} units within boundary.")
                        cnt = cnt.loc[kept, :]

                    sub = sub.loc[:,['j','random_index']].merge(right = df, on='j', how = 'inner')
                    sub = sub.groupby(by = ['random_index','gene']).agg(adt).reset_index()
                    sub = sub.merge(right = cnt, on = 'random_index', how = 'inner')
                    sub['X'] = [f"{x:.{args.precision}f}" for x in sub.X.values]
                    sub['Y'] = [f"{x:.{args.precision}f}" for x in sub.Y.values]
                    sub = sub.astype({x:int for x in ct_header})
                    # Add offset combination as prefix to random_index
                    sub.random_index = prefix + sub.random_index.values
                    sub.loc[:, output_header].to_csv(args.output, mode='a', sep='\t', index=False, header=False)
                    n_unit += len(ct)
                    logging.info(f"Sliding offset {offs_x}, {offs_y}. Add {len(ct)} units, median count {mid_ct}, {n_unit} units so far.")
                    offs_y += 1
                offs_y = 0
                offs_x += 1
        df_full = copy.copy(left)
        logging.info(f"Left over size {df_full.shape[0]} ({df_full[mj].iloc[0] * mu_scale :.0f}, {df_full[mj].iloc[-1] * mu_scale :.0f}).")

if __name__ == "__main__":
    make_dge(sys.argv[1:])
