import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
from random import shuffle
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from scipy.sparse import *
import sklearn.neighbors
import sklearn.preprocessing


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help='Output prefix')
parser.add_argument('--sub_output', type=str, default=None, help='Output prefix for single factor maps')
parser.add_argument('--color_table', type=str, help='Pre-defined color map')

parser.add_argument('--scale', type=float, default=-1, help="")
parser.add_argument('--origin', type=int, default=[0,0], help="{0, 1} x {0, 1}, specify how to orient the image w.r.t. the coordinates. (0, 0) means the lower left corner has the minimum x-value and the minimum y-value; (0, 1) means the lower left corner has the minimum x-value and the maximum y-value;")
parser.add_argument('--category_column', type=str, default='', help='')
parser.add_argument('--color_table_category_name', type=str, default='Name', help='When --category_column is provided, which column to use as the category name')
parser.add_argument('--binary_cmap_name', type=str, default="plasma", help="Name of Matplotlib colormap to use for ploting individual factors")

parser.add_argument("--plot_fit", action='store_true', help="")
parser.add_argument('--xmin', type=float, default=-np.inf, help="")
parser.add_argument('--ymin', type=float, default=-np.inf, help="")
parser.add_argument('--xmax', type=float, default=np.inf, help="")
parser.add_argument('--ymax', type=float, default=np.inf, help="")
parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Size of the output pixels in um")

parser.add_argument("--tif", action='store_true', help="Store as 16-bit tif instead of png")
parser.add_argument("--skip_full_plot", action='store_true', help="")
parser.add_argument("--plot_individual_factor", action='store_true', help="")
parser.add_argument("--debug", action='store_true', help="")

args = parser.parse_args()
logging.basicConfig(level= getattr(logging, "INFO", None))
dt = np.uint16 if args.tif else np.uint8
nbit = 16 if args.tif else 8
nbit = 2**nbit - 1
kcol = args.category_column
ccol = args.color_table_category_name

# Dangerous way to detect which columns to use as factor loadings
with gzip.open(args.input, "rt") as rf:
    header = rf.readline().strip().split('\t')
recolumn = {'X':'x', 'Y':'y'}
for i,x in enumerate(header):
    if x in recolumn:
        header[i] = recolumn[x]
if kcol not in header:
    sys.exit(f"ERROR: {args.category_column} not found in header")
if not os.path.exists(args.color_table):
    sys.exit(f"ERROR: --color_table is required for categorical input")
color_info = pd.read_csv(args.color_table, sep='\t', header=0)
color_info[ccol] = color_info[ccol].astype(str)
color_idx = {x:i for i,x in enumerate(color_info[ccol])}
cmtx = np.array(color_info.loc[:, ["R","G","B"]])
cmtx = np.clip(np.around(cmtx * nbit),0,nbit).astype(dt)
K = len(color_idx)
logging.info(f"Read {K} colors")

# Read data
adt={x:float for x in ["x", "y"]}
adt[kcol] = str
df = pd.DataFrame()
chunksize=int(1e5) if args.debug else int(1e6)
for chunk in pd.read_csv(gzip.open(args.input, 'rt'), sep='\t', \
    chunksize=chunksize, skiprows=1, names=header, dtype=adt):
    if args.scale > 0:
        chunk.x = chunk.x / args.scale
        chunk.y = chunk.y / args.scale
    chunk = chunk[(chunk.y > args.ymin) & (chunk.y < args.ymax)]
    chunk = chunk[(chunk.x > args.xmin) & (chunk.x < args.xmax)]
    chunk['x_indx'] = np.round(chunk.x.values / args.plot_um_per_pixel, 0).astype(int)
    chunk['y_indx'] = np.round(chunk.y.values / args.plot_um_per_pixel, 0).astype(int)
    chunk = chunk.groupby(by = ['x_indx', 'y_indx'])[kcol].agg(lambda x: pd.Series.mode(x)[0]).to_frame().reset_index()
    df = pd.concat([df, chunk])
    logging.info(f"Read (and collapsed) {df.shape[0]} pixels")
    if args.debug:
        print(df[:2])
        break

df = df.groupby(by = ['x_indx', 'y_indx'])[kcol].agg(lambda x: pd.Series.mode(x)[0]).to_frame().reset_index()
df['K'] = df[kcol].map(color_idx)
x_indx_min = int(args.xmin / args.plot_um_per_pixel )
y_indx_min = int(args.ymin / args.plot_um_per_pixel )
x_indx_max = int(args.xmax / args.plot_um_per_pixel )
y_indx_max = int(args.ymax / args.plot_um_per_pixel )
if args.plot_fit or np.isinf(args.xmin):
    df.x_indx = df.x_indx - np.max([x_indx_min, df.x_indx.min()])
else:
    df.x_indx -= x_indx_min
if args.plot_fit or np.isinf(args.ymin):
    df.y_indx = df.y_indx - np.max([y_indx_min, df.y_indx.min()])
else:
    df.y_indx -= y_indx_min
if args.plot_fit or np.isinf(args.xmax):
    x_indx_max = df.x_indx.max()
else:
    x_indx_max -= x_indx_min
if args.plot_fit or np.isinf(args.ymax):
    y_indx_max = df.y_indx.max()
else:
    y_indx_max -= y_indx_min

N0 = df.shape[0]
df.index = range(N0)

hsize = x_indx_max + 1
wsize = y_indx_max + 1
logging.info(f"Read region {N0} pixels, image size {hsize} x {wsize}")


# Note: PIL default origin is upper-left
df.x_indx = np.clip(hsize - df.x_indx.values, 0, hsize-1)
df.y_indx = np.clip(df.y_indx.values, 0, wsize-1)
if args.origin[0] > 0:
    df.x_indx = hsize - 1 - df.x_indx.values
if args.origin[1] > 0:
    df.y_indx = wsize - 1 - df.y_indx.values

logging.info(f"Start constructing RGB image")

if not args.skip_full_plot:
    out_path = os.path.dirname(args.output)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img = np.zeros( (hsize, wsize, 3), dtype=dt)
    for r in range(3):
        img[:, :, r] = coo_array((cmtx[:, r][df.K.values], \
            (df.x_indx, df.y_indx)), shape=(hsize, wsize), dtype = dt).toarray()
    if args.tif:
        img = Image.fromarray(img, mode="I;16")
    else:
        img = Image.fromarray(img)
    outf = args.output
    outf += ".tif" if args.tif else ".png"
    img.save(outf)
    logging.info(f"Made full color categorical image\n{outf}")

if args.plot_individual_factor:
    pref = args.sub_output
    if args.sub_output is None:
        pref = args.output
    out_path = os.path.dirname(pref)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if args.binary_cmap_name not in plt.colormaps():
        args.binary_cmap_name = "plasma"
    for k in range(K):
        indx = df.index[df.K.eq(k)]
        img = np.zeros( (hsize, wsize, 3), dtype=dt)
        for r in range(3):
            img[:, :, r] = coo_array((cmtx[:, r][indx], \
                (df.loc[indx, 'x_indx'], df.loc[indx, 'y_indx'])),\
                shape=(hsize, wsize), dtype = dt).toarray()
        if args.tif:
            img = Image.fromarray(img, mode="I;16")
        else:
            img = Image.fromarray(img)
        outf = pref + ".F_"+str(k)
        outf += ".tif" if args.tif else ".png"
        img.save(outf)
        logging.info(f"Made factor specific image - {k}\n{outf}")
        if args.debug:
            break
