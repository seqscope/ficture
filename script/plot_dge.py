# Generate (not super high resolution) dge plot
# Input: barcodes.tsv.gz

import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd

from PIL import Image

from scipy.sparse import coo_array

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *
from utilt import plot_colortable

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--outpref', type=str, help='Output prefix')
parser.add_argument("--tif", action='store_true', help="Store as 16-bit tif instead of png")
parser.add_argument("--transpose", action='store_true', help="If X is the vertical axis")
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--usecols', nargs='*', type=int, help="Specify which columns correspond to X, Y, comma delimited counts (in order) in the input file", default=[])
parser.add_argument('-r', nargs='*', type=int, help="Specify which count column is used for R in RGB channel", default=[])
parser.add_argument('-g', nargs='*', type=int, help="Specify which count column is used for G in RGB channel", default=[])
parser.add_argument('-b', nargs='*', type=int, help="Specify which count column is used for B in RGB channel", default=[])
parser.add_argument('--channel_scale_quantile', type=float, default=.75, help='')
parser.add_argument('--range', type=str, default='', help='')
parser.add_argument('--range_um', type=str, default='', help='')

# Parameters shared by all plotting scripts
parser.add_argument('--xmin', type=float, default=-1, help="")
parser.add_argument('--ymin', type=float, default=-1, help="")
parser.add_argument('--xmax', type=float, default=np.inf, help="")
parser.add_argument('--ymax', type=float, default=np.inf, help="")
parser.add_argument('--plot_um_per_pixel', type=int, default=1, help="Size of the output pixels in um")

args = parser.parse_args()
logging.basicConfig(level= getattr(logging, "INFO", None))
mu_scale = 1./args.mu_scale
max_c = 65535 if args.tif else 255
dt = np.uint16 if args.tif else np.uint8
rgb_list = [args.r, args.g, args.b]

rdic = {}
if os.path.exists(args.range):
    with open(args.range, 'r') as rf:
        for line in rf:
            line = line.strip().split('\t')
            rdic[line[0]] = float(line[1]) * mu_scale
elif os.path.exists(args.range_um):
    with open(args.range, 'r') as rf:
        for line in rf:
            line = line.strip().split('\t')
            rdic[line[0]] = float(line[1])
if args.xmin < 0 and 'xmin' in rdic:
    args.xmin = rdic['xmin']
if args.ymin < 0 and 'ymin' in rdic:
    args.ymin = rdic['ymin']
if args.xmax == np.inf and 'xmax' in rdic:
    args.xmax = rdic['xmax']
if args.ymax == np.inf and 'ymax' in rdic:
    args.ymax = rdic['ymax']

print(args.xmin, args.ymin, args.xmax, args.ymax)

# Read data
pixel = []
with gzip.open(args.input, 'rt') as rf:
    for line in rf:
        if line[0] == "#":
            continue
        line = line.strip().split('\t')
        X, Y, ct = [line[i] for i in args.usecols]
        ct = np.array([int(x) for x in ct.split(',')] )
        rgb = [0, 0, 0]
        for i, c in enumerate([args.r, args.g, args.b]):
            if len(c) > 0:
                rgb[i] = ct[c].sum()
        pixel.append([float(X), float(Y), rgb[0], rgb[1], rgb[2]])

pixel = pd.DataFrame(pixel, columns = ['x', 'y', 'r', 'g', 'b'])
pixel.x *= mu_scale
pixel.y *= mu_scale

xmin = pixel.x.min() if args.xmin < 0 else args.xmin
ymin = pixel.y.min() if args.ymin < 0 else args.ymin
xmax = pixel.x.max() if args.xmax == np.inf else args.xmax
ymax = pixel.y.max() if args.ymax == np.inf else args.ymax

pixel.x = ((pixel.x - xmin) / args.plot_um_per_pixel).astype(int)
pixel.y = ((pixel.y - ymin) / args.plot_um_per_pixel).astype(int)

xsize = int(np.ceil((xmax - xmin) / args.plot_um_per_pixel))
ysize = int(np.ceil((ymax - ymin) / args.plot_um_per_pixel))

pixel = pixel[(pixel.x >= 0) & (pixel.x < xsize) & (pixel.y >= 0) & (pixel.y < ysize)]

pixel = pixel.groupby(by = ["x", "y"]).agg({x:np.sum for x in ['r', 'g', 'b']}).reset_index()
N = pixel.shape[0]
logging.info(f"Collapsed into {N} nonzero pixels")

for i, c in enumerate(['r', 'g', 'b']):
    if len(rgb_list[i]) > 0:
        qt = int(N * args.channel_scale_quantile)
        s = pixel[c].iloc[np.argpartition(pixel[c], kth=qt)[qt]]
        s = s if s > 0 else 1
        pixel[c] = np.clip(pixel[c] / s * max_c, 0, max_c).astype(dt)
    else:
        pixel[c] = np.zeros(pixel.shape[0], dtype=dt)

img = np.zeros( (xsize, ysize, 3), dtype=dt)
for i, c in enumerate(['r', 'g', 'b']):
    img[:, :, i] = coo_array((pixel[c].values, \
                              (pixel.x.values, pixel.y.values)),\
                             shape=(xsize, ysize), dtype = dt).toarray()
if args.transpose:
    img = img.transpose(1, 0, 2)
if args.tif:
    img = Image.fromarray(img, mode="I;16")
else:
    img = Image.fromarray(img)

outf = args.outpref + ".xmin_"+str(int(xmin))+".ymin_"+str(int(ymin))+".res_"+str(args.plot_um_per_pixel)
outf += ".tif" if args.tif else ".png"
img.save(outf)
logging.info(f"Made hard threshold image\n{outf}")
