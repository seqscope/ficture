import sys, os, copy, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd
import sklearn.neighbors
from random import shuffle

import matplotlib as mpl
import matplotlib.pyplot as plt
import png

from scipy.sparse import *

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *
from utilt import plot_colortable
from image_fn import ImgRowIterator as RowIterator

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help='Output prefix')
parser.add_argument('--fill_range', type=float, default=5, help="um")
parser.add_argument('--horizontal_axis', type=str, default="x", help="Which coordinate is horizontal, x or y")
parser.add_argument('--color_table', type=str, default='', help='Pre-defined color map')
parser.add_argument('--cmap_name', type=str, default="turbo", help="Name of Matplotlib colormap to use")
parser.add_argument('--binary_cmap_name', type=str, default="plasma", help="Name of Matplotlib colormap to use for ploting individual factors")
parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Size of the output pixels in um")
parser.add_argument('--xmin', type=float, default=-1, help="um")
parser.add_argument('--ymin', type=float, default=-1, help="um")
parser.add_argument('--xmax', type=float, default=np.inf, help="um")
parser.add_argument('--ymax', type=float, default=np.inf, help="um")
parser.add_argument("--plot_fit", action='store_true', help="")
parser.add_argument("--plot_discretized", action='store_true', help="")
parser.add_argument("--plot_individual_factor", action='store_true', help="")

args = parser.parse_args()
logging.basicConfig(level= getattr(logging, "INFO", None))
radius = args.fill_range/args.plot_um_per_pixel
dt = np.uint8
haxis = args.horizontal_axis.lower()
if haxis not in ["x", "y"]:
    sys.exit("Unrecognized --horizontal_axis")
logging.info(f"Range: ({args.xmin}, {args.xmax}) x ({args.ymin}, {args.ymax})")

# Dangerous way to detect which columns to use as factor loadings
with gzip.open(args.input, "rt") as rf:
    header = rf.readline().strip().split('\t')
# Temporary - to be compatible with older input files
recolumn = {'Hex_center_x':'x', 'Hex_center_y':'y', 'X':'x', 'Y':'y'}
for i,x in enumerate(header):
    if x in recolumn:
        header[i] = recolumn[x]
factor_header = []
for x in header:
    y = re.match('^[A-Za-z]+_\d+$', x)
    if y:
        factor_header.append(y.group(0))
K = len(factor_header)

# Colormap
if os.path.exists(args.color_table):
    color_info = pd.read_csv(args.color_table, sep='\t', header=0)
    cmtx = np.array(color_info.loc[:, ["R","G","B"]])
    logging.info(f"Read color map")
else:
    cmap_name = args.cmap_name
    if args.cmap_name not in plt.colormaps():
        cmap_name = "turbo"
    cmap = plt.get_cmap(cmap_name, K)
    cmtx = np.array([cmap(i) for i in range(K)] )
    indx = np.arange(K)
    shuffle(indx)
    cmtx = cmtx[indx, ]
    cmtx = cmtx[:, :3]
    cdict = {k:cmtx[k,:] for k in range(K)}
    # Plot color bar separately
    fig = plot_colortable(cdict, "Factor label", sort_colors=False, ncols=4)
    f = args.output + ".cbar"
    fig.savefig(f, format="png")
    logging.info(f"Set up color map for {K} factors")

# Read data
adt={x:float for x in ["x", "y"]+factor_header}
df = pd.DataFrame()
nc = 0
for chunk in pd.read_csv(gzip.open(args.input, 'rt'), sep='\t', \
    chunksize=1000000, skiprows=1, names=header, dtype=adt):
    print( ((chunk.y > args.ymin) & (chunk.y < args.ymax)).sum() )
    print(chunk.y.min(), chunk.y.max(), chunk.x.min(), chunk.x.max())
    chunk = chunk.loc[(chunk.y > args.ymin) & (chunk.y < args.ymax), :]
    chunk = chunk.loc[(chunk.x > args.xmin) & (chunk.x < args.xmax), :]
    chunk['x_indx'] = np.round(chunk.x.values / args.plot_um_per_pixel, 0).astype(int)
    chunk['y_indx'] = np.round(chunk.y.values / args.plot_um_per_pixel, 0).astype(int)
    chunk = chunk.groupby(by = ['x_indx', 'y_indx']).agg({ x:np.mean for x in factor_header }).reset_index()
    df = pd.concat([df, chunk])
    nr = df.shape[0]
    nc += 1
    logging.info(f"...reading file ({nc}, {nr})")

df = df.groupby(by = ['x_indx', 'y_indx']).agg({ x:np.mean for x in factor_header }).reset_index()
x_indx_min = int(args.xmin / args.plot_um_per_pixel )
y_indx_min = int(args.ymin / args.plot_um_per_pixel )
if args.plot_fit or args.xmin >= 0:
    df.x_indx = df.x_indx - np.max([x_indx_min, df.x_indx.min()])
if args.plot_fit or args.ymin >= 0:
    df.y_indx = df.y_indx - np.max([y_indx_min, df.y_indx.min()])

N0 = df.shape[0]
df.index = range(N0)
if haxis != "x":
    df.rename(columns = {"x_indx":"y_indx", "y_indx":"x_indx"}, inplace=True)
width, height = df[['x_indx','y_indx']].max(axis = 0) + 1
height_um = height * args.plot_um_per_pixel
width_um  = width  * args.plot_um_per_pixel
logging.info(f"Read region {N0} pixels in region {height_um} x {width_um}")

pts = np.array(df.loc[:, ["x_indx", "y_indx"]], dtype=int)
mtx = np.clip(np.around( np.array(df.loc[:,factor_header]) @\
              cmtx * 255),0,255).astype(dt)

logging.info(f"Got RGB matrix ({mtx.shape}) for {pts.shape[0]} datapoints.\nStart making fractional image...")
outf = args.output + ".png"
obj  = RowIterator(pts, mtx, radius)
wpng = png.Writer(size=(width, height),greyscale=False,bitdepth=8,planes=3)
with open(outf, 'wb') as f:
    wpng.write(f, obj)
logging.info(f"Made fractional image\n{outf}")


if args.plot_individual_factor:
    if args.binary_cmap_name not in plt.colormaps():
        args.binary_cmap_name = "plasma"
    for k in range(K):
        v = np.clip(df.loc[:, factor_header[k]].values,0,1)
        mtx = np.clip(mpl.colormaps[args.binary_cmap_name](v)[:,:3]*255,0,255).astype(dt)
        outf = args.output + ".F_"+str(k)+".png"
        obj  = RowIterator(pts, mtx, radius)
        wpng = png.Writer(size=(width, height),greyscale=False,bitdepth=8,planes=3)
        with open(outf, 'wb') as f:
            wpng.write(f, obj)
        logging.info(f"Made factor specific image - {k}\n{outf}")

if args.plot_discretized:
    mtx = coo_matrix( (np.ones(N0,dtype=dt),\
        (range(N0), np.array(df.loc[:, factor_header]).argmax(axis = 1))),\
        shape=(N0, K)).toarray()
    mtx = np.clip(np.around(mtx @ cmtx * 255),0,255).astype(dt)
    outf = args.output + ".top.png"
    obj  = RowIterator(pts, mtx, radius)
    wpng = png.Writer(size=(width, height),greyscale=False,bitdepth=8,planes=3)
    with open(outf, 'wb') as f:
        wpng.write(f, obj)
    logging.info(f"Made hard threshold image\n{outf}")
