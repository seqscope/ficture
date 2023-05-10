import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd
from random import shuffle

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from scipy.sparse import *
from sklearn.preprocessing import normalize

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *
from utilt import plot_colortable

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help='Output prefix')
parser.add_argument("--plot_fit", action='store_true', help="")
parser.add_argument("--plot_top_k", type=int, default=-1, help="")
parser.add_argument("--plot_prob_cut", type=float, default=.99, help="")
parser.add_argument("--tif", action='store_true', help="Store as 16-bit tif instead of png")

# Parameters shared by all plotting scripts
parser.add_argument('--cmap_name', type=str, default="turbo", help="Name of Matplotlib colormap to use")
parser.add_argument('--binary_cmap_name', type=str, default="plasma", help="Name of Matplotlib colormap to use for ploting individual factors")
parser.add_argument('--color_table', type=str, default='', help='Pre-defined color map')
parser.add_argument('--xmin', type=float, default=-1, help="")
parser.add_argument('--ymin', type=float, default=-1, help="")
parser.add_argument('--xmax', type=float, default=np.inf, help="")
parser.add_argument('--ymax', type=float, default=np.inf, help="")
parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Size of the output pixels in um")
parser.add_argument("--plot_individual_factor", action='store_true', help="")
parser.add_argument("--plot_discretized", action='store_true', help="")
parser.add_argument('--single_factor_cutoff', type=float, default=.05, help="")

args = parser.parse_args()
logging.basicConfig(level= getattr(logging, "INFO", None))
dt = np.uint16 if args.tif else np.uint8

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
    y = re.match('^[A-Za-z]*_*(\d+)$', x)
    if y:
        factor_header.append([y.group(0), int(y.group(1)) ])
factor_header.sort(key = lambda x : x[1] )
factor_header = [x[0] for x in factor_header]
K = len(factor_header)

# Colormap
if args.plot_discretized:
    if os.path.exists(args.color_table):
        color_info = pd.read_csv(args.color_table, sep='\t', header=0)
        color_info.Name = color_info.Name.astype(int)
        color_info.sort_values(by=['Name'], inplace=True)
        cmtx = np.array(color_info.loc[:, ["R","G","B"]])
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
        f = args.output + ".cbar.png"
        fig.savefig(f, format="png")
        logging.info(f"Set up color map for {K} factors")

# Read data
adt={x:float for x in ["x", "y"]+factor_header}
df = pd.DataFrame()
for chunk in pd.read_csv(gzip.open(args.input, 'rt'), sep='\t', \
    chunksize=1000000, skiprows=1, names=header, dtype=adt):
    chunk = chunk[(chunk.y > args.ymin) & (chunk.y < args.ymax)]
    chunk = chunk[(chunk.x > args.xmin) & (chunk.x < args.xmax)]
    chunk['x_indx'] = np.round(chunk.x.values / args.plot_um_per_pixel, 0).astype(int)
    chunk['y_indx'] = np.round(chunk.y.values / args.plot_um_per_pixel, 0).astype(int)
    chunk = chunk.groupby(by = ['x_indx', 'y_indx']).agg({ x:np.mean for x in factor_header }).reset_index()
    df = pd.concat([df, chunk])
    print(df.x_indx.iloc[-1], df.y_indx.iloc[-1])

df = df.groupby(by = ['x_indx', 'y_indx']).agg({ x:np.mean for x in factor_header }).reset_index()
df.loc[:, factor_header] = normalize(df.loc[:, factor_header].values, axis=1, norm='l1')
x_indx_min = int(args.xmin / args.plot_um_per_pixel )
y_indx_min = int(args.ymin / args.plot_um_per_pixel )
if args.plot_fit or args.xmin < 0:
    df.x_indx = df.x_indx - np.max([x_indx_min, df.x_indx.min()])
else:
    df.x_indx -= x_indx_min
if args.plot_fit or args.ymin < 0:
    df.y_indx = df.y_indx - np.max([y_indx_min, df.y_indx.min()])
else:
    df.y_indx -= y_indx_min

N0 = df.shape[0]
df.index = range(N0)
hsize, wsize = df[['x_indx','y_indx']].max(axis = 0) + 1
hsize_um = hsize * args.plot_um_per_pixel
wsize_um = wsize * args.plot_um_per_pixel
logging.info(f"Collapse to {N0} pixels in region {hsize_um} x {wsize_um}")


# Make images
if args.plot_discretized:
    binary_mtx = coo_array((np.ones(N0,dtype=bool),\
            (range(N0), np.array(df[factor_header]).argmax(axis = 1))),\
            shape=(N0, K)).toarray()
    rgb_mtx = np.clip(np.around(binary_mtx @ cmtx * 255),0,255).astype(dt)
    img = np.zeros( (hsize, wsize, 3), dtype=dt)
    for r in range(3):
        img[:, :, r] = coo_array((rgb_mtx[:, r], \
                                  (df.x_indx.values, df.y_indx.values)),\
                                 shape=(hsize, wsize), dtype = dt).toarray()
    if args.tif:
        img = Image.fromarray(img, mode="I;16")
    else:
        img = Image.fromarray(img)
    outf = args.output + ".top"
    outf += ".tif" if args.tif else ".png"
    img.save(outf)
    logging.info(f"Made hard threshold image\n{outf}")

if args.plot_individual_factor:
    cutoff = args.single_factor_cutoff
    if args.binary_cmap_name not in plt.colormaps():
        args.binary_cmap_name = "plasma"
    v = np.array(df.loc[:, factor_header].sum(axis = 0) )
    u = np.argsort(-v)
    indiv_k = u
    if args.plot_top_k < K and args.plot_top_k > 0:
        indiv_k = u[:args.plot_top_k]
    elif args.plot_prob_cut < 1 and args.plot_prob_cut > 0:
        w = np.cumsum(v[u])
        k = np.arange(K)[w >= args.plot_prob_cut].min()
        indiv_k = u[:k]
    for k in indiv_k:
        indx = df[factor_header[k]] > cutoff
        v = np.clip(df.loc[indx, factor_header[k]].values,0,1)
        rgb_mtx = np.clip(mpl.colormaps[args.binary_cmap_name](v)[:,:3]*255,0,255).astype(dt)
        img = np.zeros( (hsize, wsize, 3), dtype=dt)
        for r in range(3):
            img[:, :, r] = coo_array((rgb_mtx[:, r],\
                                      (df.loc[indx, "x_indx"].values,\
                                       df.loc[indx, "y_indx"].values )),\
                                    shape=(hsize, wsize), dtype = dt).toarray()
        if args.tif:
            img = Image.fromarray(img, mode="I;16")
        else:
            img = Image.fromarray(img)
        outf = args.output + ".F_"+str(k)
        outf += ".tif" if args.tif else ".png"
        img.save(outf)
        logging.info(f"Made factor specific image - {k}\n{outf}")
