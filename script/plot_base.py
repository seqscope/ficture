import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd
from random import shuffle

import matplotlib.pyplot as plt
from PIL import Image

from scipy.sparse import *
import sklearn.neighbors
import sklearn.preprocessing

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *
from utilt import plot_colortable

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help='Output prefix')
parser.add_argument('--fill_range', type=float, default=5, help="um")
parser.add_argument('--batch_size', type=float, default=1000000, help="")
parser.add_argument('--cmap_name', type=str, default="turbo", help="Name of Matplotlib colormap to use")
parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Size of the output pixels in um")
parser.add_argument('--xmin', type=float, default=-1, help="")
parser.add_argument('--ymin', type=float, default=-1, help="")
parser.add_argument('--xmax', type=float, default=np.inf, help="")
parser.add_argument('--ymax', type=float, default=np.inf, help="")
parser.add_argument("--tif", action='store_true', help="Store as 16-bit tif instead of png")
parser.add_argument("--plot_fit", action='store_true', help="")
parser.add_argument("--plot_individual_factor", action='store_true', help="")
parser.add_argument("--plot_discretized", action='store_true', help="")

args = parser.parse_args()
logging.basicConfig(level= getattr(logging, "INFO", None))

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
dt = np.uint16 if args.tif else np.uint8
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
for chunk in pd.read_csv(gzip.open(args.input, 'rt'), sep='\t', \
    chunksize=1000000, skiprows=1, names=header, dtype=adt):
    chunk = chunk[(chunk.y > args.ymin) & (chunk.y < args.ymax)]
    chunk = chunk[(chunk.x > args.xmin) & (chunk.x < args.xmax)]
    chunk['x_indx'] = np.round(chunk.x.values / args.plot_um_per_pixel, 0).astype(int)
    chunk['y_indx'] = np.round(chunk.y.values / args.plot_um_per_pixel, 0).astype(int)
    chunk = chunk.groupby(by = ['x_indx', 'y_indx']).agg({ x:np.mean for x in factor_header }).reset_index()
    df = pd.concat([df, chunk])

df = df.groupby(by = ['x_indx', 'y_indx']).agg({ x:np.mean for x in factor_header }).reset_index()
x_indx_min = int(args.xmin / args.plot_um_per_pixel )
y_indx_min = int(args.ymin / args.plot_um_per_pixel )
if args.plot_fit or args.xmin < 0:
    df.x_indx = df.x_indx - np.max([x_indx_min, df.x_indx.min()])
if args.plot_fit or args.ymin < 0:
    df.y_indx = df.y_indx - np.max([y_indx_min, df.y_indx.min()])

N0 = df.shape[0]
df.index = range(N0)
hsize, wsize = df[['x_indx','y_indx']].max(axis = 0) + 1
hsize_um = hsize * args.plot_um_per_pixel
wsize_um = wsize * args.plot_um_per_pixel
logging.info(f"Read region {N0} pixels in region {hsize_um} x {wsize_um}")


# Make images
binary_mtx = coo_matrix((np.ones(N0,dtype=bool),\
        (range(N0), np.array(df[factor_header]).argmax(axis = 1))),\
        shape=(N0, K)).toarray()
wst = df.y_indx.min()
wed = df.y_indx.max()
wstep = np.max([10, int(args.batch_size / hsize)])
rgb_mtx = np.clip(np.around(np.array(df[factor_header]) @ cmtx * 255),0,255).astype(dt)
rgb_mtx_hard = np.clip(np.around(binary_mtx @ cmtx * 255),0,255).astype(dt)
pts = np.array(df[['x_indx', 'y_indx']], dtype=int)
pts_indx = list(df.index)
ref = sklearn.neighbors.BallTree(pts)

st = wst
while st < wed:
    ed = min([st + wstep, wed])
    print(st, ed, pts.shape[0])
    if ((df.y_indx > st) & (df.y_indx < ed)).sum() < 10:
        st = ed
        continue
    mesh = np.meshgrid(np.arange(hsize), np.arange(st, ed))
    nodes = np.array(list(zip(*(dim.flat for dim in mesh))), dtype=int)
    dv, iv = ref.query(nodes, k = 1, dualtree=True)
    indx = (dv[:, 0] < args.fill_range/args.plot_um_per_pixel) & (dv[:, 0] > 0)
    iv = iv[indx, 0]
    if sum(indx) == 0:
        st = ed
        continue
    pts = np.vstack((pts, nodes[indx, :]) )
    pts_indx += list(df.index[iv] )
    rgb_mtx = np.vstack((rgb_mtx,\
        np.clip(np.around( np.array(\
            df.loc[iv,factor_header]) @ cmtx * 255),0,255).astype(dt)))
    if args.plot_discretized:
        rgb_mtx_hard = np.vstack((rgb_mtx_hard,\
            np.clip(np.around(np.array(\
                binary_mtx[iv, :]) @ cmtx * 255),0,255).astype(dt)))
    st = ed

pts[:,0] = np.clip(hsize - pts[:, 0], 0, hsize-1) # Origin is lower-left
pts[:,1] = np.clip(pts[:, 1], 0, wsize-1)

img = np.zeros( (hsize, wsize, 3), dtype=dt)
for r in range(3):
    img[:, :, r] = coo_array((rgb_mtx[:, r], (pts[:,0], pts[:,1])),\
        shape=(hsize, wsize), dtype = dt).toarray()
if args.tif:
    img = Image.fromarray(img, mode="I;16")
else:
    img = Image.fromarray(img)

outf = args.output
outf += ".tif" if args.tif else ".png"
img.save(outf)
logging.info(f"Made fractional image\n{outf}")

if args.plot_discretized:
    img = np.zeros( (hsize, wsize, 3), dtype=dt)
    for r in range(3):
        img[:, :, r] = coo_array((rgb_mtx_hard[:, r], (pts[:,0], pts[:,1])),\
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
    binary_rgb = np.array([[255,153,0], [17,101,154]])
    for k in range(K):
        v = np.clip(df.loc[pts_indx, factor_header[k]].values,0,1)
        rgb_mtx = np.clip(np.vstack((v, 1-v)).T @ binary_rgb, 0, 255).astype(dt)
        img = np.zeros( (hsize, wsize, 3), dtype=dt)
        for r in range(3):
            img[:, :, r] = coo_array((rgb_mtx[:, r], (pts[:,0], pts[:,1])),\
                shape=(hsize, wsize), dtype = dt).toarray()
        if args.tif:
            img = Image.fromarray(img, mode="I;16")
        else:
            img = Image.fromarray(img)
        outf = args.output + ".F_"+str(k)
        outf += ".tif" if args.tif else ".png"
        img.save(outf)
        logging.info(f"Made factor specific image - {k}\n{outf}")
