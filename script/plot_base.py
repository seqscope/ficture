import sys, os, copy, gc, gzip, pickle, argparse, logging, warnings
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
parser.add_argument('--cmap_name', type=str, default="turbo", help="Name of Matplotlib colormap to use")
parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Size of the output pixels in um")
parser.add_argument("--plot_individual_factor", action='store_true', help="")
parser.add_argument("--plot_fit", action='store_true', help="")
parser.add_argument('--xmin', type=float, default=-1, help="")
parser.add_argument('--ymin', type=float, default=-1, help="")
parser.add_argument('--xmax', type=float, default=np.inf, help="")
parser.add_argument('--ymax', type=float, default=np.inf, help="")
parser.add_argument("--tif", action='store_true', help="Store as 16-bit tif instead of png")

args = parser.parse_args()

df = pd.read_csv(args.input, sep='\t')
factor_header = []
for x in df.columns:
    y = re.match('^[A-Za-z]+_\d+$', x)
    if y:
        factor_header.append(y.group(0))

K = len(factor_header)
# Temporary
df.rename(columns = {'Hex_center_x':'x', 'Hex_center_y':'y'}, inplace=True)
#
print(f"Read X coordinates {df.x.min(), df.x.max()}, Y coordinates {df.y.min(), df.y.max()}, {K} factors.")
print(factor_header)

df = df[(df.y > args.ymin) & (df.y < args.ymax)]
df = df[(df.x > args.xmin) & (df.x < args.xmax)]
if args.plot_fit:
    df.y = df.y - np.max([args.ymin, df.y.min()])
    df.x = df.x - np.max([args.xmin, df.x.min()])

print(f"Fill plot for region X {df.x.max()-df.x.min()} um and Y {df.y.max()-df.y.min()} um.")

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
print(cmtx)
cdict = {k:cmtx[k,:] for k in range(K)}

# Plot color bar separately
fig = plot_colortable(cdict, "Factor label", sort_colors=False, ncols=4)
f = args.output + ".cbar.png"
fig.savefig(f)

# Make images

df['x_indx'] = np.round(df.x.values / args.plot_um_per_pixel, 0).astype(int)
df['y_indx'] = np.round(df.y.values / args.plot_um_per_pixel, 0).astype(int)

df = df.groupby(by = ['x_indx', 'y_indx']).agg({ x:np.mean for x in factor_header }).reset_index()
df.index = range(df.shape[0])

amax = np.array(df[factor_header]).argmax(axis = 1)
binary_mtx = coo_matrix((np.ones(df.shape[0],dtype=bool), (range(df.shape[0]), amax)), shape=(df.shape[0], K)).toarray()

h, wsize = df[['x_indx','y_indx']].max(axis = 0) + 1
wst = df.y_indx.min()
wed = df.y_indx.max()
wsize = wed - wst + 1

rgb_mtx = np.clip(np.around(np.array(df[factor_header]) @ cmtx * 255),0,255).astype(dt)
rgb_mtx_hard = np.clip(np.around(binary_mtx @ cmtx * 255),0,255).astype(dt)
pts = np.array(df[['x_indx', 'y_indx']], dtype=int)
pts_indx = list(df.index)
ref = sklearn.neighbors.BallTree(pts)

st = wst
bsize = 1000
while st < wed:
    ed = min([st + bsize, wed])
    if ((df.y_indx > st) & (df.y_indx < ed)).sum() < 10:
        st = ed
        continue
    mesh = np.meshgrid(np.arange(h), np.arange(st, ed))
    nodes = np.array(list(zip(*(dim.flat for dim in mesh))), dtype=int)
    dv, iv = ref.query(nodes, k = 1, dualtree=True)
    indx = (dv[:, 0] < args.fill_range) & (dv[:, 0] > 0)
    pts = np.vstack((pts, nodes[indx, :]) )
    pts_indx += list(df.index[iv[indx, 0]] )
    rgb_mtx = np.vstack((rgb_mtx,\
        np.clip(np.around(np.array(df.loc[iv[indx, 0], factor_header]) @ cmtx * 255),0,255).astype(dt)) )
    rgb_mtx_hard = np.vstack((rgb_mtx_hard,\
        np.clip(np.around(np.array(binary_mtx[iv[indx, 0], :]) @ cmtx * 255),0,255).astype(dt)) )
    print(st, ed, sum(indx), pts.shape[0])
    st = ed

pts[:,0] = np.clip(h - pts[:, 0], 0, h-1)
pts[:,1] = np.clip(pts[:, 1], 0, wsize-1)

img = np.zeros( (h, wsize, 3), dtype=dt)
for r in range(3):
    img[:, :, r] = coo_array((rgb_mtx[:, r], (pts[:,0], pts[:,1])),\
        shape=(h, wsize), dtype = dt).toarray()
if args.tif:
    img = Image.fromarray(img, mode="I;16")
else:
    img = Image.fromarray(img)

outf = args.output
outf += ".tif" if args.tif else ".png"
img.save(outf)
print(f"Made fractional image\n{outf}")

img = np.zeros( (h, wsize, 3), dtype=dt)
for r in range(3):
    img[:, :, r] = coo_array((rgb_mtx_hard[:, r], (pts[:,0], pts[:,1])),\
        shape=(h, wsize), dtype = dt).toarray()
if args.tif:
    img = Image.fromarray(img, mode="I;16")
else:
    img = Image.fromarray(img)

outf = args.output + ".top"
outf += ".tif" if args.tif else ".png"
img.save(outf)
print(f"Made hard threshold image\n{outf}")

binary_rgb = np.array([[255,153,0], [17,101,154]])
if args.plot_individual_factor:
    for k in range(K):
        v = np.clip(df.loc[pts_indx, factor_header[k]].values,0,1)
        rgb_mtx = np.clip(np.vstack((v, 1-v)).T @ binary_rgb, 0, 255).astype(dt)
        img = np.zeros( (h, wsize, 3), dtype=dt)
        for r in range(3):
            img[:, :, r] = coo_array((rgb_mtx[:, r], (pts[:,0], pts[:,1])),\
                shape=(h, wsize), dtype = dt).toarray()
        if args.tif:
            img = Image.fromarray(img, mode="I;16")
        else:
            img = Image.fromarray(img)
        outf = args.output + ".F_"+str(k)
        outf += ".tif" if args.tif else ".png"
        img.save(outf)
        print(f"Made factor specific image - {k}\n{outf}")
