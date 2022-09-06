import sys, os, re, gzip, copy
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import *
import sklearn.neighbors

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilt import plot_colortable

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help="Input file name")
parser.add_argument('--figure_path', type=str, help="Output path")
parser.add_argument('--output_id', type=str, help="Output file prefix")
parser.add_argument('--cmap_name', default='turbo', type=str, help="Specify harmonized color code")
parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Size of the output pixels in um")
parser.add_argument('--fill_range', type=float, default=10, help="um")
parser.add_argument('--chunk_size', type=int, default=5000, help="um")
parser.add_argument("--plot_top", default=False, action='store_true', help="")
parser.add_argument("--tif", default=False, action='store_true', help="Store as 16-bit tif instead of png")
args = parser.parse_args()

cmap_name = args.cmap_name
if args.cmap_name not in plt.colormaps():
    cmap_name = "turbo"

dt = np.uint16 if args.tif else np.uint8

lda_base_result = pd.read_csv(gzip.open(args.input, 'rb'), sep='\t',header=0)

topic_header = []
for x in lda_base_result.columns:
    v = re.match('(^[A-Za-z]+_\d+$)', x)
    if v:
        topic_header.append(v.group(0))
K = len(topic_header)

print(K, lda_base_result.shape)

cmap_name = args.cmap_name
if args.cmap_name not in plt.colormaps():
    cmap_name = "turbo"
cmap = plt.get_cmap(cmap_name, K)
cmtx = np.array([cmap(i) for i in range(K)] )
shuffle(cmtx)
cdict = {k:cmtx[k,:3] for k in range(K)}

# Plot color bar separately
fig = plot_colortable(cdict, "Factor label", sort_colors=False, ncols=4)
f = args.figure_path + "/color_legend."+args.output_id+".png"
fig.savefig(f)

lda_base_result['x_indx'] = np.round(lda_base_result.x.values / args.plot_um_per_pixel, 0).astype(int)
lda_base_result['y_indx'] = np.round(lda_base_result.y.values / args.plot_um_per_pixel, 0).astype(int)

if args.plot_top:
    amax = np.array(lda_base_result[topic_header]).argmax(axis = 1)
    lda_base_result[topic_header] = coo_matrix((np.ones(lda_base_result.shape[0],dtype=np.int8), (range(lda_base_result.shape[0]), amax)), shape=(lda_base_result.shape[0], K)).toarray()

lda_base_result = lda_base_result.groupby(by = ['x_indx', 'y_indx']).agg({ x:np.mean for x in topic_header }).reset_index()
h, w = lda_base_result[['x_indx','y_indx']].max(axis = 0) + 1

# plot by chunk?
wsize = args.chunk_size
bsize = 1000
w_bins = np.arange(0,w,wsize).astype(int)
w_labs = np.arange(len(w_bins)-1)
lda_base_result["plot_window"] = pd.cut(lda_base_result.y_indx, bins=w_bins,labels=w_labs)

for bk in lda_base_result.plot_window.unique():
    org_fit = copy.copy(lda_base_result[lda_base_result.plot_window.eq(bk)])
    if org_fit.shape[0] < 100:
        continue
    org_fit.index = range(org_fit.shape[0])
    org_fit["y_indx"] = org_fit["y_indx"] - w_bins[bk]
    print(org_fit.y_indx.min(), org_fit.y_indx.max(), org_fit.shape[0])
    rgb_mtx = np.clip(np.around(np.array(org_fit[topic_header]) @ cmtx * 255),0,255).astype(dt)

    ref = sklearn.neighbors.BallTree(np.array(org_fit[['x_indx', 'y_indx']]))
    fill_mtx = np.zeros((0, 4), dtype=dt)
    fill_pts = np.zeros((0, 2), dtype=int)

    st = 0
    while st < wsize:
        ed = min([st + bsize, wsize+1])
        if ((org_fit.y_indx > st) & (org_fit.y_indx < ed)).sum() < 10:
            st = ed
            continue
        mesh = np.meshgrid(np.arange(h+1), np.arange(st, ed))
        nodes = np.array(list(zip(*(dim.flat for dim in mesh))), dtype=int)
        dv, iv = ref.query(nodes, k = 1)
        indx = (dv[:, 0] < args.fill_range) & (dv[:, 0] > 0)
        fill_pts = np.vstack((fill_pts, nodes[indx, :]) )
        fill_mtx = np.vstack((fill_mtx, np.clip(np.around(np.array(org_fit.loc[iv[indx, 0], topic_header]) @ cmtx * 255),0,255).astype(dt)) )
        print(st, ed, sum(indx), fill_pts.shape[0])
        st = ed

    img = np.zeros( (h+2, wsize+2, 3), dtype=dt)
    for r in range(3):
        img[:, :, r] = coo_array((rgb_mtx[:, r], (org_fit.x_indx.values+1, org_fit.y_indx.values+1)), shape=(h+2,wsize+2), dtype = dt).toarray() +\
                       coo_array((fill_mtx[:, r], (fill_pts[:, 0]+1, fill_pts[:, 1]+1)), shape=(h+2,wsize+2), dtype = dt).toarray()

    if args.tif:
        img_rgb = Image.fromarray(img, mode="I;16")
    else:
        img_rgb = Image.fromarray(img)

    outf = args.figure_path + "/"+args.output_id+".X"+str(int(bk*wsize))
    outf += ".tif" if args.tif else ".png"
    img_rgb.save(outf)
    print(bk)
