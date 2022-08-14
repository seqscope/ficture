import sys, os, re
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import *

import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help="Input file name")
parser.add_argument('--output', type=str, help="Output file prefix")
parser.add_argument('--cmap_name', default='turbo', type=str, help="Specify harmonized color code")
parser.add_argument('--um_per_pixel', type=float, default=4, help="Size of the output pixels in um")
parser.add_argument("--top", default=False, action='store_true', help="")
parser.add_argument("--tif", default=False, action='store_true', help="Store as 16-bit tif instead of png")
parser.add_argument("--plot_fit", default=False, action='store_true', help="")
args = parser.parse_args()

cmap_name = args.cmap_name
if args.cmap_name not in plt.colormaps():
    cmap_name = "turbo"

dt = np.uint16 if args.tif else np.uint8
outf = args.output + ".tif" if args.tif else args.output + ".png"

org_fit = pd.read_csv(args.input, sep='\t',header=0)

topic_header = [] 
for x in org_fit.columns:
    v = re.match('(^[A-Za-z]+_\d+$)', x)
    if v:
        topic_header.append(v.group(0))
K = len(topic_header)

cmap = plt.get_cmap('turbo', K)
cmtx = np.array([cmap(i) for i in range(K)] )[:, :3]
np.random.shuffle(cmtx)

org_fit.rename(columns = {'Hex_center_x':'x', 'Hex_center_y':'y'}, inplace=True)
if args.plot_fit:
    org_fit.x -= org_fit.x.min()
    org_fit.y -= org_fit.y.min()

org_fit['x_indx'] = np.round(np.clip(org_fit.x.values / args.um_per_pixel,0,np.inf), 0).astype(int)
org_fit['y_indx'] = np.round(np.clip(org_fit.y.values / args.um_per_pixel,0,np.inf), 0).astype(int)

if args.top:
    amax = np.array(org_fit[topic_header]).argmax(axis = 1)
    org_fit[topic_header] = coo_array((np.ones(org_fit.shape[0],dtype=np.int8), (range(org_fit.shape[0]), amax)), shape=(org_fit.shape[0], K)).toarray()

org_fit = org_fit.groupby(by = ['x_indx', 'y_indx']).agg({ x:np.mean for x in topic_header }).reset_index()
h, w = org_fit[['x_indx','y_indx']].max(axis = 0) + 1

rgb_mtx = np.clip(np.around(np.array(org_fit[topic_header]) @ cmtx * 255).astype(dt),0,255)
img = np.zeros( (h, w, 3), dtype=dt)
for r in range(3):
    img[:, :, r] = coo_array((rgb_mtx[:, r], (org_fit.x_indx.values, org_fit.y_indx.values)), shape=(h,w), dtype = dt).toarray()
# print(dt,K,h,w,rgb_mtx.shape,img.shape,img.dtype) 

if args.tif:
    img_rgb = Image.fromarray(img, mode="I;16")
else:
    img_rgb = Image.fromarray(img)
img_rgb.save(outf)