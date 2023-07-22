# Visualize pixel level single factor heatmap
# Input file contains only top k factors and probabilities per pixel
# Meant to make use of the indexed input to plot for specified regions quickly
# Would take a huge amount of memory if trying to plot many factors simultaneously in a large region

import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd
from scipy.sparse import *
import subprocess as sp
from joblib.parallel import Parallel, delayed
import matplotlib as mpl
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utilt
from pixel_factor_loader import BlockIndexedLoader

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help="Output prefix")
parser.add_argument('--id_list', type=str, nargs="*", default=[], help="List of IDs of the factors to plot")
parser.add_argument('--category_column', type=str, help='If the input contains categorical labels instead of probabilities')
parser.add_argument('--category_map', type=str, default='', help='If needed, map the input category to a different set of names matching that specified in the color table. The mathcing could be multiple to one, input should have two columns as "From" and "To"')
parser.add_argument('--unmapped', type=str, default='', help='')

parser.add_argument('--binary_cmap_name', type=str, default="plasma", help="Name of Matplotlib colormap to use for ploting individual factors")

parser.add_argument('--xmin', type=float, default=-np.inf, help="")
parser.add_argument('--ymin', type=float, default=-np.inf, help="")
parser.add_argument('--xmax', type=float, default=np.inf, help="")
parser.add_argument('--ymax', type=float, default=np.inf, help="")
parser.add_argument('--org_coord', action='store_true', help="If the input coordinates do not include the offset (if your coordinates are from an existing figure, the offset is already factored in)")
parser.add_argument('--full', action='store_true', help="Read full input")
parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Actual size (um) corresponding to each pixel in the output image")
parser.add_argument('--all', action="store_true", help="")
parser.add_argument("--debug", action='store_true', help="")

args = parser.parse_args()
logging.basicConfig(level= getattr(logging, "INFO", None), format='%(asctime)s %(message)s', datefmt='%I:%M:%S %p')
if not os.path.exists(os.path.dirname(args.output)):
    os.makedirs(os.path.dirname(args.output))

kcol = args.category_column
kept_cols = ['X','Y',kcol]

category_rename = False
category_map = {}
if os.path.exists(args.category_map):
    if args.category_map.endswith(".gz"):
        rf = gzip.open(args.category_map, "rt")
    else:
        rf = open(args.category_map, "r")
    for line in rf:
        wd = line.strip().split('\t')
        category_map[wd[0]] = wd[1]
    rf.close()
    category_rename = True
    category_list = sorted(list(set(category_map.values())))
    K = len(category_list)
    logging.info(f"Read category map ({len(category_map)}to {K} categories)")
    if args.debug:
        print(category_list)

loader = BlockIndexedLoader(args.input, args.xmin, args.xmax, args.ymin, args.ymax, args.full, not args.org_coord, idtype={kcol:str} )
width = int((loader.xmax - loader.xmin + 1)/args.plot_um_per_pixel)
height= int((loader.ymax - loader.ymin + 1)/args.plot_um_per_pixel)
logging.info(f"Image size {height} x {width}")
if args.debug:
    logging.info(f"{loader.xmin}, {loader.xmax}, {loader.ymin}, {loader.ymax}")

# Read input file
df = pd.DataFrame()
for chunk in loader:
    if chunk.shape[0] == 0:
        continue
    chunk = chunk.loc[:, kept_cols]
    chunk['X'] = np.clip(((chunk.X - loader.xmin) / args.plot_um_per_pixel).astype(int),0,width-1)
    chunk['Y'] = np.clip(((chunk.Y - loader.ymin) / args.plot_um_per_pixel).astype(int),0,height-1)
    if category_rename:
        chunk[kcol] = chunk[kcol].map(category_map)
        if args.unmapped != '':
            chunk[kcol].fillna(args.unmapped, inplace=True)
        else:
            chunk = chunk.loc[~chunk[kcol].isna(), :]
        if args.debug:
            print(chunk.shape[0], chunk[kcol].value_counts())
    if len(args.id_list) > 0:
        chunk = chunk.loc[chunk[kcol].isin(args.id_list), :]
    chunk.drop_duplicates(inplace=True)
    df = pd.concat([df, chunk])
    logging.info(f"Reading pixels... {chunk.X.iloc[-1]}, {chunk.Y.iloc[-1]}, {df.shape[0]}")
    if args.debug:
        break

df.drop_duplicates(inplace=True)
df.index = np.arange(df.shape[0])
logging.info(f"Read {df.shape[0]} pixels")

id_list = args.id_list
if len(args.id_list) == 0:
    ct = df[kcol].value_counts()
    ct.sort_values(ascending=False, inplace=True)
    id_list = ct.index.tolist()

for k in id_list:
    indx = df.index[df[kcol].eq(k)]
    if len(indx) < 10:
        continue
    img = coo_array( ( np.ones(len(indx), dtype=np.uint8) * 255, \
        (df.loc[indx, 'Y'], df.loc[indx, 'X']) ), shape=(height, width) ).toarray()
    cv2.imwrite(args.output +".F_" +k+".png",img)
    logging.info(f"Made image for {k}")
    if args.debug:
        break

logging.info(f"Finished")
