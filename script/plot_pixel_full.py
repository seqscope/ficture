# Visualize pixel level factor analysis results
# Input file contains only top k factors and probabilities per pixel
# Meant to make use of the indexed input to plot for specified regions quickly

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
parser.add_argument('--output', type=str, help='Output file full path')
parser.add_argument('--color_table', type=str, help='Pre-defined color map')
parser.add_argument('--color_table_index_column', type=str, default='Name', help='')
parser.add_argument('--xmin', type=float, default=-np.inf, help="in um")
parser.add_argument('--ymin', type=float, default=-np.inf, help="in um")
parser.add_argument('--xmax', type=float, default=np.inf, help="in um")
parser.add_argument('--ymax', type=float, default=np.inf, help="in um")
parser.add_argument('--full', action='store_true', help="Read full input")
parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Actual size (um) corresponding to each pixel in the output image")
parser.add_argument('--org_coord', action='store_true', help="If the input coordinates do not include the offset (if your coordinates are from an existing figure, the offset is already factored in)")
parser.add_argument('--plot_top', action='store_true', help="Plot top factor only")

args = parser.parse_args()
logging.basicConfig(level= getattr(logging, "INFO", None), format='%(asctime)s %(message)s', datefmt='%I:%M:%S %p')

# Read color table
rgb=['B','G','R'] # opencv rgb order
cdty = {x:float for x in rgb}
color_info = pd.read_csv(args.color_table, sep='\t', header=0, index_col=args.color_table_index_column, dtype=cdty)
color_info.index = color_info.index.astype(str)
logging.info(f"Read color table ({color_info.shape[0]})")

loader = BlockIndexedLoader(args.input, args.xmin, args.xmax, args.ymin, args.ymax, args.full, not args.org_coord)
width = int((loader.xmax - loader.xmin + 1)/args.plot_um_per_pixel)
height= int((loader.ymax - loader.ymin + 1)/args.plot_um_per_pixel)
logging.info(f"Image size {height} x {width}")

# Read input file, fill the rgb matrix
img = np.zeros((height,width,3), dtype=np.uint8)
df = pd.DataFrame()
keptcol = ['X','Y'] + rgb
for df in loader:
    if df.shape[0] == 0:
        continue
    logging.info(f"Reading pixels... {df.X.iloc[-1]}, {df.Y.iloc[-1]}, {df.shape[0]}")
    df['X'] = np.clip(((df.X - loader.xmin) / args.plot_um_per_pixel).astype(int),0,width-1)
    df['Y'] = np.clip(((df.Y - loader.ymin) / args.plot_um_per_pixel).astype(int),0,height-1)
    if args.plot_top:
        for c in rgb:
            df[c] = color_info.loc[df['K1'].values, c].values
    else:
        for c in rgb:
            df[c] = 0
            for k in range(1,loader.meta['TOPK']+1):
                df[c] += color_info.loc[df['K'+str(k)].values, c].values * df['P'+str(k)].values
    df = df.groupby(by=['X','Y']).agg({c:np.mean for c in rgb}).reset_index()
    for i,c in enumerate(rgb):
        df[c] = np.clip(np.around(df[c] * 255),0,255).astype(np.uint8)
        img[df.Y.values, df.X.values, [i]*df.shape[0]] = df[c].values

if not args.output.endswith(".png"):
    args.output += ".png"
cv2.imwrite(args.output,img)
logging.info(f"Finished")
