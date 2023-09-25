# Visualize selected genes
# Experimental: specific to input file format

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
parser.add_argument('--channel_list', type=str, nargs='*', default = [], help="Select individual genes or gene sets each defined by a regex")
parser.add_argument('--weight_by_column', type=str, default='', help='')
parser.add_argument('--color_list', type=str, nargs='*', default=["1781b5", "FF9900", "DD65E6", "FFEC11"], help='') # blue, orange, purple, yellow
parser.add_argument('--color_table', type=str, default='', help='Color maps written in a tab delimited file, with columns channel, R, G, B')
parser.add_argument('--color_table_index_column', type=str, default='channel', help='')
parser.add_argument('--xmin', type=float, default=-np.inf, help="")
parser.add_argument('--ymin', type=float, default=-np.inf, help="")
parser.add_argument('--xmax', type=float, default=np.inf, help="")
parser.add_argument('--ymax', type=float, default=np.inf, help="")
parser.add_argument('--org_coord', action='store_true', help="If the input coordinates do not include the offset (if your coordinates are from an existing figure, the offset is already factored in)")
parser.add_argument('--full', action='store_true', help="")
parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Actual size (um) corresponding to each pixel in the output image")
parser.add_argument('--debug', action='store_true', help="")

args = parser.parse_args()
logging.basicConfig(level= getattr(logging, "INFO", None), format='%(asctime)s %(message)s', datefmt='%I:%M:%S %p')

rgb=list("RGB")
bgr=list("BGR") # opencv default channel order
pcut = 0.01
spcut = 0.1
if len(args.channel_list) > 5:
    logging.warning("Be colorblind friendly, visualize a smaller number of channels at a time or be careful in choosing the colors")
# Read color table including the input gene(set)s and RGB
if os.path.exists(args.color_table):
    ccol=args.color_table_index_column
    dty = {x:float for x in rgb}
    dty[ccol] = str
    color_info = pd.read_csv(args.color_table, sep='\t', header=0, dtype=dty)
    if ccol not in color_info.columns:
        logging.error(f"Input color table does not contain column {ccol}")
        sys.exit(1)
    if color_info[rgb].max().max() > 1:
        logging.warning("Color table should contain RGB values in 0-1 range, but found values > 1, will interpret as in 0-255 range")
        for c in rgb:
            color_info[c] = np.clip(color_info[c]/255,0,1)
    if len(args.channel_list) > 0:
        color_info = color_info.loc[color_info[ccol].isin(args.channel_list), :]
    if len(args.channel_list) < len(color_info):
        logging.error("Some channels are not found in the input color table")
        sys.exit(1)
    color_info = {v[ccol] : np.array(v[rgb]) for i,v in color_info.iterrows()}
else:
    if len(args.color_list) < len(args.channel_list):
        logging.error("Number of input colors and channels do not match")
        sys.exit(1)
    color_list = [[int(x.strip('#')[i:i+2],16)/255 for i in (0,2,4)] for x in args.color_list]
    color_info = {x:np.array(color_list[i]) for i,x in enumerate(args.channel_list)}

logging.info(f"Set up color map {color_info}")
channels = list(color_info.keys())

# Temporary dangerous hack to allow regex in channel_list
def translate_regex(regex):
    regex = regex.replace('\\^', '__CARET__').replace('\\$', '__DOLLAR__')
    regex = re.sub(r'((?<!\[)\^)|(\$(?!\]))', '\\\\b', regex)
    regex = regex.replace('__CARET__', r'^').replace('__DOLLAR__', r'$')
    return regex
channel_regex = []
for v in channels:
    if re.match("^[a-z\d]+$", v, re.IGNORECASE): # pure gene name
        channel_regex.append(r"\b" + v + r"\b")
    else:
        channel_regex.append(translate_regex(v))
regex = "|".join(["("+x+")" for x in channel_regex])
filter_cmd = "perl -lane ' if (m/" + regex + "/) {print $_}'"
if args.debug:
    print(channels)
    print(channel_regex)
    print(filter_cmd)

loader = BlockIndexedLoader(args.input, args.xmin, args.xmax, args.ymin, args.ymax, args.full, not args.org_coord, filter_cmd=filter_cmd)
width = int((loader.xmax - loader.xmin + 1)/args.plot_um_per_pixel)
height= int((loader.ymax - loader.ymin + 1)/args.plot_um_per_pixel)
logging.info(f"Image size {height} x {width}")

# Read input file, fill the rgb matrix
img = np.zeros((height,width,3), dtype=np.uint8)
for df in loader:
    if df.shape[0] == 0:
        continue
    logging.info(f"Reading pixels... {df.X.iloc[-1]}, {df.Y.iloc[-1]}, {df.shape[0]}")
    df['X'] = np.clip(((df.X - loader.xmin) / args.plot_um_per_pixel).astype(int),0,width-1)
    df['Y'] = np.clip(((df.Y - loader.ymin) / args.plot_um_per_pixel).astype(int),0,height-1)
    df[rgb] = 0
    for i,k in enumerate(channels):
        indx = df.gene.str.match(channel_regex[i], case=False)
        if indx.sum() == 0:
            continue
        detected_gene = " ".join(list(df.loc[indx, 'gene'].unique()))
        logging.info(f"{k}: {detected_gene}")
        df.loc[indx, rgb] = np.array(list(color_info[k]))
    if args.weight_by_column in df.columns:
        df[rgb] *= df[args.weight_by_column].values.reshape((-1,1))
        adt = {c:np.sum for c in rgb}
        adt[args.weight_by_column] = np.sum
        df = df.groupby(by=['X','Y']).agg(adt).reset_index()
        df[rgb] /= df[args.weight_by_column].values.reshape((-1,1))
    else:
        df = df.groupby(by=['X','Y']).agg({c:np.mean for c in rgb}).reset_index()
    df[rgb] = np.clip(np.around(df[rgb] * 255),0,255).astype(np.uint8)
    for i,c in enumerate(bgr):
        img[df.Y.values, df.X.values, [i]*df.shape[0]] = df[c].values
    if args.debug:
        break

if not args.output.endswith(".png"):
    args.output += ".png"
cv2.imwrite(args.output,img)
logging.info(f"Finished.\n{args.output}")
