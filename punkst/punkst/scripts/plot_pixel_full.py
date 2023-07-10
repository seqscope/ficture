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

from punkst.utils import utilt

def plot_pixel_full(_args):

    parser = argparse.ArgumentParser(prog="plot_pixel_full")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help='Output file full path')
    parser.add_argument('--color_table', type=str, default='', help='Pre-defined color map')
    parser.add_argument('--color_table_index_column', type=str, default='Name', help='')
    parser.add_argument('--xmin', type=float, default=-np.inf, help="")
    parser.add_argument('--ymin', type=float, default=-np.inf, help="")
    parser.add_argument('--xmax', type=float, default=np.inf, help="")
    parser.add_argument('--ymax', type=float, default=np.inf, help="")
    parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Actual size (um) corresponding to each pixel in the output image")
    parser.add_argument('--plot_top', action='store_true', help="Plot top factor only")

    args = parser.parse_args(_args)
    logger =  logging.getLogger(__name__)

    # Read color table
    rgb=['B','G','R'] # opencv rgb order
    cdty = {x:float for x in rgb}
    color_info = pd.read_csv(args.color_table, sep='\t', header=0, index_col=args.color_table_index_column, dtype=cdty)
    color_info.index = color_info.index.astype(str)
    logger.info(f"Read color table ({color_info.shape[0]})")

    # Read header
    meta = {}
    nheader = 0
    with gzip.open(args.input, 'rt') as rf:
        for line in rf:
            if line[0] != "#":
                break
            nheader += 1
            if line[:2] == "##":
                wd = line[(line.rfind("#")+1):].strip().split(';')
                wd = [[y.strip() for y in x.strip().split("=")] for x in wd]
                for v in wd:
                    if v[1].lstrip('-+').isdigit():
                        meta[v[0]] = int(v[1])
                    elif v[1].replace('.','',1).lstrip('-+').isdigit():
                        meta[v[0]] = float(v[1])
                    else:
                        meta[v[0]] = v[1]
            else:
                header = line[(line.rfind("#")+1):].strip().split('\t')
    logger.info("Read header %s", meta)

    # Input reader
    dty={'BLOCK':str, 'X':int, 'Y':int}
    dty.update({'K'+str(k+1) : str for k in range(meta['TOPK']) })
    dty.update({'P'+str(k+1) : float for k in range(meta['TOPK']) })
    if not np.isfinite(args.xmin) or not np.isfinite(args.xmax):
        reader = pd.read_csv(args.input,sep='\t',skiprows=nheader,chunksize=1000000,names=header, dtype=dty)
    else:
        # Translate target region to index
        block = [int(x / meta['BLOCK_SIZE']) for x in [args.xmin, args.xmax - 1] ]
        pos_range = [int((x - meta['OFFSET_Y'])*meta['SCALE']) for x in [args.ymin, args.ymax]]
        if meta['BLOCK_AXIS'] == "Y":
            block = [int(x / meta['BLOCK_SIZE']) for x in [args.ymin, args.ymax - 1] ]
            pos_range = [int((x - meta['OFFSET_X'])*meta['SCALE']) for x in [args.xmin, args.xmax]]
        block = np.arange(block[0], block[1]+1) * meta['BLOCK_SIZE']
        query = []
        pos_range = '-'.join([str(x) for x in pos_range])
        for i,b in enumerate(block):
            query.append( str(b)+':'+pos_range )
        cmd = ["tabix", args.input]+query
        process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT)
        reader = pd.read_csv(process.stdout,sep='\t',chunksize=1000000,names=header, dtype=dty)
        logger.info(" ".join(cmd))

    args.xmin = max(args.xmin, meta['OFFSET_X'])
    args.xmax = min(args.xmax, meta['OFFSET_X'] + meta["SIZE_X"])
    args.ymin = max(args.ymin, meta['OFFSET_Y'])
    args.ymax = min(args.ymax, meta['OFFSET_Y'] + meta["SIZE_Y"])

    width = int((args.xmax - args.xmin + 1)/args.plot_um_per_pixel)
    height= int((args.ymax - args.ymin + 1)/args.plot_um_per_pixel)

    logger.info(f"Image size {height} x {width}")

    # Read input file, fill the rgb matrix
    df = pd.DataFrame()
    keptcol = ['X','Y'] + rgb
    for chunk in reader:
        chunk['X']=chunk.X/meta['SCALE']+meta['OFFSET_X']
        chunk['Y']=chunk.Y/meta['SCALE']+meta['OFFSET_Y']
        drop_index = chunk.index[(chunk.X<args.xmin)|(chunk.X>args.xmax)|\
                                (chunk.Y<args.ymin)|(chunk.Y>args.ymax)]
        chunk.drop(index=drop_index, inplace=True)
        logger.info(f"Reading pixels... {chunk.X.iloc[-1]}, {chunk.Y.iloc[-1]}, {df.shape[0]}")
        chunk['X'] = np.clip(((chunk.X - args.xmin) / args.plot_um_per_pixel).astype(int),0,width-1)
        chunk['Y'] = np.clip(((chunk.Y - args.ymin) / args.plot_um_per_pixel).astype(int),0,height-1)
        if args.plot_top:
            for c in rgb:
                chunk[c] = color_info.loc[chunk['K1'].values, c].values
        else:
            for c in rgb:
                chunk[c] = 0
                for k in range(1,meta['TOPK']+1):
                    chunk[c] += color_info.loc[chunk['K'+str(k)].values, c].values * chunk['P'+str(k)].values
        chunk = chunk.groupby(by=['X','Y']).agg({c:np.mean for c in rgb}).reset_index()
        for c in rgb:
            chunk[c] = np.clip(np.around(chunk[c] * 255),0,255).astype(np.uint8)
        df = pd.concat([df, chunk[keptcol]])
    df.drop_duplicates(subset=['X','Y'], inplace=True)
    logger.info(f"Read {df.shape[0]} pixels")

    img = np.zeros((height,width,3), dtype=np.uint8)
    for i,c in enumerate(rgb):
        img[:, :, i] = coo_array( ( df[c], (df.Y, df.X) ), shape=(height, width) ).toarray()

    if not args.output.endswith(".png"):
        args.output += ".png"
    cv2.imwrite(args.output,img)
    logger.info(f"Finished")
