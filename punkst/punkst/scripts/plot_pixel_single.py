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

from punkst.utils import utilt

def plot_pixel_single(_args):

    parser = argparse.ArgumentParser(prog="plot_pixel_single")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help="Output prefix")
    parser.add_argument('--id_list', type=str, nargs="+", help="List of IDs of the factors to plot")

    parser.add_argument('--pcut', type=float, default=1e-2, help="")
    parser.add_argument('--binary_cmap_name', type=str, default="plasma", help="Name of Matplotlib colormap to use for ploting individual factors")

    parser.add_argument('--xmin', type=float, default=-np.inf, help="")
    parser.add_argument('--ymin', type=float, default=-np.inf, help="")
    parser.add_argument('--xmax', type=float, default=np.inf, help="")
    parser.add_argument('--ymax', type=float, default=np.inf, help="")
    parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Actual size (um) corresponding to each pixel in the output image")
    parser.add_argument('--all', action="store_true", help="Caution: when set, assume factors are named as 0, 1, ... K-1, where K is defined in the input header. Only use when plotting a small region.")

    args = parser.parse_args(_args)
    logger = logging.getLogger(__name__)

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

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

    id_list = args.id_list
    if args.all:
        id_list = [str(k) for k in range(meta['K'])]

    # Input reader
    dty={'BLOCK':str, 'X':int, 'Y':int}
    dty.update({'K'+str(k+1) : str for k in range(meta['TOPK']) })
    dty.update({'P'+str(k+1) : np.float16 for k in range(meta['TOPK']) })
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

    # Read input file
    df = pd.DataFrame()
    for chunk in reader:
        chunk['X']=chunk.X/meta['SCALE']+meta['OFFSET_X']
        chunk['Y']=chunk.Y/meta['SCALE']+meta['OFFSET_Y']
        drop_index = chunk.index[(chunk.X<args.xmin)|(chunk.X>args.xmax)|\
                                (chunk.Y<args.ymin)|(chunk.Y>args.ymax)]
        chunk.drop(index=drop_index, inplace=True)
        logger.info(f"Reading pixels... {chunk.X.iloc[-1]}, {chunk.Y.iloc[-1]}, {df.shape[0]}")
        chunk['X'] = np.clip(((chunk.X - args.xmin) / args.plot_um_per_pixel).astype(int),0,width-1)
        chunk['Y'] = np.clip(((chunk.Y - args.ymin) / args.plot_um_per_pixel).astype(int),0,height-1)
        for k in id_list:
            chunk[k] = np.zeros(chunk.shape[0], dtype=np.float16)
            for i in range(1, meta['TOPK']+1):
                indx = chunk['K'+str(i)].eq(k) & chunk['P'+str(i)].gt(args.pcut)
                chunk.loc[indx, k] = chunk.loc[indx, 'P'+str(i)]
        chunk = chunk.groupby(by=['X','Y']).agg({k:np.mean for k in id_list}).reset_index()
        pmax = chunk.loc[:, id_list].max(axis = 1)
        df = pd.concat([df, chunk.loc[pmax > args.pcut, :]])
    df.drop_duplicates(subset=['X','Y'], inplace=True)
    logger.info(f"Read {df.shape[0]} pixels")

    for k in id_list:
        indx = df[k] > args.pcut
        if (df[k] > .5).sum() < 10 and indx.sum() < 100:
            continue
        rgb = np.clip(mpl.colormaps['plasma'](df.loc[indx, k].values)[:,:3]*255,0,255).astype(np.uint8)
        img = np.zeros((height,width,3), dtype=np.uint8)
        for i in range(3):
            img[:, :, i] = coo_array( ( rgb[:, i], (df.loc[indx, 'Y'], df.loc[indx, 'X']) ), shape=(height, width) ).toarray()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output +".F_" +k+".png",img)
        logger.info(f"Made image for {k}")

    logger.info(f"Finished")
