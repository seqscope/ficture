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

from ficture.utils import utilt
from ficture.loaders.pixel_factor_loader import BlockIndexedLoader

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
    parser.add_argument('--full', action='store_true', help="Read full input")
    parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Actual size (um) corresponding to each pixel in the output image")
    parser.add_argument('--all', action="store_true", help="Caution: when set, assume factors are named as 0, 1, ... K-1, where K is defined in the input header. Only use when plotting a small region.")

    args = parser.parse_args(_args)
    logger = logging.getLogger(__name__)
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    loader = BlockIndexedLoader(args.input, args.xmin, args.xmax, args.ymin, args.ymax, args.full)
    width = int((loader.xmax - loader.xmin + 1)/args.plot_um_per_pixel)
    height= int((loader.ymax - loader.ymin + 1)/args.plot_um_per_pixel)
    logger.info(f"Image size {height} x {width}")

    id_list = args.id_list
    if args.all:
        id_list = [str(k) for k in range(loader.meta['K'])]

    # Read input file
    df = pd.DataFrame()
    for chunk in loader:
        if chunk.shape[0] == 0:
            continue
        logger.info(f"Reading pixels... {chunk.X.iloc[-1]}, {chunk.Y.iloc[-1]}, {df.shape[0]}")
        chunk['X'] = np.clip(((chunk.X - loader.xmin) / args.plot_um_per_pixel).astype(int),0,width-1)
        chunk['Y'] = np.clip(((chunk.Y - loader.ymin) / args.plot_um_per_pixel).astype(int),0,height-1)
        for k in id_list:
            chunk[k] = np.zeros(chunk.shape[0], dtype=np.float16)
            for i in range(1, loader.meta['TOPK']+1):
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
