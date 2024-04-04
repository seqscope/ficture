# Visualize pixel level factor analysis results
# <= 5 (preferably 3) factors at a time, with specified colors
# Input file contains only top k factors and probabilities per pixel
# Meant to make use of the indexed input to plot for specified regions quickly

import sys, os, copy, re, gzip, argparse, logging, warnings
import numpy as np
import pandas as pd
from scipy.sparse import *
import cv2

from ficture.loaders.pixel_factor_loader import BlockIndexedLoader

def plot_pixel_multi(_args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help='Output file full path')
    parser.add_argument('--channel_list', type=str, nargs='+', help="Select up to 3 channels/factors to visualize")
    parser.add_argument('--color_list', type=str, nargs='*', default=["144A74", "FF9900", "DD65E6", "FFEC11"], help='') # blue, orange, purple, yellow
    parser.add_argument('--color_table', type=str, default='', help='Pre-defined color map')
    parser.add_argument('--color_table_index_column', type=str, default='Name', help='')
    parser.add_argument('--input_rgb_uint8', action="store_true",help="If input rgb is from 0-255 instead of 0-1")

    parser.add_argument('--xmin', type=float, default=-np.inf, help="")
    parser.add_argument('--ymin', type=float, default=-np.inf, help="")
    parser.add_argument('--xmax', type=float, default=np.inf, help="")
    parser.add_argument('--ymax', type=float, default=np.inf, help="")
    parser.add_argument('--pcut', type=float, default=0.01, help="")
    parser.add_argument('--spcut', type=float, default=0.1, help="")
    parser.add_argument('--org_coord', action='store_true', help="If the input coordinates do not include the offset (if your coordinates are from an existing figure, the offset is already factored in)")
    parser.add_argument('--full', action='store_true', help="")
    parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Actual size (um) corresponding to each pixel in the output image")
    parser.add_argument('--categorical', action='store_true', help="Plot top factor for each pixel categorically, without probability or mixture")
    parser.add_argument('--debug', action='store_true', help="")

    args = parser.parse_args(_args)
    if len(_args) == 0:
        parser.print_help()
        return

    logging.basicConfig(level= getattr(logging, "INFO", None), format='%(asctime)s %(message)s', datefmt='%I:%M:%S %p')

    rgb=list("RGB")
    bgr=list("BGR") # opencv default channel order
    bgra=list("BGRA") # opencv bgra
    pcut = 0.01
    spcut = 0.1
    if len(args.channel_list) > 5:
        logging.warning("Be colorblind friendly")
    channels = args.channel_list
    # Read color table
    if os.path.exists(args.color_table):
        color_info = pd.read_csv(args.color_table, sep='\t', header=0, index_col=args.color_table_index_column)
        if color_info[rgb].max().max() > 1.1 and not args.input_rgb_uint8:
            logging.warning("Color table should contain RGB values in 0-1 range, but found values > 1, will assume unit8")
            args.input_rgb_uint8 = True
        if args.input_rgb_uint8:
            for c in rgb:
                color_info[c] = np.clip(color_info[c]/255,0,1)
        color_info.index = color_info.index.astype(str)
        channels = [x for x in channels if x in color_info.index]
        if len(channels) != len(args.channel_list):
            logging.error("Some channels are not found in the input color table")
            sys.exit(1)
        color_info = {i: np.array(color_info.loc[i, rgb]) for i in channels}
    else:
        if len(args.color_list) < len(channels):
            logging.error("Number of input colors and channels do not match")
            sys.exit(1)
        color_list = [[int(x.strip('#')[i:i+2],16)/255 for i in (0,2,4)] for x in args.color_list]
        color_info = {x:np.array(color_list[i]) for i,x in enumerate(channels)}
        ctab = pd.DataFrame(color_info, index=rgb).T
        ctab['Name'] = ctab.index
        if not args.output.endswith(".png"):
            f = args.output + ".color_table.tsv"
        else:
            f = args.output.replace(".png", ".color_table.tsv")
        ctab.to_csv(f, sep='\t', header=True, index=False)

    logging.info(f"Set up color map {color_info}")

    loader = BlockIndexedLoader(args.input, args.xmin, args.xmax, args.ymin, args.ymax, args.full, not args.org_coord)
    width = int((loader.xmax - loader.xmin + 1)/args.plot_um_per_pixel)
    height= int((loader.ymax - loader.ymin + 1)/args.plot_um_per_pixel)
    logging.info(f"Image size {height} x {width}")

    # Read input file, fill the rgb matrix
    img = np.zeros((height,width,4), dtype=np.uint8)
    img[:,:,3] = 255
    clps = {x:np.mean for x in rgb}
    clps['A'] = np.max
    for df in loader:
        if df.shape[0] == 0:
            continue
        indx = np.zeros(df.shape[0], dtype=bool)
        if args.categorical:
            indx = df.K1.isin(channels)
        else:
            for k in range(1, loader.meta['TOPK']):
                indx = indx | (df[f"K{k}"].isin(channels) & df[f"P{k}"].gt(pcut))
        df = df.loc[indx, :]
        if args.debug:
            print(df.shape[0])
        if df.shape[0] == 0:
            continue
        df['X'] = np.clip(((df.X - loader.xmin) / args.plot_um_per_pixel).astype(int),0,width-1)
        df['Y'] = np.clip(((df.Y - loader.ymin) / args.plot_um_per_pixel).astype(int),0,height-1)
        df[bgra] = 0
        if args.categorical:
            df.loc[:,rgb] = np.array(list(df.K1.map(color_info)) )
            df['A'] = 1
        else:
            for k in range(1, loader.meta['TOPK']):
                indx = df[f"K{k}"].isin(channels)
                if indx.sum() == 0:
                    continue
                df.loc[indx, rgb] += np.array(list(df.loc[indx, f"K{k}"].map(color_info))) * df.loc[indx, f"P{k}"].values.reshape((-1,1))
                df.loc[indx, 'A'] += df.loc[indx, f"P{k}"]
        if args.debug:
            print(df.shape[0], np.median(df.A), np.mean(df.A), df.A.gt(spcut).sum())
        df = df.groupby(by=['X','Y']).agg(clps).reset_index()
        df = df.loc[df.A.gt(spcut), :]
        df[bgra] = np.clip(np.around(df[bgra] * 255),0,255).astype(np.uint8)
        logging.info(f"Reading pixels... {df.X.iloc[-1]}, {df.Y.iloc[-1]}, {df.shape[0]}")
        for i,c in enumerate(bgra):
            img[df.Y.values, df.X.values, [i]*df.shape[0]] = df[c].values
        if args.debug:
            break

    if not args.output.endswith(".png"):
        args.output += ".png"
    cv2.imwrite(args.output,img)
    logging.info(f"Finished\n{args.output}")


if __name__ == "__main__":
    plot_pixel_multi(sys.argv[1:])
