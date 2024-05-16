# Visualize pixel level factor analysis results
# Input file contains only top k factors and probabilities per pixel
# Meant to make use of the indexed input to plot for specified regions quickly

import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd
from scipy.sparse import *
import cv2

from ficture.loaders.pixel_factor_loader import BlockIndexedLoader

def plot_pixel_full(_args):

    parser = argparse.ArgumentParser(prog="plot_pixel_full")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help='Output file full path')
    parser.add_argument('--category_column', type=str, default='', help='If the input contains categorical labels instead of probabilities')
    parser.add_argument('--category_map', type=str, default='', help='If needed, map the input category to a different set of names matching that specified in the color table. The mathcing could be multiple to one, input should have two columns as "From" and "To"')
    parser.add_argument('--unmapped', type=str, default='', help='')
    parser.add_argument('--color_table', type=str, help='Pre-defined color map')
    parser.add_argument('--color_table_index_column', type=str, default='Name', help='')
    parser.add_argument('--input_rgb_uint8', action="store_true",help="If input rgb is from 0-255 instead of 0-1")
    parser.add_argument('--background', type=str, default="000000", help='')

    parser.add_argument('--xmin', type=float, default=-np.inf, help="in um")
    parser.add_argument('--ymin', type=float, default=-np.inf, help="in um")
    parser.add_argument('--xmax', type=float, default=np.inf, help="in um")
    parser.add_argument('--ymax', type=float, default=np.inf, help="in um")
    parser.add_argument('--full', action='store_true', help="Read full input")
    parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Actual size (um) corresponding to each pixel in the output image")
    parser.add_argument('--org_coord', action='store_true', help="If the input coordinates do not include the offset (if your coordinates are from an existing figure, the offset is already factored in)")
    parser.add_argument('--plot_top', action='store_true', help="Plot top factor only")
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args(_args)
    if len(_args) == 0:
        parser.print_help()
        return

    logging.basicConfig(level= getattr(logging, "INFO", None), format='%(asctime)s %(message)s', datefmt='%I:%M:%S %p')

    # Read color table
    rgb=['B','G','R'] # opencv rgb order
    args.background = args.background.lstrip('#')
    match = re.search(r'^(?:[0-9a-fA-F]{3}){1,2}$', args.background)
    if not match:
        logging.warning(f"Invalid background color {args.background}")
        args.background = "000000"
    logging.info(f"Background color {args.background}")
    cdty = {x:float for x in rgb}
    cdty[args.color_table_index_column] = str
    sep=',' if args.color_table.endswith(".csv") else '\t'
    color_info = pd.read_csv(args.color_table, sep=sep, header=0, index_col=args.color_table_index_column, dtype=cdty)
    if args.input_rgb_uint8 or color_info[rgb].max().max() > 2:
        for c in rgb:
            color_info[c] = color_info[c] / 255
    # color_info.index = color_info.index.astype(str)
    logging.info(f"Read color table ({color_info.shape[0]})")
    print(color_info.index)
    dty = {}
    if args.category_column != '':
        dty[args.category_column] = str

    loader = BlockIndexedLoader(args.input, args.xmin, args.xmax, args.ymin, args.ymax, args.full, not args.org_coord, idtype=dty)
    width = int((loader.xmax - loader.xmin + 1)/args.plot_um_per_pixel)
    height= int((loader.ymax - loader.ymin + 1)/args.plot_um_per_pixel)
    logging.info(f"Image size {height} x {width}")
    print(loader.header)

    categorical = False
    if args.category_column != '':
        if args.category_column not in loader.header:
            sys.exit(f"ERROR: {args.category_column} not found in input file")
        categorical = True
        logging.info("Input is categorical")

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
        logging.info(f"Read category map ({len(category_map)}), {K} categories")
        if args.debug:
            print(category_list)

    # Read input file, fill the rgb matrix
    img = np.zeros((height,width,3), dtype=np.uint8)
    bg = args.background
    bg = [ np.uint8(int(bg[i:i+2], 16) ) for i in [0,2,4] ]
    for c in range(3):
        img[:,:,c] = bg[c]
    keptcol = ['X','Y'] + rgb
    for df in loader:
        if df.shape[0] == 0:
            continue
        df['X'] = np.clip(((df.X - loader.xmin) / args.plot_um_per_pixel).astype(int),0,width-1)
        df['Y'] = np.clip(((df.Y - loader.ymin) / args.plot_um_per_pixel).astype(int),0,height-1)
        if categorical:
            if args.debug:
                print(df.loc[~df[args.category_column].isin(category_map), :][args.category_column].unique() )
            if category_rename:
                df[args.category_column] = df[args.category_column].map(category_map)
                if args.unmapped != '':
                    df[args.category_column].fillna(args.unmapped, inplace=True)
                else:
                    df = df.loc[~df[args.category_column].isna(), :]
                if args.debug:
                    print(df.shape[0], df[args.category_column].value_counts())
            else:
                df = df.loc[df[args.category_column].isin(color_info.index), :]
            for c in rgb:
                df[c] = color_info.loc[df[args.category_column].values, c].values
            if args.debug:
                print(df[args.category_column].value_counts())
        else:
            if args.plot_top:
                for c in rgb:
                    df[c] = color_info.loc[df['K1'].values, c].values
            else:
                for c in rgb:
                    df[c] = 0
                    for k in range(1,loader.meta['TOPK']+1):
                        df[c] += color_info.loc[df['K'+str(k)].values, c].values * df['P'+str(k)].values
        df = df.groupby(by=['X','Y']).agg({c:"mean" for c in rgb}).reset_index()
        logging.info(f"Reading pixels... {df.X.iloc[-1]}, {df.Y.iloc[-1]}, {df.shape[0]}")
        for i,c in enumerate(rgb):
            df[c] = np.clip(np.around(df[c] * 255),0,255).astype(np.uint8)
            img[df.Y.values, df.X.values, [i]*df.shape[0]] = df[c].values
        if args.debug:
            break

    if not args.output.endswith(".png"):
        args.output += ".png"
    cv2.imwrite(args.output,img)
    logging.info(f"Finished\n{args.output}")

if __name__ == "__main__":
    plot_pixel_full(sys.argv[1:])
