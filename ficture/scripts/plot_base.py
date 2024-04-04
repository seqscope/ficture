import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd
from random import shuffle

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from scipy.sparse import *
import sklearn.neighbors
import sklearn.preprocessing

from ficture.utils.hexagon_fn import *
from ficture.utils.utilt import plot_colortable

def plot_base(_args):

    parser = argparse.ArgumentParser(prog = "plot_base")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help='Output prefix')
    parser.add_argument('--fill_range', type=float, default=0, help="um")
    parser.add_argument('--batch_size', type=float, default=500, help="")
    parser.add_argument("--tif", action='store_true', help="Store as 16-bit tif instead of png")
    parser.add_argument('--scale', type=float, default=-1, help="")
    parser.add_argument('--origin', type=int, default=[0,0], help="{0, 1} x {0, 1}, specify how to orient the image w.r.t. the coordinates. (0, 0) means the lower left corner has the minimum x-value and the minimum y-value; (0, 1) means the lower left corner has the minimum x-value and the maximum y-value;")
    parser.add_argument('--category_column', type=str, default='', help='')

    parser.add_argument('--color_table_category_name', type=str, default='Name', help='When --category_column is provided, which column to use as the category name')
    parser.add_argument('--binary_cmap_name', type=str, default="plasma", help="Name of Matplotlib colormap to use for ploting individual factors")
    parser.add_argument('--color_table', type=str, default='', help='Pre-defined color map')
    parser.add_argument('--input_rgb_uint8', action="store_true",help="If input rgb is from 0-255 instead of 0-1")
    parser.add_argument('--cmap_name', type=str, default="turbo", help="Name of Matplotlib colormap, only used when --color_table is not provided")

    parser.add_argument("--plot_fit", action='store_true', help="")
    parser.add_argument('--xmin', type=float, default=-np.inf, help="")
    parser.add_argument('--ymin', type=float, default=-np.inf, help="")
    parser.add_argument('--xmax', type=float, default=np.inf, help="")
    parser.add_argument('--ymax', type=float, default=np.inf, help="")
    parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Size of the output pixels in um")

    parser.add_argument("--skip_mixture_plot", action='store_true', help="")
    parser.add_argument("--plot_discretized", action='store_true', help="")
    parser.add_argument("--plot_individual_factor", action='store_true', help="")
    parser.add_argument("--debug", action='store_true', help="")

    args = parser.parse_args(_args)
    if len(_args) == 0:
        parser.print_help()
        return

    logging.basicConfig(level= getattr(logging, "INFO", None))
    dt = np.uint16 if args.tif else np.uint8
    kcol = args.category_column
    ccol = args.color_table_category_name

    # Dangerous way to detect which columns to use as factor loadings
    with gzip.open(args.input, "rt") as rf:
        header = rf.readline().strip().split('\t')
    # Temporary - to be compatible with older input files
    recolumn = {'Hex_center_x':'x', 'Hex_center_y':'y', 'X':'x', 'Y':'y'}
    for i,x in enumerate(header):
        if x in recolumn:
            header[i] = recolumn[x]
    factor_header = []
    categorical = False
    if args.category_column != '':
        if args.category_column not in header:
            sys.exit(f"ERROR: {args.category_column} not found in header")
        categorical = True
        if not os.path.exists(args.color_table):
            sys.exit(f"ERROR: --color_table is required for categorical input")
        color_info = pd.read_csv(args.color_table, sep='\t', header=0, dtype={ccol:str})
        if args.input_rgb_uint8 or color_info[["R","G","B"]].max().max() > 2:
            for c in list("RGB"):
                color_info[c] = color_info[c] / 255
        color_info = color_info[~color_info[ccol].isna()]
        color_idx = {x:i for i,x in enumerate(color_info[ccol].values)}
        cmtx = np.array(color_info.loc[:, ["R","G","B"]])
        K = len(color_idx)
        factor_header = [str(k) for k in range(K)]
        logging.info(f"Use categorical input ({K} categories)")
    else:
        for x in header:
            y = re.match('^[A-Za-z]*_*(\d+)$', x)
            if y:
                factor_header.append([y.group(0), int(y.group(1)) ])
        factor_header.sort(key = lambda x : x[1] )
        factor_header = [x[0] for x in factor_header]
        K = len(factor_header)
        if os.path.exists(args.color_table):
            color_info = pd.read_csv(args.color_table, sep='\t', header=0)
            if args.input_rgb_uint8 or color_info[["R","G","B"]].max().max() > 2:
                for c in list("RGB"):
                    color_info[c] = color_info[c] / 255
            cmtx = np.array(color_info.loc[:, ["R","G","B"]])
        else:
            cmap_name = args.cmap_name
            if args.cmap_name not in plt.colormaps():
                cmap_name = "turbo"
            cmap = plt.get_cmap(cmap_name, K)
            cmtx = np.array([cmap(i) for i in range(K)] )
            indx = np.arange(K)
            shuffle(indx)
            cmtx = cmtx[indx, ]
            cmtx = cmtx[:, :3]
            cdict = {k:cmtx[k,:] for k in range(K)}
            # Plot color bar separately
            fig = plot_colortable(cdict, "Factor label", sort_colors=False, ncols=4)
            f = args.output + ".cbar"
            fig.savefig(f, format="png")
            logging.info(f"Set up color map for {K} factors")

    # Read data
    adt={x:float for x in ["x", "y"]+factor_header}
    adt[kcol] = str
    df = pd.DataFrame()
    for chunk in pd.read_csv(gzip.open(args.input, 'rt'), sep='\t', \
        chunksize=1000000, skiprows=1, names=header, dtype=adt):
        if args.scale > 0:
            chunk.x = chunk.x / args.scale
            chunk.y = chunk.y / args.scale
        chunk = chunk[(chunk.y > args.ymin) & (chunk.y < args.ymax)]
        chunk = chunk[(chunk.x > args.xmin) & (chunk.x < args.xmax)]
        chunk['x_indx'] = np.round(chunk.x.values / args.plot_um_per_pixel, 0).astype(int)
        chunk['y_indx'] = np.round(chunk.y.values / args.plot_um_per_pixel, 0).astype(int)
        if categorical: # Add dummy columns
            chunk[kcol] = chunk[kcol].map(color_idx).astype(int)
            for k in range(K):
                chunk[str(k)] = chunk[kcol].eq(k).astype(int)
        chunk = chunk.groupby(by = ['x_indx', 'y_indx']).agg({ x:np.mean for x in factor_header }).reset_index()
        df = pd.concat([df, chunk])

    df = df.groupby(by = ['x_indx', 'y_indx']).agg({ x:np.mean for x in factor_header }).reset_index()
    x_indx_min = args.xmin / args.plot_um_per_pixel
    y_indx_min = args.ymin / args.plot_um_per_pixel
    if args.plot_fit or np.isinf(args.xmin):
        df.x_indx = df.x_indx - int(np.max([x_indx_min, df.x_indx.min()]) )
    else:
        df.x_indx -= int(x_indx_min)
    if args.plot_fit or np.isinf(args.ymin):
        df.y_indx = df.y_indx - int(np.max([y_indx_min, df.y_indx.min()]) )
    else:
        df.y_indx -= int(y_indx_min)

    N0 = df.shape[0]
    df.index = range(N0)
    hsize, wsize = df[['x_indx','y_indx']].max(axis = 0) + 1
    hsize_um = hsize * args.plot_um_per_pixel
    wsize_um = wsize * args.plot_um_per_pixel
    logging.info(f"Read region {N0} pixels in region {hsize_um} x {wsize_um}")


    # Make images
    wst = df.y_indx.min()
    wed = df.y_indx.max()
    wstep = np.max([10, int(args.batch_size)])
    radius = args.fill_range/args.plot_um_per_pixel
    print(wsize, wstep)


    if radius <= 0:
        pts = df[['x_indx', 'y_indx']].values
        pts_indx = df.index.values
    else:
        pts = np.zeros((0, 2), dtype=int)
        pts_indx = []
        st = wst
        while st < wed:
            ed = min([st + wstep, wed])
            logging.info(f"Filling pixels {st} - {ed} / {wed}")
            if ((df.y_indx > st) & (df.y_indx < ed)).sum() < 10:
                st = ed
                continue
            block = df.index[(df.y_indx > st - 2*radius) & (df.y_indx < ed + 2*radius)]
            ref = sklearn.neighbors.BallTree(np.array(df.loc[block, ['x_indx', 'y_indx']]))
            mesh = np.meshgrid(np.arange(hsize), np.arange(st, ed))
            nodes = np.array(list(zip(*(dim.flat for dim in mesh))), dtype=int)
            dv, iv = ref.query(nodes, k = 1, dualtree=True)
            indx = dv[:, 0] < radius
            iv = block[iv[indx, 0] ]
            if sum(indx) == 0:
                st = ed
                continue
            pts = np.vstack((pts, nodes[indx, :]) )
            pts_indx += list(iv)
            st = ed
            if args.debug:
                break

    # Note: PIL default origin is upper-left
    pts[:,0] = np.clip(hsize - pts[:, 0], 0, hsize-1)
    pts[:,1] = np.clip(pts[:, 1], 0, wsize-1)
    if args.origin[0] > 0:
        pts[:,0] = hsize - 1 - pts[:, 0]
    if args.origin[1] > 0:
        pts[:,1] = wsize - 1 - pts[:, 1]

    logging.info(f"Start constructing RGB image")

    if not args.skip_mixture_plot:
        rgb_mtx = np.clip(np.around(np.array(df.loc[pts_indx,factor_header]) @\
                                    cmtx * 255),0,255).astype(dt)
        img = np.zeros( (hsize, wsize, 3), dtype=dt)
        for r in range(3):
            img[:, :, r] = coo_array((rgb_mtx[:, r], (pts[:,0], pts[:,1])),\
                shape=(hsize, wsize), dtype = dt).toarray()
        if args.tif:
            img = Image.fromarray(img, mode="I;16")
        else:
            img = Image.fromarray(img)

        outf = args.output
        outf += ".tif" if args.tif else ".png"
        img.save(outf)
        logging.info(f"Made fractional image\n{outf}")

    if args.plot_discretized:
        kvec = np.array(df.loc[pts_indx,factor_header]).argmax(axis = 1)
        cmtx = np.clip(np.around(cmtx * 255), 0, 255).astype(dt)
        img = np.zeros( (hsize, wsize, 3), dtype=dt)
        for r in range(3):
            img[:, :, r] = coo_array((cmtx[kvec, r], (pts[:,0], pts[:,1])),\
                shape=(hsize, wsize), dtype = dt).toarray()
        if args.tif:
            img = Image.fromarray(img, mode="I;16")
        else:
            img = Image.fromarray(img)

        outf = args.output + ".top"
        outf += ".tif" if args.tif else ".png"
        img.save(outf)
        logging.info(f"Made hard threshold image\n{outf}")

    if args.plot_individual_factor:
        if args.binary_cmap_name not in plt.colormaps():
            args.binary_cmap_name = "plasma"
        v = np.array(df.loc[pts_indx, factor_header].sum(axis = 0) )
        u = np.argsort(-v)
        for k in u:
            v = np.clip(df.loc[pts_indx, factor_header[k]].values,0,1)
            rgb_mtx = np.clip(mpl.colormaps[args.binary_cmap_name](v)[:,:3]*255,0,255).astype(dt)
            img = np.zeros( (hsize, wsize, 3), dtype=dt)
            for r in range(3):
                img[:, :, r] = coo_array((rgb_mtx[:, r], (pts[:,0], pts[:,1])),\
                    shape=(hsize, wsize), dtype = dt).toarray()
            if args.tif:
                img = Image.fromarray(img, mode="I;16")
            else:
                img = Image.fromarray(img)
            outf = args.output + ".F_"+str(k)
            outf += ".tif" if args.tif else ".png"
            img.save(outf)
            logging.info(f"Made factor specific image - {k}\n{outf}")

if __name__ == "__main__":
    plot_base(sys.argv[1:])
