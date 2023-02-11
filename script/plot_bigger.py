import sys, os, copy, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
import png

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *
from utilt import plot_colortable
from image_fn import ImgRowIterator_stream as RowIterator

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help="Input file has to be sorted according to the y-axis (if --horizontal_axis x, default) or the x-axis if --horizontal_axis y")
parser.add_argument('--output', type=str, help='Output prefix')
parser.add_argument('--xmin', type=float, help="")
parser.add_argument('--ymin', type=float, help="")
parser.add_argument('--xmax', type=float, help="")
parser.add_argument('--ymax', type=float, help="")
parser.add_argument('--horizontal_axis', type=str, default="x", help="Which coordinate is horizontal, x or y")
parser.add_argument('--color_table', type=str, default='', help='Pre-defined color map')
parser.add_argument('--cmap_name', type=str, default="turbo", help="Name of Matplotlib colormap to use")
parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Size of the output pixels in um")
parser.add_argument('--chunksize', type=float, default=1e6, help="How many lines to read from input file at a time")
parser.add_argument("--plot_discretized", action='store_true', help="")

args = parser.parse_args()
logging.basicConfig(level= getattr(logging, "INFO", None))
haxis = args.horizontal_axis.lower()
if haxis not in ["x", "y"]:
    sys.exit("Unrecognized --horizontal_axis")
chunksize = int(args.chunksize)

with gzip.open(args.input, "rt") as rf:
    header = rf.readline().strip().split('\t')
    header = [x.strip() for x in header]
# Temporary - to be compatible with older input files
recolumn = {'Hex_center_x':'x', 'Hex_center_y':'y', 'X':'x', 'Y':'y'}
for i,x in enumerate(header):
    if x in recolumn:
        header[i] = recolumn[x]
if args.horizontal_axis == "y": # Transpose
    for i,x in enumerate(header):
        if x == "x":
            header[i] = "y"
        if x == "y":
            header[i] = "x"
factor_header = []
for i,x in enumerate(header):
    # Dangerous way to detect which columns to use as factor loadings
    y = re.match('^[A-Za-z]*_*(\d+)$', x)
    if y:
        header[i] = y.group(1)
        factor_header.append(y.group(1))
K = len(factor_header)

# Colormap
if os.path.exists(args.color_table):
    color_info = pd.read_csv(args.color_table, sep='\t', header=0)
    cmtx = np.array(color_info.loc[:, ["R","G","B"]])
else:
    cmap_name = args.cmap_name
    if args.cmap_name not in plt.colormaps():
        cmap_name = "turbo"
    cmap = plt.get_cmap(cmap_name, K)
    cmtx = np.array([cmap(i) for i in range(K)] )
    indx = np.arange(K)
    shuffle(indx)
    cmtx = cmtx[indx, :3]
    cdict = {k:cmtx[k,:] for k in range(K)}
    cmtx_df = pd.DataFrame(cmtx, columns = ["R","G","B"])
    cmtx_df["Name"] = indx
    f = args.output + ".rgb.tsv"
    cmtx_df.to_csv(f, sep='\t', index=False, header=True)
    # Plot color bar separately
    fig = plot_colortable(cdict, "Factor label", sort_colors=False, ncols=4)
    f = args.output + ".cbar"
    fig.savefig(f, format="png")
    logging.info(f"Set up color map for {K} factors")


adt={x:float for x in ["x","y"]+factor_header}
reader = pd.read_csv(gzip.open(args.input, 'rt'), sep='\t', \
                     chunksize=chunksize, skiprows=1, names=header, dtype=adt)
h = args.ymax - args.ymin
w = args.xmax - args.xmin
outf = args.output + ".png"
obj  = RowIterator(reader, w, h, cmtx, xmin = args.xmin, ymin = args.ymin,\
        pixel_size = args.plot_um_per_pixel, verbose=1000, dtype=np.uint8, plot_top=args.plot_discretized)
wpng = png.Writer(size=(obj.width, obj.height),\
                  greyscale=False,bitdepth=8,planes=3)
with open(outf, 'wb') as f:
    wpng.write(f, obj)
logging.info(f"Made image\n{outf}")
