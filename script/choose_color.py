import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilt import plot_colortable

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--cmap_name', type=str, default="turbo", help="Name of Matplotlib colormap to use")
args = parser.parse_args()

df = pd.read_csv(args.input, sep='\t', header=0)
header = df.columns
factor_header = []
for x in header:
    y = re.match('^[A-Za-z]+_\d+$', x)
    if y:
        factor_header.append(y.group(0))
K = len(factor_header)

try:
    f = os.path.split(os.path.dirname(args.input))[0] + "/coordinate_minmax.tsv"
    if not os.path.exists(f):
        xmin, ymin = df[["x", "y"]].min(axis = 0)
        xmax, ymax = df[["x", "y"]].max(axis = 0)
        line = ["xmin\t"+str(xmin), "xmax\t"+str(xmax),\
                "ymin\t"+str(ymin), "ymax\t"+str(ymax)]
        line = "\n".join(line) + "\n"
        with open(f, "w") as wf:
            wf.write(line)
except:
    print("Did not output coordinate limits, probabily because the input file does not contain header x and y")

# Colormap
cmap_name = args.cmap_name
if args.cmap_name not in plt.colormaps():
    cmap_name = "turbo"
cmap = plt.get_cmap(cmap_name, K)
cmtx = np.array([cmap(i) for i in range(K)] )

mtx = np.array(df.loc[:, factor_header].corr())
linear = MDS(n_components=1, dissimilarity="precomputed").fit_transform(1-mtx).squeeze()
c_order = np.argsort(linear)

# Output RGB table
df = pd.DataFrame({"Name":range(K), "Color_index":c_order,\
        "R":cmtx[c_order, 0], "G":cmtx[c_order, 1], "B":cmtx[c_order, 2]})
f = args.output + ".rgb.tsv"
df.to_csv(f, sep='\t', index=False)

# Plot color bar
cdict = {i:cmtx[x,:] for i,x in enumerate(c_order)}
fig = plot_colortable(cdict, "Factor label", sort_colors=False, ncols=4)
f = args.output + ".cbar"
fig.savefig(f, format="png")
