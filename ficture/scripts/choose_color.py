import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.neighbors
from scipy.sparse import coo_array
from datetime import datetime
from sklearn.manifold import MDS

from ficture.utils.utilt import plot_colortable
# from ficture.utils.mds_color_circle import assign_color_mds_circle

def assign_color_mds_line(mtx, cmap_name, weight=None, top_color=None, seed=None):
    # mtx is a K by K similarity/proximity matrix
    assert mtx.shape[0] == mtx.shape[1], "mtx must be square"
    K = mtx.shape[0]
    # weight is a K vector of factor abundance
    if weight is None:
        weight = np.ones(K)
    weight /= weight.sum()
    # The color of the top factor (the one with the largest weight)
    if top_color is None:
        top_color = "#fcd217"
    else:
        match = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', top_color)
        if match is None:
            top_color = "#fcd217"
    # Find the offset to map the top factor to the desired color
    cgrid = 200
    cmtx=plt.get_cmap(cmap_name)(np.arange(cgrid)/cgrid)
    h = top_color.lstrip('#')
    top_color_rgb = [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
    d = np.abs(cmtx[:, :3] - np.array(top_color_rgb).reshape((1, -1)) ).sum(axis = 1)
    anchor_pos = d.argmin() / cgrid
    anchor_angle = anchor_pos * 2 * np.pi

    mds = MDS(n_components=1, dissimilarity="precomputed", random_state=seed)
    mds_coordinates = mds.fit_transform(mtx).flatten()
    c_order = np.argsort(np.argsort(mds_coordinates))
    w_vec = weight[np.argsort(mds_coordinates)]
    w_vec = np.cumsum(w_vec) - w_vec/2
    normalized_coordinates = np.zeros(K)
    normalized_coordinates[c_order] = w_vec
    angle = normalized_coordinates * 2 * np.pi
    anchor_k = np.argmax(weight)
    angle_shift = angle + (anchor_angle - angle[anchor_k])
    if angle_shift.max() > 2*np.pi:
        angle_shift -= np.pi * 2
    angle_shift[angle_shift < 0] = 2 * np.pi + angle_shift[angle_shift < 0]
    c_pos = angle_shift / np.pi / 2
    return c_pos

def choose_color(_args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--cmap_name', type=str, default="turbo", help="Name of Matplotlib colormap to use (better close to a circular colormap)")
    parser.add_argument('--top_color', type=str, default="#fcd217", help="HEX color code for the top factor")
    parser.add_argument('--even_space', action='store_true', help="Evenly space the factors on the circle")
    parser.add_argument('--annotation', type=str, default = '', help='')
    parser.add_argument('--circle', action='store_true', help="")
    parser.add_argument('--seed', type=int, default=-1, help='')
    args = parser.parse_args(_args)

    if len(_args) == 0:
        parser.print_help()
        return

    ## obtain seed if not provided
    seed = args.seed
    if seed <= 0:
        seed = int(datetime.now().timestamp()) % 2147483648
    np.random.seed(seed)

    factor_name = {}
    if os.path.isfile(args.annotation):
        with open(args.annotation) as f:
            for line in f:
                x = line.strip().split('\t')
                factor_name[x[0]] = x[1]

    cmap_name = args.cmap_name
    if args.cmap_name not in plt.colormaps():
        cmap_name = "turbo"

    df = pd.read_csv(args.input, sep='\t', header=0)
    df.rename(columns = {"X":"x","Y":"y"},inplace=True)
    header = df.columns
    factor_header = []
    for x in header:
        y = re.match('^[A-Za-z]*_*(\d+)$', x)
        if y:
            factor_header.append(y.group(0))
    K = len(factor_header)
    N = df.shape[0]

    # Factor abundance (want top factors to have more distinct colors)
    if args.even_space:
        weight=None
    else:
        weight = df.loc[:, factor_header].sum(axis = 0).values
        weight = weight**(1/2)
        weight /= weight.sum()
        weight = np.clip(weight, .2/K, 1)
        weight /= weight.sum()

    # Find neearest neighbors
    bt = sklearn.neighbors.BallTree(df.loc[:, ["x", "y"]])
    dist, indx = bt.query(df.loc[:, ["x", "y"]], k = 7, return_distance=True)
    r_indx = np.array([i for i,v in enumerate(indx) for y in range(len(v))], dtype=int)
    c_indx = indx.reshape(-1)
    dist = dist.reshape(-1)
    nn = dist[dist > 0].min()
    mask = (dist < nn + .5) & (dist > 0)
    r_indx = r_indx[mask]
    c_indx = c_indx[mask]
    # Compute spatial similarity
    Sig = coo_array((np.ones(len(r_indx)), (r_indx, c_indx)), shape=(N, N)).tocsr()
    W = np.array(df.loc[:, factor_header])
    mtx = W.T @ Sig @ W
    # Translate into a symmetric similarity measure
    # Large values in mtx indicate close proximity, to be mapped to distinct colors
    np.fill_diagonal(mtx, 0)
    mtx /= mtx.sum(axis = 1)
    mtx = mtx + mtx.T

    if args.circle:
        # Assign color using MDS with circular constraint
        from ficture.utils.mds_color_circle import assign_color_mds_circle
        c_pos = assign_color_mds_circle(mtx, cmap_name, weight=weight, top_color=args.top_color, seed=seed)
    else:
        c_pos = assign_color_mds_line(mtx, cmap_name, weight=weight, top_color=args.top_color, seed=seed)

    # # Assign color using MDS with circular constraint
    # c_pos = assign_color_mds_circle(mtx, cmap_name, weight=weight, top_color=args.top_color, seed=seed)

    spectral_offset = .05 # avoid extremely dark colors
    c_pos = (c_pos - c_pos.min()) / (c_pos.max() - c_pos.min()) * (1 - spectral_offset) + spectral_offset

    c_rank = np.argsort(np.argsort(c_pos))
    cmtx = plt.get_cmap(cmap_name)(c_pos) # K x 4
    df = pd.DataFrame({"Name":np.arange(K).astype(str), "Color_index":c_rank})
    df = pd.concat([df, pd.DataFrame(cmtx[:, :3], columns=["R", "G", "B"])], axis=1)
    cdict = {k:cmtx[k,:] for k in range(K)}
    if len(factor_name) > 0:
        df["Annotation"] = df.Name.map(factor_name)
        cdict = {factor_name[str(k)]:cmtx[k,:] for k in range(K)}

    # Output RGB table
    f = args.output + ".rgb.tsv"
    df.to_csv(f, sep='\t', index=False)

    # Plot color bar
    fig = plot_colortable(cdict, "Factor label", sort_colors=False, ncols=4)
    f = args.output + ".cbar.png"
    fig.savefig(f, format="png", transparent=True)


if __name__ == "__main__":
    choose_color(sys.argv[1:])




# ### zz - Previous approach

# linear = MDS(n_components=1, dissimilarity="precomputed",normalized_stress='auto').fit_transform(mtx).squeeze()
# c_order = np.argsort(linear)
# c_rank  = np.argsort(c_order)

# # Uniform
# c_pos = np.linspace(.1, .9, K)

# # Weighted
# weight = df.loc[:, factor_header].sum(axis = 0)
# weight = weight**(1/4)
# weight /= weight.sum()
# weight = weight[c_order]
# c_up = np.cumsum(weight)
# c_down =  np.concatenate([[0], np.cumsum(weight[:-1]) ])
# c_pos = (c_up + c_down) / 2
# c_pos = (c_pos - c_pos[0]) / (c_pos[-1] - c_pos[0]) * .8 + .1

# cmtx = plt.get_cmap(cmap_name)(c_pos) # K x 4
# cdict = {i:cmtx[x,:] for i,x in enumerate(c_rank)}
# df = pd.DataFrame({"Name":range(K), "Color_index":c_rank})
# df = pd.concat([df, pd.DataFrame(cmtx[c_rank, :3], columns=["R", "G", "B"]) ], axis=1)

# # Allocate colors to major factors first
# weight = df.loc[:, factor_header].sum(axis = 0)
# weight /= weight.sum()
# w_order = np.argsort(weight)[::-1]
# w_cdf = np.cumsum(weight[w_order])
# k = np.arange(K)[w_cdf > .9][0]
# major_k = sorted(w_order[:k+1] )
# minor_k = sorted(w_order[k+1:] )
# c_rank_major = np.argsort( np.argsort(linear[major_k]) )
# c_rank_minor = np.argsort( np.argsort(linear[minor_k]) )
# c_pos = np.linspace(args.major_lower, args.major_upper, len(major_k))
# cmtx_major = plt.get_cmap(cmap_name)(c_pos)[c_rank_major, :]
# cdict = {x:cmtx_major[i,:] for i,x in enumerate(major_k)}
# c_pos = np.linspace(0, 1, len(minor_k))
# cmtx_minor = plt.get_cmap(cmap_name)(c_pos)[c_rank_minor, :]
# cdict.update({x:cmtx_minor[i,:] for i,x in enumerate(minor_k)} )
# cmtx = np.vstack([cmtx_major, cmtx_minor])
# df = pd.DataFrame({"Name":list(major_k)+list(minor_k), "Color_index":c_rank})
# df = pd.concat([df, pd.DataFrame(cmtx[:, :3], columns=["R", "G", "B"]) ], axis=1)
# df.sort_values(by = "Name", inplace=True)

# # This is wrong
# cmap = plt.get_cmap(cmap_name, K)
# cmtx = np.array([cmap(i) for i in range(K)] )
# c_order = np.argsort(linear)
# df = pd.DataFrame({"Name":range(K), "Color_index":c_order,\
#         "R":cmtx[c_order, 0], "G":cmtx[c_order, 1], "B":cmtx[c_order, 2]})
# cdict = {i:cmtx[x,:] for i,x in enumerate(c_order)}
