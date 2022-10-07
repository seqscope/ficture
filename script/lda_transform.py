import sys, os, copy, gc, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd
from random import shuffle

import matplotlib.pyplot as plt
from PIL import Image

from scipy.sparse import *
import sklearn.neighbors
import sklearn.preprocessing

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *
from utilt import plot_colortable

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--model', type=str, help='')
parser.add_argument('--output_path', type=str, help='')
parser.add_argument('--output_id', type=str, help='')
parser.add_argument('--region_id', type=str, help='lane')
parser.add_argument('--log', default = '', type=str, help='files to write log to')

parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
parser.add_argument('--buffer_step', type=int, default=2000, help='um')
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--min_ct_per_unit', type=int, default=10, help='')
parser.add_argument('--n_move', type=int, default=-1, help='')
parser.add_argument('--hex_width', type=int, default=18, help='')
parser.add_argument('--hex_radius', type=int, default=-1, help='')

parser.add_argument('--skip_plot', action='store_true')
parser.add_argument('--cmap_name', type=str, default="turbo", help="Name of Matplotlib colormap to use")
parser.add_argument('--plot_um_per_pixel', type=float, default=1, help="Size of the output pixels in um")
parser.add_argument('--fill_range', type=float, default=-1, help="um")
# parser.add_argument('--chunk_size', type=int, default=5000, help="um")
parser.add_argument("--plot_top", default=False, action='store_true', help="")
parser.add_argument("--plot_individual_factor", default=False, action='store_true', help="")
parser.add_argument("--tif", default=False, action='store_true', help="Store as 16-bit tif instead of png")
parser.add_argument("--overwrite", default=False, action='store_true', help="")

args = parser.parse_args()
if args.log != '':
    try:
        logging.basicConfig(filename=args.log, filemode='a', encoding='utf-8', level=logging.INFO)
    except:
        logging.basicConfig(level= getattr(logging, "INFO", None))
else:
    logging.basicConfig(level= getattr(logging, "INFO", None))

mu_scale = 1./args.mu_scale
key = args.key

### Input and output
if not os.path.exists(args.input) or not os.path.exists(args.model):
    sys.exit("ERROR: cannot find input files.")

figure_path = args.output_path + "/analysis/figure"
if not os.path.exists(figure_path):
    arg="mkdir -p "+figure_path
    os.system(arg)

lda = pickle.load( open( args.model, "rb" ) )
feature_kept = lda.feature_names_in_
ft_dict = {x:i for i,x in enumerate( feature_kept ) }
K, M = lda.exp_dirichlet_component_.shape
factor_header = ['k_'+str(x) for x in range(K)]
lda.feature_names_in_ = None

b_size = 512
diam = args.hex_width
radius = args.hex_radius
if radius < 0:
    radius = diam / np.sqrt(3)
else:
    diam = radius*np.sqrt(3)
n_move = args.n_move
if n_move >= diam or n_move < 0:
    n_move = diam // 4
fill_range = max([args.fill_range, radius])
ovlp_buffer = diam * 2

res_f = args.output_path + "/analysis/"+args.output_id+"_"+str(int(diam))+".fit_result.tsv"
dtp = {x:int for x in ['offs_x','offs_y','hex_x','hex_y','topK']}
dtp.update({x:float for x in ['topP','x','y']+factor_header})

do_transform = True
if os.path.exists(res_f + ".gz") and not args.overwrite:
    f = res_f + ".gz"
    try:
        df = pd.read_csv(gzip.open(f,'rt'),header=0,sep='\t', dtype=dtp)
        do_transform = False
        logging.info(f"Use existing file {f}")
    except:
        logging.info(f"Overwrite invalid file {f}")

if do_transform:
    # Apply fitted model
    df = pd.DataFrame()
    nbatch = 0
    for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=1000000, header=0, usecols=["X","Y","gene",key],dtype={'X':int,'Y':int,'gene':str,key:int}):
        chunk['j'] = chunk.X.astype(str).values + "_" + chunk.Y.astype(str).values
        chunk = chunk[chunk.gene.isin(feature_kept)]
        if chunk.shape[0] == 0:
            continue
        df = pd.concat((df, chunk))
        st = df.Y.min() * mu_scale
        ed = df.Y.max() * mu_scale
        print(st, ed, df.Y.min(), df.Y.max(), chunk.Y.min(), chunk.Y.max())
        if ed - st < args.buffer_step:
            continue
        brc = df.groupby(by = ['j','X','Y']).agg({key: sum}).reset_index()
        brc.index = range(brc.shape[0])
        brc['X'] = brc.X.values * mu_scale
        brc['Y'] = brc.Y.values * mu_scale
        # Make DGE
        barcode_kept = list(brc.j.values)
        df = df[df.j.isin(barcode_kept)]
        bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
        indx_row = [ bc_dict[x] for x in df['j']]
        indx_col = [ ft_dict[x] for x in df['gene']]
        N = len(barcode_kept)
        dge_mtx = coo_matrix((df[key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
        logging.info(f"Made DGE {dge_mtx.shape}")
        offs_x = 0
        offs_y = 0
        while offs_x < n_move:
            while offs_y < n_move:
                x,y = pixel_to_hex(np.array(brc[['X','Y']]), radius, offs_x/n_move, offs_y/n_move)
                hex_pt  = pd.DataFrame({'hex_x':x,'hex_y':y,'ct':brc[key].values}).groupby(by=['hex_x','hex_y']).agg({'ct':np.sum}).reset_index()
                hex_pt['x'], hex_pt['y'] = hex_to_pixel(hex_pt.hex_x.values, hex_pt.hex_y.values, radius, offs_x/n_move, offs_y/n_move)
                hex_pt = hex_pt.loc[ (hex_pt.y > st + diam/2) & (hex_pt.y < ed - diam/2) & (hex_pt.ct >= args.min_ct_per_unit), :]
                if hex_pt.shape[0] < 2:
                    offs_y += 1
                    continue
                hex_list = list(zip(hex_pt.hex_x.values, hex_pt.hex_y.values))
                hex_crd = list(zip(x,y))
                hex_dict = {x:i for i,x in enumerate(hex_list)}
                indx = [i for i,x in enumerate(hex_crd) if x in hex_dict]
                hex_crd = [hex_crd[i] for i in indx]
                sub = pd.DataFrame({'cRow':[hex_dict[x] for x in hex_crd], 'cCol':indx, 'hexID':hex_crd})
                nunit = len(hex_dict)
                n_pixel = sub.shape[0]
                mtx = coo_matrix((np.ones(n_pixel, dtype=bool),\
                        (sub.cRow.values, sub.cCol.values)),\
                        shape=(nunit,N) ).tocsr() @ dge_mtx
                logl = lda.score(mtx) / mtx.shape[0]
                theta = lda.transform(mtx)
                lines = pd.DataFrame({'offs_x':offs_x,'offs_y':offs_y, 'hex_x':hex_pt.hex_x.values, 'hex_y':hex_pt.hex_y.values})
                lines['x'], lines['y'] = hex_to_pixel(hex_pt.hex_x.values,hex_pt.hex_y.values, radius, offs_x/n_move, offs_y/n_move)
                lines = pd.concat((lines, pd.DataFrame(theta, columns = factor_header)), axis = 1)
                lines['topK'] = np.argmax(theta, axis = 1).astype(int)
                lines['topP'] = np.max(theta, axis = 1)
                lines = lines.astype(dtp)
                if nbatch == 0:
                    lines.to_csv(res_f, sep='\t', mode='w', float_format="%.5f", index=False, header=True)
                else:
                    lines.to_csv(res_f, sep='\t', mode='a', float_format="%.5f", index=False, header=False)
                logging.info(f"Batch {nbatch} from {int(st)} to {int(ed)} with {nunit} units, log likelihood {logl:.3f}")
                nbatch += 1
                print(nbatch, offs_x, offs_y)
                offs_y += 1
            offs_y = 0
            offs_x += 1
        df = df[df.Y > ed - ovlp_buffer]

    gc.collect()

    if args.skip_plot:
        sys.exit()

    df = pd.read_csv(res_f,header=0,sep='\t', dtype=dtp)

### Make figure
dt = np.uint16 if args.tif else np.uint8
cmap_name = args.cmap_name
if args.cmap_name not in plt.colormaps():
    cmap_name = "turbo"
cmap = plt.get_cmap(cmap_name, K)
cmtx = np.array([cmap(i) for i in range(K)] )
indx = np.arange(K)
shuffle(indx)
cmtx = cmtx[indx, ]
cmtx = cmtx[:, :3]
print(cmtx)
cdict = {k:cmtx[k,:] for k in range(K)}

# Plot color bar separately
fig = plot_colortable(cdict, "Factor label", sort_colors=False, ncols=4)
f = figure_path + "/"+args.output_id+"_"+str(int(diam))+".cbar.png"
fig.savefig(f)

df['x_indx'] = np.round(df.x.values / args.plot_um_per_pixel, 0).astype(int)
df['y_indx'] = np.round(df.y.values / args.plot_um_per_pixel, 0).astype(int)
df = df.groupby(by = ['x_indx', 'y_indx']).agg({ x:np.mean for x in factor_header }).reset_index()
offset = df.y_indx.min()
logging.info(f"Y index starting at : {offset}")
df['y_indx'] = df.y_indx - df.y_indx.min()

h = df.x_indx.max() + 1
wsize = df.y_indx.max() + 1
logging.info(f"Size: {h} x {wsize}")

amax = np.array(df[factor_header]).argmax(axis = 1)
binary_mtx = coo_matrix((np.ones(df.shape[0],dtype=bool), (range(df.shape[0]), amax)), shape=(df.shape[0], K)).toarray()

df.index = range(df.shape[0])
rgb_mtx = np.clip(np.around(np.array(df[factor_header]) @ cmtx * 255),0,255).astype(dt)
rgb_mtx_hard = np.clip(np.around(binary_mtx @ cmtx * 255),0,255).astype(dt)
pts = np.array(df[['x_indx', 'y_indx']], dtype=int)
pts_indx = list(df.index)
ref = sklearn.neighbors.BallTree(pts)
st = df.y_indx.min()
bsize = 1000
while st < wsize:
    ed = min([st + bsize, wsize])
    if ((df.y_indx > st) & (df.y_indx < ed)).sum() < 10:
        st = ed
        continue
    mesh = np.meshgrid(np.arange(h), np.arange(st, ed))
    nodes = np.array(list(zip(*(dim.flat for dim in mesh))), dtype=int)
    dv, iv = ref.query(nodes, k = 1, dualtree=True)
    indx = (dv[:, 0] < fill_range) & (dv[:, 0] > 0)
    pts = np.vstack((pts, nodes[indx, :]) )
    pts_indx += list(df.index[iv[indx, 0]])
    rgb_mtx = np.vstack((rgb_mtx,\
        np.clip(np.around(np.array(df.loc[iv[indx, 0], factor_header]) @ cmtx * 255),0,255).astype(dt)) )
    rgb_mtx_hard = np.vstack((rgb_mtx_hard,\
        np.clip(np.around(np.array(binary_mtx[iv[indx, 0], :]) @ cmtx * 255),0,255).astype(dt)) )

    print(st, ed, sum(indx), pts.shape[0])
    st = ed

pts[:,0] = h - np.clip(pts[:, 0], 1, h)
pts[:,1] = np.clip(pts[:, 1], 1, wsize)

img = np.zeros( (h, wsize, 3), dtype=dt)
for r in range(3):
    img[:, :, r] = coo_array((rgb_mtx[:, r], (pts[:,0], pts[:,1])),\
        shape=(h, wsize), dtype = dt).toarray()
if args.tif:
    img = Image.fromarray(img, mode="I;16")
else:
    img = Image.fromarray(img)

outf = figure_path + "/" + args.output_id+"_"+str(int(diam))
outf += ".tif" if args.tif else ".png"
img.save(outf)

img = np.zeros( (h, wsize, 3), dtype=dt)
for r in range(3):
    img[:, :, r] = coo_array((rgb_mtx_hard[:, r], (pts[:,0], pts[:,1])),\
        shape=(h, wsize), dtype = dt).toarray()
if args.tif:
    img = Image.fromarray(img, mode="I;16")
else:
    img = Image.fromarray(img)

outf = figure_path + "/" + args.output_id+"_"+str(int(diam)) + ".top"
outf += ".tif" if args.tif else ".png"
img.save(outf)




binary_rgb = np.array([[255,153,0], [17,101,154]])
if args.plot_individual_factor:
    for k in range(K):
        v = np.clip(df.loc[pts_indx, factor_header[k]].values,0,1)
        rgb_mtx = np.clip(np.vstack((v, 1-v)).T @ binary_rgb, 0, 255).astype(dt)
        img = np.zeros( (h, wsize, 3), dtype=dt)
        for r in range(3):
            img[:, :, r] = coo_array((rgb_mtx[:, r], (pts[:,0], pts[:,1])),\
                shape=(h, wsize), dtype = dt).toarray()
        if args.tif:
            img = Image.fromarray(img, mode="I;16")
        else:
            img = Image.fromarray(img)
        outf = figure_path+"/"+args.output_id+"_"+str(int(diam))+".F_"+str(k)
        outf += ".tif" if args.tif else ".png"
        img.save(outf)
