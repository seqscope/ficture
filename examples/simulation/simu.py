import sys, os, copy, gzip, time, logging, pickle, argparse
import numpy as np
import pandas as pd
import sklearn.metrics
from scipy.sparse import *
from sklearn.preprocessing import normalize
import sklearn.neighbors
import matplotlib.colors
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='')
parser.add_argument('--model', type=str, help='')
parser.add_argument('--pixel_density', type=float, help="")
parser.add_argument('--avg_umi_per_pixel', type=float, default=1, help="")
parser.add_argument('--seed', type=int, default=-1, help="")

parser.add_argument('--r_circle', type=float, default=6, help="")
parser.add_argument('--d_rod', type=float, default=7, help="")
parser.add_argument('--r_rod', type=float, default=2, help="")
parser.add_argument('--d_diamond', type=float, default=15, help="")
parser.add_argument('--gamma_diamond', type=float, default=np.pi/16, help="")
parser.add_argument('--f_rod', type=float, default=0.3, help="")
parser.add_argument('--f_diamond', type=float, default=0.3, help="")
parser.add_argument('--buff', type=float, default=2, help="")
parser.add_argument('--buff_mix', type=float, default=0.5, help="")
parser.add_argument('--avg_cdist', type=float, default=15, help="")
parser.add_argument('--min_ct_per_feature', type=int, default=10, help="")
parser.add_argument('--f_scatter', type=float, default=0.1, help="")
parser.add_argument('--background', type=float, default=0.2, help="")

parser.add_argument('--spike', nargs="*", type=str)
parser.add_argument('--spike_color', nargs="*", type=str, default=\
    ["#f0c9cf","#813c85"])
parser.add_argument('--block', nargs="*", type=str)
parser.add_argument('--block_color', nargs="*", type=str, default=\
    ['#fea12f', '#4666dd', '#8bfe4b', '#e84b0c', '#04751c', '#dae236', '#a91501', '#2fb1f3'])

parser.add_argument('--R', type=int, default=1, help="")
parser.add_argument('--block_x', type=int, default=1000, help="")
parser.add_argument('--block_y', type=int, default=1000, help="")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

from ficture.utils.utilt import plot_colortable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from batman import batman

if len(args.spike_color) != len(args.spike):
    sys.exit("len(args.spike_color) != len(args.spike)")

seed = args.seed
if args.seed > 0:
    np.random.seed(args.seed)
else:
    seed = int(time.time())
    np.random.seed(seed)
logging.info(f"Use random seed {seed}")

path = args.path
avg_umi_per_pixel = args.avg_umi_per_pixel
pixel_density = args.pixel_density
spotty = args.spike
block = args.block
buff = args.buff
buff_mix = args.buff_mix
f_scatter = args.f_scatter
background = args.background

r_circle = args.r_circle
d_rod = args.d_rod
r_rod = args.r_rod
d_diamond = args.d_diamond
gamma_diamond = args.gamma_diamond

xmax = args.block_x
ymax = args.block_y
R = args.R # total area xmax x ymax x R
avg_cdist = args.avg_cdist
n_cell = int(xmax/avg_cdist) * int(ymax/avg_cdist)

f_rod = args.f_rod
f_diamond = args.f_diamond
f_circle = 1-f_rod-f_diamond

n_diamond = int(n_cell * f_diamond)
n_rod = int(n_cell * f_rod)
n_circle = n_cell - n_diamond - n_rod
shape_list = ['circle', 'rod', 'diamond']
shape_frac = [f_circle, f_rod, f_diamond]

nblock_x = 2
block_order = [5,0,3,6,2,4,7,1]
l = -np.log(1/0.95-1)

if not os.path.exists(path):
    os.makedirs(path)

def rod(x, y, theta=0, d=d_rod, r=r_rod):
    n = len(x)
    l = np.sqrt(x**2 + y**2)
    alpha = theta - np.arctan2(y, x)
    prj1 = l * np.cos(alpha)
    prj2 = l * np.sin(alpha)
    flag = np.zeros(n, dtype=bool)
    flag[(np.abs(prj2) <= r) & (np.abs(prj1) <= d)] =  1
    indx = (np.abs(prj2) <= r) & (np.abs(prj1) > d) & (np.abs(prj1) <= d + r)
    xi = x[indx] - d * np.cos(theta)
    yi = y[indx] - d * np.sin(theta)
    xj = x[indx] + d * np.cos(theta)
    yj = y[indx] + d * np.sin(theta)
    flag[indx] = (xi**2 + yi**2 < r**2) | ((xj**2 + yj**2 < r**2))
    return flag

def diamond(x, y, theta=0, d=d_diamond, gamma=np.pi/16):
    l = np.sqrt(x**2 + y**2)
    alpha = theta - np.arctan2(y, x)
    prj1 = l * np.cos(alpha)
    prj2 = l * np.sin(alpha)
    return (np.abs(prj1) < d) & (np.arctan2(np.abs(prj2), d - np.abs(prj1)) < gamma )

model=pd.read_csv(args.model,sep='\t',header=0,index_col=0)
cell_type_list=model.columns
spotty = [x for x in spotty if x in cell_type_list]
block = [x for x in block if x in cell_type_list]
cell_type_list = spotty + block
model = model.loc[:, cell_type_list]
k_spotty = len(spotty)
k_block = len(block)
K = len(cell_type_list)
logging.info(f"Read model, will use {k_spotty} + {k_block} cell types")

model_p = copy.copy(model)
for k in model_p.columns:
    model_p[k] /= model_p[k].sum()
M, K = model.shape
feature_kept = model.index.values

if len(args.block_color) != len(args.block):
    print(len(args.block_color), len(args.block))
    cmap = plt.get_cmap("turbo", k_block*2+2)
    cmtx = [cmap(2*block_order[k]+2)[:3] for k in range(k_block)]
    logging.info(f"Use turbo color map for block cell types")
else:
    cmtx = [matplotlib.colors.to_rgb(x) for x in args.block_color]

cmtx = [matplotlib.colors.to_rgb(x) for x in args.spike_color] + cmtx
df = pd.DataFrame({"Name":range(K), 'cell_label':cell_type_list})
df = pd.concat([df, pd.DataFrame(cmtx, columns=["R", "G", "B"])], axis=1)
df.index = range(K)
f=path+"/model.rgb.tsv"
df.to_csv(f, sep='\t', index=False, float_format="%.5f")

cdict = {df.loc[k, 'cell_label']:cmtx[k] for k in range(K)}
f=path+"/model.cbar.png"
fig = plot_colortable(cdict, "Cell type label", sort_colors=False,
                      ncols=1, cell_width = 400, title_fontsize=18, text_fontsize=18)
fig.savefig(f, format="png", transparent=True)


cheader = ['INDEX','x','y','cell_shape','cell_label','layer']
lheader = ['X','Y','cell_id','cell_label','cell_shape']
oheader = ['X','Y', 'gene', 'Count', 'cell_id','cell_shape']
outf_label=path+"/pixel_label.uniq.tsv.gz"
with gzip.open(outf_label, 'wt') as wf:
    _=wf.write('\t'.join(lheader) + '\n')
outf=path+"/matrix.tsv.gz"
with gzip.open(outf, 'wt') as wf:
    _=wf.write('\t'.join(oheader) + '\n')
outf_cell = path+"/cell_info.tsv.gz"
with gzip.open(outf_cell, 'wt') as wf:
    _=wf.write('\t'.join(cheader) + '\n')

block_shuffle = np.array(block)

for r in range(R):

    ### Simulate cells
    # create regularly spaced points
    x = np.linspace(0., xmax, int(xmax / avg_cdist), dtype=np.float32)
    y = np.linspace(0., ymax, int(ymax / avg_cdist), dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1,2)
    # perturb points
    max_movement = avg_cdist*.5
    noise = np.random.uniform(low=-max_movement, high=max_movement, size=(len(coords), 2))
    coords += noise
    # Wrap around out of boundary points
    cell_center_x = coords[:, 0] % xmax
    cell_center_y = coords[:, 1] % ymax
    # assign background cell to cell types
    level0 = batman(cell_center_x, cell_center_y, scale = (xmax/14, xmax/14*1.5), shift = (xmax//2, ymax//2)).astype(int)
    level1 = np.clip(cell_center_x // (xmax//nblock_x), 0, nblock_x-1).astype(int)
    bsize = xmax//nblock_x
    # Fuzzy transition boundary
    pgrd =  1/(1+np.exp(-l*(cell_center_x % bsize / bsize * 2 - 1)))
    pgrd[cell_center_x % bsize / bsize > .95] = 1
    pgrd[cell_center_x % bsize / bsize < .05] = 0
    level2 = np.random.binomial(1, p=pgrd)

    cell_label = block_shuffle[(level0 * 4 + level1 * 2 + level2)]

    cell = pd.DataFrame({'x':cell_center_x, 'y':cell_center_y,
                        'level0':level0, 'cell_label':cell_label})
    cell.index = np.arange(n_cell)
    cell['cell_shape'] = np.random.choice( shape_list, size = n_cell, p = shape_frac )
    cell['layer'] = 0
    indx = np.random.choice(n_cell, size=int(n_cell * f_scatter), replace=False)
    cell.loc[indx, 'cell_label'] = np.random.choice(spotty, size=len(indx), replace=True)
    cell.loc[indx, 'layer'] = 1

    logging.info(f"Created {n_cell} cells")
    print(cell.cell_shape.value_counts())

    ### Simulate pixel
    n_pixel = int(xmax * ymax * pixel_density)
    pixel = pd.DataFrame({'x':np.random.uniform(0, xmax, size=n_pixel),
                          'y':np.random.uniform(0, ymax, size=n_pixel)})
    pixel.index = np.arange(n_pixel)
    pixel['level0'] = batman(pixel.x.values, pixel.y.values,
                            scale = (xmax/14, xmax/14*1.5),
                            shift = (xmax//2, ymax//2)).astype(int)
    pixel['cell_id'] = -1
    pixel['cell_label'] = ''
    pixel['cell_shape'] = ''
    pixbt = sklearn.neighbors.BallTree(pixel[['x', 'y']])
    # assigne pixel to cells
    for layer in [0, 1]:
        if layer == 0 and background >= 1:
            continue
        for k in range(2):
            if layer == 0:
                cell_indx = cell.level0.eq(k) & cell.layer.eq(layer)
                pixel_indx = pixel.index[pixel.level0.eq(k)]
            else:
                if k == 1:
                    continue
                cell_indx = cell.layer.eq(layer)
                pixel_indx = pixel.index.values

            indx = cell.index[cell.cell_shape.eq('circle') & cell_indx]
            ivec, dvec = pixbt.query_radius(cell.loc[indx, ['x','y']],r=r_circle+buff,return_distance=True)
            for i, idx in enumerate(indx):
                if layer == 0:
                    flt = pixel.loc[ivec[i], 'level0'].eq(k)
                    ivec[i] = ivec[i][flt]
                    dvec[i] = dvec[i][flt]
                n = len(ivec[i])
                if n == 0:
                    continue
                inner = dvec[i] <= r_circle
                p_paint = np.ones(n) * buff_mix
                p_paint[inner] = 1
                paint = np.random.binomial(1, p=p_paint)
                indx = ivec[i][paint==1]
                pixel.loc[indx, 'cell_id'] = idx
                pixel.loc[indx, 'cell_label'] = cell.loc[idx, 'cell_label']
                pixel.loc[indx, 'cell_shape'] = 'circle'

            indx = cell.index[cell.cell_shape.eq('diamond') & cell_indx]
            ivec = pixbt.query_radius(cell.loc[indx, ['x','y']],r=d_diamond+buff,return_distance=False)
            for i, idx in enumerate(indx):
                if layer == 0:
                    flt = pixel.loc[ivec[i], 'level0'].eq(k)
                    ivec[i] = ivec[i][flt]
                n = len(ivec[i])
                if n == 0:
                    continue
                theta = np.pi * np.random.uniform(0, 1)
                x = pixel.loc[ivec[i], 'x'].values - cell.loc[idx, 'x']
                y = pixel.loc[ivec[i], 'y'].values - cell.loc[idx, 'y']
                outer = diamond(x,y,theta=theta,d=d_diamond+buff,gamma=gamma_diamond)
                inner = diamond(x,y,theta=theta,d=d_diamond,gamma=gamma_diamond)
                p_paint = np.zeros(len(ivec[i]))
                p_paint[outer] = buff_mix
                p_paint[inner] = 1
                paint = np.random.binomial(1, p=p_paint)
                indx = ivec[i][paint==1]
                pixel.loc[indx, 'cell_id'] = idx
                pixel.loc[indx, 'cell_label'] = cell.loc[idx, 'cell_label']
                pixel.loc[indx, 'cell_shape'] = 'diamond'

            indx = cell.index[cell.cell_shape.eq('rod') & cell_indx]
            ivec = pixbt.query_radius(cell.loc[indx, ['x','y']],r=d_rod+r_rod+buff,return_distance=False)
            for i, idx in enumerate(indx):
                if layer == 0:
                    flt = pixel.loc[ivec[i], 'level0'].eq(k)
                    ivec[i] = ivec[i][flt]
                n = len(ivec[i])
                if n == 0:
                    continue
                theta = np.pi * np.random.uniform(0, 1)
                x = pixel.loc[ivec[i], 'x'].values - cell.loc[idx, 'x']
                y = pixel.loc[ivec[i], 'y'].values - cell.loc[idx, 'y']
                outer = rod(x,y,theta=theta,d=d_rod,r=r_rod+buff)
                inner = rod(x,y,theta=theta,d=d_rod,r=r_rod)
                p_paint = np.zeros(len(ivec[i]))
                p_paint[outer] = buff_mix
                p_paint[inner] = 1
                paint = np.random.binomial(1, p=p_paint)
                indx = ivec[i][paint==1]
                pixel.loc[indx, 'cell_id'] = idx
                pixel.loc[indx, 'cell_label'] = cell.loc[idx, 'cell_label']
                pixel.loc[indx, 'cell_shape'] = 'rod'

    for k in range(2):
        indx = cell.index[cell.level0.eq(k) & cell.layer.eq(0)]
        cpt = np.array(cell.loc[indx, ['x', 'y']] )
        ref = sklearn.neighbors.BallTree(cpt)
        indx0 = pixel.index[pixel.level0.eq(k) & pixel.cell_id.eq(-1) ]
        if background < 1:
            indx0 = np.random.choice(indx0, size=int(len(indx0) * background))
        mapi = ref.query(pixel.loc[indx0, ['x','y']],k=1,return_distance=False).reshape(-1)
        pixel.loc[indx0, 'cell_label'] = cell.loc[indx[mapi], 'cell_label'].values
        pixel.loc[indx0, 'cell_id'] = indx[mapi]
        pixel.loc[indx0, 'cell_shape'] = "background"

    pixel = pixel.loc[pixel.cell_id != -1]
    n_pixel = pixel.shape[0]
    if avg_umi_per_pixel > 1:
        pixel['umi'] = np.random.poisson(avg_umi_per_pixel-1, n_pixel) + 1
    else:
        pixel['umi'] = 1

    logging.info(f"Generated {n_pixel} pixels")

    umi_celltype = pixel.groupby(by='cell_label').agg({'umi':sum})
    N = n_pixel

    feature = pd.DataFrame()
    mtx = pd.DataFrame()
    cell['INDEX'] = np.arange(cell.shape[0]) + r * n_cell
    cell['x'] += r * xmax
    for k,v in umi_celltype.iterrows():
        v = v['umi']
        r_indx = [i for i,v in enumerate(pixel.loc[pixel.cell_label.eq(k), 'umi'].values) for x in range(v) ]
        w = np.random.multinomial(n=v, pvals=model_p[k])
        c_indx = [i for i,x in enumerate(w) for y in range(x)]
        np.random.shuffle(c_indx)
        r_indx = pixel.index[pixel.cell_label.eq(k) ].values[r_indx]
        sub=pd.DataFrame({'j':r_indx, 'i':c_indx, 'Count':np.ones(len(r_indx), dtype=int)})
        sub=sub.groupby(by=['j','i']).agg({'Count':sum}).reset_index()
        sub['cell_label'] = k
        sub = sub.merge(right=pixel[['x', 'y', 'cell_id','cell_shape']], left_on = 'j', right_index=True)
        sub['gene'] = feature_kept[sub.i.values]
        sub['x'] += r * xmax
        sub['cell_id'] += r * n_cell
        sub.drop(columns = ['j','i'], inplace=True)
        mtx = pd.concat([mtx, sub])

    mtx.sort_values(by = 'x', inplace=True)
    mtx.rename(columns={'x':'X', 'y':'Y'}, inplace=True)
    mtx.loc[:, oheader].to_csv(outf,sep='\t',mode='a',index=False,header=False,float_format="%.2f")
    brc = mtx[lheader].drop_duplicates(subset=['X', 'Y'])
    brc.to_csv(outf_label,sep='\t',mode='a',index=False,header=False,float_format="%.2f")
    cell[cheader].to_csv(outf_cell,sep='\t',mode='a',index=False,header=False,float_format="%.2f")
    feature = pd.concat([feature, mtx.groupby(by = 'gene').agg({'Count':sum}).reset_index() ])
    logging.info(f"Finished the {r}-th replicate")
    np.random.shuffle(block_shuffle)

feature = feature.groupby(by = 'gene').agg({'Count':sum}).reset_index()
feature = feature[feature.Count >= args.min_ct_per_feature]
f=path+"/feature.tsv.gz"
feature.to_csv(f,sep='\t',index=False,header=True)

model = model.loc[feature.gene.values, :]
f=path+"/model.true.tsv.gz"
model.to_csv(f,sep='\t',index=True,header=True)
