import sys, os, gzip, argparse, warnings
import subprocess as sp
import numpy as np
import pandas as pd
import sklearn.neighbors
import sklearn.mixture
import ruptures as rpt

# Add parent directory
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *
from layout_fn import layout_map


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help='')
parser.add_argument('--output_path', type=str, help='')
parser.add_argument('--identifier', type=str, help='Identifier for the processed data e.g. 1stID-2ndZ-Species-L')
parser.add_argument("--meta_data", type=str, help="Per tile meta data menifest.tsv")
parser.add_argument("--layout", type=str, help="Layout file of tiles to draw [lane] [tile] [row] [col] format in each line")
parser.add_argument('--lane', type=str, help='')
parser.add_argument('--tile', type=str, help='',default='')
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--filter_based_on', type=str, default='gn', help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced, velo: velo total')

parser.add_argument('--gene_type_info', type=str, help='A file containing two columns, gene name and gene type. Used only if specific types of genes are kept.', default = '')
parser.add_argument('--gene_type_keyword', type=str, help='Key words (separated by ,) of gene types to keep, only used is gene_type_info is provided.', default="IG,TR,protein,lnc")
parser.add_argument('--rm_gene_keyword', type=str, help='Key words (separated by ,) of gene names to remove, only used is gene_type_info is provided.', default="")

parser.add_argument('--min_count_per_feature', type=int, default=20, help='')
parser.add_argument('--min_abs_mol_density_squm', type=float, default=0.1, help='')
parser.add_argument('--hex_diam', type=int, default=12, help='')
parser.add_argument('--hex_n_move', type=int, default=6, help='')
parser.add_argument('--redo_filter', action='store_true')

args = parser.parse_args()

path=args.input_path
outbase=args.output_path
lane=args.lane
kept_list=args.tile.split(',')
mu_scale = 1./args.mu_scale
# Parameter for filter
clps_bin = 10
nbin_max = 40
nbin_min = 8

### Output
outpath = '/'.join([outbase,lane])
if not os.path.exists(outpath):
    arg="mkdir -p "+outpath
    os.system(arg)
flt_f = outpath+"/matrix_merged."+args.identifier+".tsv"
print(f"Output file:\n{flt_f}")
if os.path.exists(flt_f) and not args.redo_filter:
    sys.exit("Output file already exists. Do you want to --redo_filter?")


### Layout
layout, lanes, tiles = layout_map(args.meta_data, args.layout, lane)
xbin_min, ybin_min = layout.xmin.min(), layout.ymin.min()
xbin_max, ybin_max = layout.xmax.max(), layout.ymax.max()
xr = xbin_max-xbin_min+1
yr = ybin_max-ybin_min+1
print(f"Read meta data. Xmax, Ymax: {xbin_max}, {ybin_max}")
nrows = layout.row.max() + 1
# Code the output as the tile numbers of the lower-left and upper-right corners
tile_ll = tiles[-1][0]
tile_ur = tiles[0][-1]
print(f"Read layout info. lane {lane}, tile {tile_ll}-{tile_ur}")


### Read barcode, detect tissue region
ct_header = ['gn', 'gt', 'spl', 'unspl', 'ambig', 'velo']
key = args.filter_based_on
f=path+"/"+str(lane)+"/barcodes.sorted.tsv.gz"
if key == "velo":
    df = pd.read_csv(gzip.open(f,'rb'), sep='\t|,', dtype=int,\
            names=["v1","lane","tile","Y","X"]+ct_header,\
            usecols=["tile","Y","X","spl","unspl","ambig"], engine='python')
    df['velo'] = df.spl.values + df.unspl.values + df.ambig.values
    df.drop(columns = ["spl","unspl","ambig"], inplace=True)
else:
    df = pd.read_csv(gzip.open(f,'rb'), sep='\t|,', dtype=int,\
            names=["v1","lane","tile","Y","X"]+ct_header,\
            usecols=["tile","Y","X",key], engine='python')
df['tile'] = df.tile.astype(str)
for itr_r in range(len(lanes)):
    for itr_c in range(len(lanes[itr_r])):
        lane, tile = lanes[itr_r][itr_c], tiles[itr_r][itr_c]
        indx = df.tile.eq(tile)
        df.loc[indx, 'X'] = (nrows - itr_r - 1) * xr + df.loc[indx, 'X'].values - xbin_min
        df.loc[indx, 'Y'] = itr_c * yr + df.loc[indx, 'Y'].values - ybin_min
df['x'] = np.round(df.X.values * mu_scale,0).astype(int)
df['y'] = np.round(df.Y.values * mu_scale,0).astype(int)
df = df.groupby(by = ['x','y'], as_index=False).agg({key:sum})
df['xindx'] = np.around(df.x.values / clps_bin, 0).astype(int) * clps_bin
df['yindx'] = np.around(df.y.values / clps_bin, 0).astype(int) * clps_bin
df['Keep'] = 0
print(f"Read barcodes, collapsed into {df.shape[0]} pts")
yclps = df.groupby(by = 'yindx', as_index=False).agg({key:sum})
yclps.sort_values(by ='yindx', inplace=True)
yclps['tot'] = np.log10(yclps[key].values + 1)
algo = rpt.Pelt(model="rbf", min_size=200).fit(yclps.tot.values)
tunning = np.log(yclps.shape[0]) * 2
my_bkps = algo.predict(pen=tunning)
print(len(my_bkps))
niter = 0
while len(my_bkps) > nbin_max and niter < 5:
    tunning *= 2
    my_bkps = algo.predict(pen=tunning)
    niter += 1
    print(len(my_bkps))
while len(my_bkps) < nbin_min and niter < 5:
    tunning /= 2
    my_bkps = algo.predict(pen=tunning)
    niter += 1
    print(len(my_bkps))
if len(my_bkps) < nbin_min or len(my_bkps) > nbin_max:
    algo = rpt.BottomUp(model="rbf").fit(yclps.tot.values)
    my_bkps = algo.predict(n_bkps=25)

ycut = list(yclps.yindx.iloc[np.array([1]+my_bkps)-1])
ycut[-1] = df.yindx.max()
nw = len(ycut) - 1
df['win'] = pd.cut(df.yindx, bins = ycut, include_lowest=True, labels=range(nw))
wct = df.groupby(by = 'win').agg({key:sum, 'y':[max, min]}).reset_index()
wct['width'] = wct[('y','max')].values - wct[('y','min')]
xmax, xmin = df.x.max(), df.x.min()
wct['yDensity'] = wct[(key,'sum')].values / (wct[('width','')].values * (xmax-xmin))
w = wct.yDensity.values
v = np.sort(w)
gap = (v[1:]-v[:-1]) / v[:-1]
indx = np.argmax(gap)
print(indx, v[indx], v[indx+1])
cut = (v[indx] + v[indx+1])/2
wct['Primary'] = 0
wct.loc[wct.yDensity > cut, 'Primary'] = 1
w_center = wct.loc[wct.Primary.eq(1), ('win','')].values
wmax = wct[('win','')].max()
print(np.around(wct.loc[wct.Primary.eq(1), 'yDensity'].values, 3))
print(np.around(wct.loc[wct.Primary.eq(0), 'yDensity'].values, 3))


### Detect grid points falling inside dense tissue region
diam = args.hex_diam
n_move = args.hex_n_move
radius = diam / np.sqrt(3)
hex_area = diam*radius*3/2
anchor = np.zeros((0,2))
log = []
for w in w_center:
    st = max([0, w-1])
    ed = min([w+1, wmax])
    sub = copy.copy(df[df.win.isin(range(st, ed+1)) & df[key]>0] )
    m0v=[]
    m1v=[]
    for i in range(n_move):
        for j in range(n_move):
            sub['hex_x'], sub['hex_y'] = pixel_to_hex(np.asarray(sub[['x','y']]), radius, i/n_move, j/n_move)
            cnt = sub.groupby(by = ['hex_x','hex_y']).agg({key:sum}).reset_index()
            cnt = cnt[cnt[key] > hex_area * args.min_abs_mol_density_squm]
            v = np.log10(cnt[key].values).reshape(-1, 1)
            gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(v)
            lab_keep = np.argmax(gm.means_.squeeze())
            cnt['det'] = gm.predict(v) == lab_keep
            m0=(10**gm.means_.squeeze()[lab_keep])/hex_area
            m1=(10**gm.means_.squeeze()[1-lab_keep])/hex_area
            if m1 > m0 * 0.5 or m0 < args.min_abs_mol_density_squm:
                v = cnt[key].values.reshape(-1, 1)
                gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(v)
                lab_keep = np.argmax(gm.means_.squeeze())
                lab = gm.predict(v)
                cnt['det'] = gm.predict(v) == lab_keep
                m0=gm.means_.squeeze()[lab_keep]/hex_area
                m1=gm.means_.squeeze()[1-lab_keep]/hex_area
            if m0 < args.min_abs_mol_density_squm:
                continue
            if m1 > m0 * .5:
                cnt['det'] = 1
            anchor_x, anchor_y = hex_to_pixel(cnt.loc[cnt.det.eq(lab_keep), 'hex_x'].values, cnt.loc[cnt.det.eq(lab_keep), 'hex_y'].values, radius,i/n_move,j/n_move)
            anchor = np.vstack( (anchor, np.concatenate([anchor_x.reshape(-1,1), anchor_y.reshape(-1,1)], axis = 1) ) )
            log.append([w,i,j,m0,m1])
            m0v.append(m0)
            m1v.append(m1)
    print(w,np.around(m0v,3),np.around(m1v,3))

pt = np.around(np.clip(anchor,0,np.inf),0).astype(int)
pt = pd.DataFrame({'x':pt[:,0], 'y':pt[:,1]})
pt.drop_duplicates(inplace=True)
f = outpath + "/kept_grid_pt.tsv"
pt.to_csv(f, sep='\t', index=False, header=False)

ref = sklearn.neighbors.BallTree(np.array(pt[['x','y']]))


### Extract list of tiles to process
if args.tile == '':
    cmd="find "+path+"/"+lane+" -type d "
    tab = sp.check_output(cmd, stderr=sp.STDOUT, shell=True).decode('utf-8')
    tab = tab.split('\n')
    kept_list = sorted([x.split('/')[-1] for x in tab])

### If work on subset of genes
gene_kept = []
if args.gene_type_info != '' and os.path.exists(args.gene_type_info):
    gencode = pd.read_csv(args.gene_type_info, sep='\t', names=['Name','Type'])
    kept_key = args.gene_type_keyword.split(',')
    kept_type = gencode.loc[gencode.Type.str.contains('|'.join(kept_key)),'Type'].unique()
    gencode = gencode.loc[ gencode.Type.isin(kept_type) ]
    if args.rm_gene_keyword != "":
        rm_list = args.rm_gene_keyword.split(",")
        for x in rm_list:
            gencode = gencode.loc[ ~gencode.Name.str.contains(x) ]
    gene_kept = list(gencode.Name)

### Decide which genes to keep
f = path + "/" + str(lane) + "/features.tsv.gz"
feature_tot = pd.read_csv(gzip.open(f, 'rb'), sep='\t|,',\
        names=["gene_id","gene","i"]+ct_header[:-1], engine='python')
if key == "velo":
    feature_tot["velo"] = feature_tot[['spl','unspl','ambig']].sum(axis = 1).values
feature_kept = list(feature_tot.loc[feature_tot[key] > args.min_count_per_feature,'gene'].values)
if len(gene_kept) > 0:
    gene_kept = [x for x in gene_kept if x in feature_kept]


### Read pixels and keep only those close to the kept grid points
output_header = ['#lane','tile','Y','X','gene','gene_id']+ct_header[:-1]
header=pd.DataFrame([], columns = output_header)
header.to_csv(flt_f, sep='\t', index=False)
n_pixel = 0
for itr_r in range(len(lanes)):
    for itr_c in range(len(lanes[0])):
        lane, tile = lanes[itr_r][itr_c], tiles[itr_r][itr_c]
        if tile not in kept_list:
            continue
        try:
            datapath = "/".join([path,lane,tile])
            f=datapath+"/barcodes.tsv.gz"
            brc = pd.read_csv(gzip.open(f, 'rb'),\
                sep='\t|,', names=["barcode","j","v2",\
                "lane","tile","X","Y"]+ct_header[:-1],\
                usecols=["j","X","Y"], engine='python')
            f=datapath+"/matrix.mtx.gz"
            mtx = pd.read_csv(gzip.open(f, 'rb'),\
                sep=' ', skiprows=3, names=["i","j"]+ct_header[:-1])
            f=datapath+"/features.tsv.gz"
            feature = pd.read_csv(gzip.open(f, 'rb'),\
                sep='\t', names=["gene_id","gene","i","ct"],\
                usecols=["i","gene","gene_id"],  engine='python')
        except:
            warnings.warm(f"Unable to read data for tile {tile}")
            continue
        sub = mtx.merge(right = brc, on = 'j', how = 'inner')
        sub = sub.merge(right = feature, on = 'i', how = 'inner')
        sub.drop(columns = ['i', 'j'], inplace=True)
        sub['X'] = (nrows - itr_r - 1) * xr + sub.X.values - xbin_min
        sub['Y'] = itr_c * yr + sub.Y.values - ybin_min
        sub['j'] = lane + '_' + tile + '_' + sub.X.astype(str) + '_' + sub.Y.astype(str)
        sub['velo'] = sub[['spl','unspl','ambig']].sum(axis = 1).values
        sub = sub.loc[sub.gene.isin(gene_kept), :]
        brc = sub[['j','X','Y']].drop_duplicates(subset=['j'])
        brc['x'] = brc.X.values * mu_scale
        brc['y'] = brc.Y.values * mu_scale
        brc = brc[['j','x','y']]
        brc["kept"] = False
        dv, iv = ref.query(X=brc[['x','y']], k=1, return_distance=True, sort_results=False)
        dv = dv.squeeze()
        brc.loc[dv < radius, 'kept'] = True
        sub['#lane'] = lane
        sub['tile'] = tile
        sub = sub.loc[sub.j.isin( brc.loc[brc.kept.eq(True), 'j'] ), output_header]
        sub.sort_values(by = ['Y','X','gene'], inplace=True)
        n_pixel += sub.shape[0]
        sub.to_csv(flt_f, mode='a', sep='\t', index=False, header=False)
        print(f"Write data for {lane}_{tile}, {sub.shape}. (total {n_pixel} so far)")

print(f"Finish filtering")
