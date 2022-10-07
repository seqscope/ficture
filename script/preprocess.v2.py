import sys, os, gzip, gc, argparse, warnings, logging
import subprocess as sp
from io import StringIO
import numpy as np
import pandas as pd
import sklearn.neighbors
import sklearn.mixture

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *
from layout_fn import layout_map


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help='')
parser.add_argument('--output_path', type=str, help='')
parser.add_argument('--ref_pts', type=str, help='')
parser.add_argument('--identifier', type=str, help='Identifier for the processed data e.g. 1stID-2ndZ-Species-L')
parser.add_argument("--meta_data", type=str, help="Per tile meta data menifest.tsv")
parser.add_argument("--layout", type=str, help="Layout file of tiles to draw [lane] [tile] [row] [col] format in each line")
parser.add_argument('--lane', type=str, help='')
parser.add_argument('--tile', type=str, help='',default='')
parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--species', type=str, default='', help='')
parser.add_argument('--feature_prefix', type=str, default='', help='e.g. mouse:mm10___,human:GRCh38_ (only for multi-species scenario)')
parser.add_argument('--filter_by_box_nrow', type=int, default=1, help='')
parser.add_argument('--filter_by_box_ncol', type=int, default=1, help='')
parser.add_argument('--filter_based_on', type=str, default='gn', help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
parser.add_argument('--precision_um', type=int, default=1, help='')
parser.add_argument('--log', default = '', type=str, help='files to write log to')

parser.add_argument('--gene_type_info', type=str, help='A file containing two columns, gene name and gene type. Used only if specific types of genes are kept.', default = '')
parser.add_argument('--gene_type_keyword', type=str, help='Key words (separated by ,) of gene types to keep, only used is gene_type_info is provided.', default="IG,TR,protein,lnc")
parser.add_argument('--rm_gene_type_keyword', type=str, help='Key words (separated by ,) of gene types to remove, only used is gene_type_info is provided.', default="pseudogene")
parser.add_argument('--rm_gene_keyword', type=str, help='Key words (separated by ,) of gene names to remove, only used is gene_type_info is provided.', default="")

parser.add_argument('--min_count_per_feature', type=int, default=20, help='')
parser.add_argument('--min_abs_mol_density_squm', type=float, default=0.02, help='A safe lowerbound to remove very sparse technical noise')
parser.add_argument('--hard_threshold', type=float, default=-1, help='If provided, filter by hard threshold (number of molecules per squared um)')
parser.add_argument('--hex_diam', type=int, default=12, help='')
parser.add_argument('--hex_n_move', type=int, default=6, help='')
parser.add_argument('--redo_filter', action='store_true')

args = parser.parse_args()
if args.log != '':
    try:
        logging.basicConfig(filename=args.log, filemode='a', encoding='utf-8', level=logging.INFO)
    except:
        logging.basicConfig(level= getattr(logging, "INFO", None))
else:
    logging.basicConfig(level= getattr(logging, "INFO", None))

path=args.input_path
outbase=args.output_path
lane=args.lane
kept_list=args.tile.split(',')
mu_scale = 1./args.mu_scale
diam = args.hex_diam
n_move = args.hex_n_move
radius = diam / np.sqrt(3)
hex_area = diam*radius*3/2
key = args.filter_based_on
ct_header = ['gn', 'gt', 'spl', 'unspl', 'ambig']

### Extract list of tiles to process
if args.tile == '':
    cmd="find "+path+"/"+lane+" -type d "
    tab = sp.check_output(cmd, stderr=sp.STDOUT, shell=True).decode('utf-8')
    tab = tab.split('\n')
    kept_list = sorted([x.split('/')[-1] for x in tab])

### Output
outpath = '/'.join([outbase,lane])
if not os.path.exists(outpath):
    arg="mkdir -p "+outpath
    os.system(arg)
flt_f = outpath+"/matrix_merged."+args.identifier+".tsv"
logging.info(f"Output file:\n{flt_f}")

if os.path.exists(flt_f) and not args.redo_filter:
    warnings.warn("Output file already exists but will be overwritten without --redo_filter")

### Layout
layout, lanes, tiles = layout_map(args.meta_data, args.layout, lane)
xbin_min, ybin_min = layout.xmin.min(), layout.ymin.min()
xbin_max, ybin_max = layout.xmax.max(), layout.ymax.max()
xr = xbin_max-xbin_min+1
yr = ybin_max-ybin_min+1
logging.info(f"Read meta data. Xmax, Ymax: {xbin_max}, {ybin_max}")
nrows = int(layout.row.max() + 1)
ncols = int(layout.col.max() + 1)
# Code the output as the tile numbers of the lower-left and upper-right corners
tile_ll = tiles[-1][0]
tile_ur = tiles[0][-1]
logging.info(f"Read layout info. lane {lane}, tile {tile_ll}-{tile_ur}")


### If work on subset of genes
gene_kept = []
if args.gene_type_info != '' and os.path.exists(args.gene_type_info):
    gencode = pd.read_csv(args.gene_type_info, sep='\t', names=['Name','Type','species'])
    if args.species in gencode.species.values:
        gencode = gencode[gencode.species.eq(args.species)]
    if args.feature_prefix != '':
        gene_prefix = [x.split(":") for x in args.feature_prefix.split(",")]
        for i,v in enumerate(gene_prefix):
            if len(v) != 2:
                continue
            indx = gencode.species.eq(v[0])
            if sum(indx) == 0:
                continue
            gencode.loc[indx, "Name"] = v[1] + gencode.loc[indx, "Name"].values
    kept_key = [x for x in args.gene_type_keyword.split(',') if x != '']
    rm_key = [x for x in args.rm_gene_type_keyword.split(',') if x != '']
    rm_name = [x for x in args.rm_gene_keyword.split(",") if x != '']
    if len(kept_key) > 0:
        gencode = gencode.loc[gencode.Type.str.contains('|'.join(kept_key)), :]
    if len(rm_key) > 0:
        gencode = gencode.loc[~gencode.Type.str.contains('|'.join(rm_key)), :]
    if len(rm_name) > 0:
        gencode = gencode.loc[~gencode.Name.str.contains('|'.join(rm_name)), :]
    gene_kept = list(gencode.Name)
    print(f"Read {len(gene_kept)} genes from gene_type_info")

### Decide which genes to keep
f = path + "/" + str(lane) + "/features.tsv.gz"
feature_tot = pd.read_csv(gzip.open(f, 'rb'), sep='\t|,',\
        names=["gene_id","gene","i"]+ct_header, engine='python')
feature_kept = list(feature_tot.loc[feature_tot[key] > args.min_count_per_feature,'gene'].values)
if len(gene_kept) > 0:
    gene_kept = [x for x in gene_kept if x in feature_kept]
else:
    gene_kept = feature_kept
logging.info(f"Kept {len(gene_kept)} genes")
print(f"Kept {len(gene_kept)} genes")


if os.path.exists(args.ref_pts) and not args.redo_filter:
    pt = pd.read_csv(args.ref_pts,sep='\t',names=['x','y'],dtype=int)
    logging.info(f"Read existing anchor positions:\n{args.ref_pts}, {pt.shape}. (Use --redo_filter to avoid using existing files)")
else:
    ### Read barcode
    # f=path+"/"+str(lane)+"/barcodes.sorted.tsv.gz"
    df=pd.DataFrame()
    bsize_row = np.max([1, nrows // args.filter_by_box_nrow])
    bsize_col = np.max([1, ncols // args.filter_by_box_ncol])
    for itr_r in range(nrows):
        for itr_c in range(ncols):
            lane, tile = lanes[itr_r][itr_c], tiles[itr_r][itr_c]
            f="/".join([path,lane,tile])+"/barcodes.tsv.gz"
            if not os.path.exists(f):
                continue
            sub = pd.read_csv(gzip.open(f, 'rb'),\
                sep='\t|,', names=["barcode","j","v2",\
                    "lane","tile","X","Y"]+ct_header,\
                usecols=["Y","X",key], engine='python')
            # cmd = "tabix " + f + " " + tile
            # sub = pd.read_csv(StringIO(sp.check_output(cmd, stderr=sp.STDOUT,\
            #         shell=True).decode('utf-8')), sep='\t|,', dtype=int,\
            #         names=["v1","lane","tile","Y","X"]+ct_header,\
            #         usecols=["Y","X",key], engine='python')
            sub = sub[sub[key] > 0]
            sub['X'] = (nrows - itr_r - 1) * xr + sub.X.values - xbin_min
            sub['Y'] = itr_c * yr + sub.Y.values - ybin_min
            sub['X'] = sub.X.values * mu_scale
            sub['Y'] = sub.Y.values * mu_scale
            if args.precision_um > 0:
                sub['X'] = (np.around(sub.X.values/args.precision_um,0)*args.precision_um).astype(int)
                sub['Y'] = (np.around(sub.Y.values/args.precision_um,0)*args.precision_um).astype(int)
            sub['win'] = str(itr_r//bsize_row) + '_' + str(itr_c//bsize_col)
            df = pd.concat((df, sub))
            logging.info(f"{lane}-{tile}, {sub.shape[0]}, {df.shape[0]}")
    logging.info(f"Read barcodes, collapsed into {df.shape[0]} pts")
    df['hex_x'] = 0
    df['hex_y'] = 0
    ### Detect grid points falling inside dense tissue region
    anchor = np.zeros((0,2))
    for w in df.win.unique():
        indx = df.win.eq(w)
        m0v=[]
        m1v=[]
        for i in range(n_move):
            for j in range(n_move):
                df.loc[indx, 'hex_x'], df.loc[indx, 'hex_y'] = pixel_to_hex(np.asarray(df.loc[indx, ['X','Y']]), radius, i/n_move, j/n_move)
                cnt = df.loc[indx, :].groupby(by = ['hex_x','hex_y']).agg({key:sum}).reset_index()
                cnt = cnt[cnt[key] > hex_area * args.min_abs_mol_density_squm]
                if cnt.shape[0] < 10:
                    continue
                if args.hard_threshold > 0:
                    cnt['det'] = cnt[key] > hex_area * args.hard_threshold
                else:
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
                    if m1 > m0 * .7:
                        cnt['det'] = 1
                    m0v.append(m0)
                    m1v.append(m1)
                if cnt.det.eq(True).sum() < 2:
                    continue
                m0 = cnt.loc[cnt.det.eq(True), key].median()/hex_area
                m1 = cnt.loc[cnt.det.eq(False), key].median()/hex_area
                m0v.append(m0)
                m1v.append(m1)
                anchor_x, anchor_y = hex_to_pixel(cnt.loc[cnt.det.eq(True), 'hex_x'].values, cnt.loc[cnt.det.eq(True), 'hex_y'].values, radius,i/n_move,j/n_move)
                anchor = np.vstack( (anchor, np.concatenate([anchor_x.reshape(-1,1), anchor_y.reshape(-1,1)], axis = 1) ) )
                logging.info(f"{m0:.3f} v.s. {m1:.3f}")
        m0 = np.mean(m0v)
        m1 = np.mean(m1v)
        logging.info(f"{str(w)}:\t{m0:.3f} v.s. {m1:.3f}")

    pt = np.around(np.clip(anchor,0,np.inf),0).astype(int)
    pt = pd.DataFrame({'x':pt[:,0], 'y':pt[:,1]})
    pt.drop_duplicates(inplace=True)
    pt.to_csv(args.ref_pts, sep='\t', index=False, header=False)
    del df
    gc.collect()


ref = sklearn.neighbors.BallTree(np.array(pt[['x','y']]))
logging.info(f"Built balltree for reference points")


### Read pixels and keep only those close to the kept grid points
feature_tot_ct = pd.DataFrame()
output_header = ['#lane','tile','Y','X','gene','gene_id']+ct_header
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
                "lane","tile","X","Y"]+ct_header,\
                usecols=["j","X","Y"], engine='python')
            f=datapath+"/matrix.mtx.gz"
            mtx = pd.read_csv(gzip.open(f, 'rb'),\
                sep=' ', skiprows=3, names=["i","j"]+ct_header)
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
        sub = sub.loc[sub.gene.isin(gene_kept), :]
        if sub.shape[0] < 10:
            continue
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
        feature_sub = sub.groupby(by = ['gene', 'gene_id']).agg({x:sum for x in ct_header}).reset_index()
        feature_tot_ct = pd.concat((feature_tot_ct, feature_sub))
        feature_tot_ct = feature_tot_ct.groupby(by = ['gene', 'gene_id']).agg({x:sum for x in ct_header}).reset_index()
        sub.sort_values(by = ['Y','X','gene'], inplace=True)
        n_pixel += sub.shape[0]
        sub.to_csv(flt_f, mode='a', sep='\t', index=False, header=False)
        logging.info(f"Write data for {lane}_{tile}, {sub.shape}. (total {n_pixel} so far)")

feature_tot_ct = feature_tot_ct.groupby(by = ['gene', 'gene_id']).agg({x:sum for x in ct_header}).reset_index()
f = outpath+"/feature."+args.identifier+".tsv.gz"
feature_tot_ct.to_csv(f, sep='\t', index=False, header=True)
logging.info(f"Finish filtering")
