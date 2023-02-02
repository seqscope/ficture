from collections import defaultdict
import sys, os, gzip, gc, argparse, warnings, logging
import subprocess as sp
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
parser.add_argument('--count_key', nargs='*', type=str, help='One or multiple column names for read counts', default=["Count"])
parser.add_argument('--filter_based_on', type=str, default="Count", help='')

parser.add_argument('--pseudo_lane', type=str, default="1", help="")
parser.add_argument('--pseudo_tile', type=str, default="1", help="")

# parser.add_argument("--meta_data", type=str, help="Per tile meta data menifest.tsv")
# parser.add_argument("--layout", type=str, help="Layout file of tiles to draw [lane] [tile] [row] [col] format in each line")
# parser.add_argument('--lanes', nargs='*', type=str, help='One or multiple lanes to work on', default=[])
# parser.add_argument('--region', nargs='*', type=str, help="In the form of \"lane1:st1-ed1 lane2:st2-ed2 ... \"", default=[])

parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
parser.add_argument('--species', type=str, default='', help='')
parser.add_argument('--feature_prefix', type=str, default='', help='e.g. mouse:mm10___,human:GRCh38_ (only for multi-species scenario)')
parser.add_argument('--filter_by_box_nrow', type=int, default=1, help='')
parser.add_argument('--filter_by_box_ncol', type=int, default=1, help='')
parser.add_argument('--precision_um', type=int, default=1, help='')
parser.add_argument('--log', default = '', type=str, help='files to write log to')

parser.add_argument('--gene_type_info', type=str, help='A file containing two columns, gene name and gene type. Used only if specific types of genes are kept.', default = '')
parser.add_argument('--gene_type_keyword', type=str, help='Key words (separated by ,) of gene types to keep, only used is gene_type_info is provided.', default="IG,TR,protein,lnc")
parser.add_argument('--rm_gene_type_keyword', type=str, help='Key words (separated by ,) of gene types to remove, only used is gene_type_info is provided.', default="pseudogene")
parser.add_argument('--rm_gene_keyword', type=str, help='Key words (separated by ,) of gene names to remove, only used is gene_type_info is provided.', default="")

parser.add_argument('--min_abs_mol_density_squm', type=float, default=0.02, help='A safe lowerbound to remove very sparse technical noise')
parser.add_argument('--hard_threshold', type=float, default=-1, help='If provided, filter by hard threshold (number of molecules per squared um)')
parser.add_argument('--hex_diam', type=int, default=12, help='')
parser.add_argument('--hex_n_move', type=int, default=6, help='')
parser.add_argument('--redo_filter', action='store_true')
parser.add_argument('--merge_raw_data', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--anchor_only', action='store_true')


args = parser.parse_args()

if args.log != '':
    try:
        logging.basicConfig(filename=args.log, filemode='a', encoding='utf-8', level=logging.INFO)
    except:
        logging.basicConfig(level= getattr(logging, "INFO", None))
else:
    logging.basicConfig(level= getattr(logging, "INFO", None))

path = args.input_path
outpath = args.output_path

mu_scale = 1./args.mu_scale
diam = args.hex_diam
n_move = args.hex_n_move
radius = diam / np.sqrt(3)
hex_area = diam*radius*3/2

ct_header = args.count_key
key = args.filter_based_on
lane= args.pseudo_lane
tile= args.pseudo_tile

### Output
if not os.path.exists(outpath):
    arg="mkdir -p "+outpath
    os.system(arg)
flt_f = outpath+"/matrix_merged."+args.identifier+".tsv"
raw_f = outpath+"/matrix_merged.raw."+args.identifier+".tsv"
logging.info(f"Output file:\n{flt_f}")

if os.path.exists(flt_f):
    warnings.warn("Output file already exists and will be overwritten")


if os.path.exists(args.ref_pts) and not args.redo_filter:
    pt = pd.read_csv(args.ref_pts,sep='\t',header=0)
    logging.info(f"Read existing anchor positions:\n{args.ref_pts}, {pt.shape}. (Use --redo_filter to avoid using existing files)")
else:
    ### Read barcode
    f=os.path.join(path, "barcodes.tsv.gz")
    if not os.path.exists(f):
        sys.exit("Input directory does not contain barcodes.tsv.gz")

    df=pd.DataFrame()
    for chunk in pd.read_csv(gzip.open(f, 'rb'), sep='\t|,',\
            names=["barcode","j","v2","lane","tile","X","Y"]+ct_header,\
            usecols=["Y","X",key], engine='python', chunksize=1000000):
        chunk = chunk[chunk[key] > 0]
        chunk['X'] = chunk.X.values * mu_scale
        chunk['Y'] = chunk.Y.values * mu_scale
        npixel = chunk.shape[0]
        if args.precision_um > 0:
            chunk['X'] = np.around(chunk.X.values/args.precision_um,0).astype(int)*args.precision_um
            chunk['Y'] = np.around(chunk.Y.values/args.precision_um,0).astype(int)*args.precision_um
        chunk = chunk.groupby(by = ["X", "Y"]).agg({key:sum}).reset_index()
        df = pd.concat([df, chunk])
        logging.info(f"Read {npixel} pixels, collapsed into {chunk.shape[0]} pts ({df.shape[0]} so far)")

    xmax,xmin = df.X.max(), df.X.min()
    ymax,ymin = df.Y.max(), df.Y.min()
    bsize_row = np.max([1, (ymax-ymin)//args.filter_by_box_nrow])
    bsize_col = np.max([1, (xmax-xmin)//args.filter_by_box_ncol])
    df['win'] = ((df.Y-ymin)/bsize_row).astype(int).astype(str) + '_' +\
                ((df.X-xmin)/bsize_col).astype(int).astype(str)

    logging.info(f"Read barcodes, collapsed into {df.shape[0]} pts")
    ### Detect grid points falling inside dense tissue region
    pt = pd.DataFrame()
    for w in df.win.unique():
        indx = df.win.eq(w)
        indx = df.win.eq(w)
        sub, m0, m1 = filter_by_density_mixture(df.loc[indx, :], key, radius, n_move, args)
        pt = pd.concat([pt, sub])
        logging.info(f"Window {str(w)}:\t{m0:.3f} v.s. {m1:.3f}")

    pt.x = np.around(np.clip(pt.x.values,0,np.inf)/args.precision_um,0).astype(int)
    pt.y = np.around(np.clip(pt.y.values,0,np.inf)/args.precision_um,0).astype(int)
    pt.drop_duplicates(inplace=True)
    pt.x = pt.x * args.precision_um
    pt.y = pt.y * args.precision_um
    pt.to_csv(args.ref_pts, sep='\t', index=False, header=True)
    del df
    gc.collect()


if args.anchor_only:
    sys.exit()


pt = pt.astype({'x':float, 'y':float, 'lane':str})
ref = {}
for lane in pt.lane.unique():
    ref[lane] = sklearn.neighbors.BallTree(np.array(pt.loc[pt.lane.eq(lane), ['x','y']]))
logging.info(f"Built balltree for reference points")



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


### Read pixels and keep only those close to the kept grid points
feature_tot_ct = pd.DataFrame()
output_header = ['#lane','tile','Y','X','gene','gene_id']+ct_header
header=pd.DataFrame([], columns = output_header)
header.to_csv(flt_f, sep='\t', index=False)



f=os.path.join(path, "matrix.mtx.gz")
mtx = pd.read_csv(gzip.open(f, 'rb'), sep=' ', \
    skiprows=3, names=["i","j"]+ct_header)

n_pixel = 0
f=os.path.join(path, "barcodes.tsv.gz")
brc = pd.DataFrame()
for sub in pd.read_csv(gzip.open(f, 'rb'), sep='\t|,', \
    names=["barcode","j","v2","lane","tile","X","Y"]+ct_header,\
    usecols=["j","X","Y"], engine='python', chunksize=500000):
    sub['x'] = sub.X.values * mu_scale
    sub['y'] = sub.Y.values * mu_scale
    dv, iv = ref[lane].query(X=sub[['x','y']], k=1, return_distance=True, sort_results=False)
    dv = dv.squeeze()
    npixel = (dv < radius).sum()
    brc = pd.concat([brc, sub.loc[dv < radius, ["j","X","Y"]]])
    logging.info(f"Read {sub.shape[0]} pixels, kept {npixel}")
    if args.debug:
        break

logging.info(f"Kept {brc.shape[0]} pixels")

f=os.path.join(path, "features.tsv.gz")
feature = pd.read_csv(gzip.open(f, 'rb'), sep='\t', \
    names=["gene_id","gene","i","ct"], usecols=["i","gene","gene_id"],  engine='python')

sub = mtx.merge(right = brc, on = 'j', how = 'inner')
sub = sub.merge(right = feature, on = 'i', how = 'inner')
sub.drop(columns = ['i', 'j'], inplace=True)
sub['j'] = sub.X.astype(str) + '_' + sub.Y.astype(str)
if len(gene_kept) > 0:
    sub = sub.loc[sub.gene.isin(gene_kept), :]

sub['#lane'] = lane
sub['tile']  = tile
sub = sub.loc[:, output_header]

if args.merge_raw_data and not args.debug:
    sub.sort_values(by = ['Y','X','gene'], inplace=True)
    sub.to_csv(raw_f, sep='\t', index=False, header=True)

feature_sub = sub.groupby(by = ['gene', 'gene_id']).agg({x:sum for x in ct_header}).reset_index()
feature_tot_ct = pd.concat((feature_tot_ct, feature_sub))
feature_tot_ct = feature_tot_ct.groupby(by = ['gene', 'gene_id']).agg({x:sum for x in ct_header}).reset_index()
sub.sort_values(by = ['Y','X','gene'], inplace=True)
n_pixel += sub.shape[0]
sub.to_csv(flt_f, mode='a', sep='\t', index=False, header=False)
logging.info(f"Write {n_pixel} pixels")


feature_tot_ct = feature_tot_ct.groupby(by = ['gene', 'gene_id']).agg({x:sum for x in ct_header}).reset_index()
f = outpath+"/feature."+args.identifier+".tsv.gz"
feature_tot_ct.to_csv(f, sep='\t', index=False, header=True)
logging.info(f"Finish filtering")
