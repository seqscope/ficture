import sys, os, gzip, copy, gc, logging
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import *
import subprocess as sp
import geojson
import shapely.prepared, shapely.geometry
from shapely.geometry import Point

# Add parent directory
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from hexagon_fn import *
from ficture.utils.hexagon_fn import pixel_to_hex, hex_to_pixel

def make_sge_by_hexagon(_args):

    parser = argparse.ArgumentParser(prog = "make_sge_by_hexagon")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--feature', type=str, help='')
    parser.add_argument('--output_path', type=str, help='')
    parser.add_argument('--boundary', type=str, default = '', help='')

    parser.add_argument('--major_axis', type=str, default="Y", help='X or Y')
    parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
    parser.add_argument('--key', default = 'gn', type=str, help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
    parser.add_argument('--precision', type=int, default=1, help='Number of digits to store spatial location (in um), 0 for integer.')
    parser.add_argument('--hex_width', type=int, default=24, help='')
    parser.add_argument('--hex_radius', type=int, default=-1, help='')
    parser.add_argument('--overlap', type=float, default=-1, help='')
    parser.add_argument('--n_move', type=int, default=1, help='')
    parser.add_argument('--min_ct_density', type=float, default=0.1, help='Minimum density of output hexagons, in nUMI/um^2')
    parser.add_argument('--min_ct_per_unit', type=float, default=-1, help='Minimum umi count of output hexagons')
    parser.add_argument('--transfer_gene_prefix', action="store_true", help='')
    args = parser.parse_args(_args)

    if len(_args) == 0:
        parser.print_help()
        return

    if not os.path.exists(args.input):
        sys.exit(f"ERROR: cannot find input file \n {args.input}")
    logging.basicConfig(level= getattr(logging, "INFO", None))

    mu_scale = 1./args.mu_scale
    radius=args.hex_radius
    b_size = 512

    diam=args.hex_width
    radius=args.hex_radius
    if radius < 0:
        radius = diam / np.sqrt(3)
    else:
        diam = int(radius*np.sqrt(3))
    if args.overlap >= 0 and args.overlap < 1:
        n_move = int(1 / (1. - args.overlap) )
    else:
        n_move = args.n_move
        if n_move < 0:
            n_move = 1

    hex_area = diam*radius*3/2

    min_ct_per_unit = args.min_ct_per_unit
    if args.min_ct_per_unit < 0:
        min_ct_per_unit = args.min_ct_density * hex_area

    ### Output
    if not os.path.exists(args.output_path):
        arg="mkdir -p " + args.output_path
        os.system(arg)

    with gzip.open(args.input, 'rt') as rf:
        input_header=rf.readline().strip().split('\t')

    feature=pd.read_csv(args.feature,sep='\t',header=0,usecols=['gene', 'gene_id', args.key])
    feature.sort_values(by=args.key, ascending=False, inplace=True)
    feature.drop_duplicates(subset='gene', inplace=True)
    if args.transfer_gene_prefix:
        prefix = feature.gene.map(lambda x: x.split('_')[0] + '_' if '_' in x else '').values
        feature.gene_id = prefix + feature.gene_id.values
    feature['dummy'] = "Gene Expression"
    f = args.output_path + "/features.tsv.gz"
    feature[['gene_id','gene','dummy']].to_csv(f, sep='\t', index=False, header=False)

    feature_kept = list(feature.gene.values)
    ft_dict = {x:i for i,x in enumerate( feature_kept ) }
    M = len(feature_kept)

    brc_f = args.output_path + "/barcodes.tsv"
    mtx_f = args.output_path + "/matrix.mtx"
    # If exists, delete
    if os.path.exists(brc_f):
        _ = os.system("rm " + brc_f)
    if os.path.exists(mtx_f):
        _ = os.system("rm " + mtx_f)

    use_boundary = False
    if os.path.isfile(args.boundary):
        try:
            mpoly = shapely.geometry.shape(geojson.load(open(args.boundary, 'rb')))
        except:
            gj = geojson.load(open(args.boundary, 'rb'))
            mpoly = shapely.geometry.shape(gj['features'][0]['geometry'] )
        mpoly = shapely.prepared.prep(mpoly)
        use_boundary = True
        logging.info(f"Load boundary from {args.boundary}")

    n_unit = 0
    T = 0

    adt = {x:float for x in ['X','Y', args.key]}
    adt.update({x:str for x in ['gene', 'gene_id']})
    df_buff = pd.DataFrame()
    for chunk in pd.read_csv(gzip.open(args.input, 'rt'),sep='\t',chunksize=1000000, header=0, usecols=['X','Y','gene','gene_id',args.key], dtype=adt):
        if chunk.shape[0] == 0:
            break
        ed = chunk[args.major_axis].iloc[-1]
        left = copy.copy(chunk[chunk[args.major_axis].gt(ed - 5 * args.mu_scale)])
        df_buff = pd.concat([df_buff, chunk])
        df_buff['j'] = df_buff.X.astype(str) + '_' + df_buff.Y.astype(str)
        df = df_buff[df_buff.gene.isin(ft_dict)]
        brc_full = df.groupby(by = ['j','X','Y']).agg({args.key: sum}).reset_index()
        brc_full.index = range(brc_full.shape[0])
        brc_full['crd'] = None
        pts = np.asarray(brc_full[['X','Y']]) * mu_scale
        logging.info(f"Read data with {brc_full.shape[0]} pixels and {feature.shape[0]} genes.")

        # Make DGE
        barcode_kept = list(brc_full.j.values)
        bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
        indx_row = [ bc_dict[x] for x in df['j']]
        indx_col = [ ft_dict[x] for x in df['gene']]
        N = len(barcode_kept)

        dge_mtx = coo_matrix((df[args.key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
        total_molecule=df[args.key].sum()
        logging.info(f"Made DGE {dge_mtx.shape}")

        offs_x = 0
        offs_y = 0
        while offs_x < n_move:
            while offs_y < n_move:
                x,y = pixel_to_hex(pts, radius, offs_x/n_move, offs_y/n_move)
                brc_full['crd'] = list(zip(x,y))
                hex_info = brc_full.groupby(by = 'crd').agg({args.key: sum})
                hex_info.rename(columns = {args.key:'tot'}, inplace=True)
                hex_info = hex_info.loc[hex_info.tot > args.min_ct_per_unit, :]
                if use_boundary:
                    hex_coord = np.array([[x,y] for x,y in hex_info.index.values])
                    x, y = hex_to_pixel(hex_coord[:, 0], hex_coord[:, 1], radius, offs_x/n_move, offs_y/n_move)
                    kept = [mpoly.contains(Point(*p)) for p in zip(x, y)]
                    logging.info(f"Keep {sum(kept)}/{len(x)} units within boundary.")
                    hex_info = hex_info.loc[kept, :]

                hex_id = set(hex_info.index.values)
                if len(hex_id) < 1:
                    offs_y += 1
                    continue
                hex_list = list(hex_id)
                hex_dict = {x:i for i,x in enumerate(hex_list)}

                # Hexagon and pixel correspondence
                sub = pd.DataFrame({'crd':brc_full.crd.values,'cCol':range(N)})
                sub = sub[sub.crd.isin(hex_id)]
                sub['cRow'] = sub.crd.map(hex_dict).astype(int)

                # Hexagon level info (for barcodes.tsv.gz)
                brc = brc_full[brc_full.crd.isin(hex_id)].groupby(by = 'crd').agg({'X':np.mean, 'Y':np.mean, args.key:sum}).reset_index()
                brc['cRow'] = brc.crd.map(hex_dict).astype(int)
                brc['X'] = [f"{x:.{args.precision}f}" for x in brc.X.values]
                brc['Y'] = [f"{x:.{args.precision}f}" for x in brc.Y.values]
                brc.sort_values(by = 'cRow', inplace=True)
                with open(brc_f, 'a') as wf:
                    _ = wf.write('\n'.join(\
                    (brc.cRow+n_unit+1).astype(str).values + '_' + \
                    str(offs_x) + '.' + str(offs_y) + '_' + \
                    brc.X.values + '_' + brc.Y.values + '_' + \
                    brc[args.key].astype(str).values) + '\n')
                n_hex = len(hex_dict)
                n_minib = n_hex // b_size + 1
                mid_ct = np.median(hex_info.tot.values)
                logging.info(f"{offs_x}, {offs_y}: {n_minib}, {n_hex} ({sub.cRow.max()}, {sub.shape[0]}), median count per unit {mid_ct}")

                # Output sparse matrix
                grd_minib = list(range(0, n_hex, b_size))
                grd_minib.append(n_hex)
                st_minib = 0
                n_minib = len(grd_minib) - 1
                while st_minib < n_minib:
                    indx_minib = (sub.cRow >= grd_minib[st_minib]) & (sub.cRow < grd_minib[st_minib+1])
                    npixel_minib = sum(indx_minib)
                    offset = sub.loc[indx_minib, 'cRow'].min()
                    nhex_minib = sub.loc[indx_minib, 'cRow'].max() - offset + 1
                    mtx = coo_matrix((np.ones(npixel_minib, dtype=bool), (sub.loc[indx_minib, 'cRow'].values-offset, sub.loc[indx_minib, 'cCol'].values)), shape=(nhex_minib, N) ).tocsr() @ dge_mtx
                    mtx.eliminate_zeros()
                    r, c = mtx.nonzero()
                    r = np.array(r,dtype=int) + offset + n_unit + 1
                    c = np.array(c,dtype=int) + 1
                    T += len(r)
                    mtx = pd.DataFrame({'i':c, 'j':r, 'v':mtx.data})
                    mtx['i'] = mtx.i.astype(int)
                    mtx['j'] = mtx.j.astype(int)
                    mtx['v'] = mtx.v.astype(int)
                    mtx.to_csv(mtx_f, mode='a', sep=' ', index=False, header=False)
                    #print(mtx.shape[0])
                    st_minib += 1
                    if ( st_minib % 50 == 0):
                        logging.info(f"{st_minib}/{n_minib}. Wrote {nhex_minib} units.")
                n_unit += brc.shape[0]
                logging.info(f"Sliding offset {offs_x}, {offs_y}. Wrote {n_unit} units so far.")
                offs_y += 1
            offs_y = 0
            offs_x += 1
        df_buff = copy.copy(left)

    _ = os.system("gzip -f " + brc_f)

    mtx_header = args.output_path + "/matrix.header"
    with open(mtx_header, 'w') as wf:
        line = "%%MatrixMarket matrix coordinate integer general\n%\n"
        line += " ".join([str(x) for x in [M, n_unit, T]]) + "\n"
        wf.write(line)

    arg = " ".join(["cat",mtx_header,mtx_f,"| gzip -c > ", mtx_f+".gz"])
    if os.system(arg) == 0:
        _ = os.system("rm " + mtx_f)
        _ = os.system("rm " + mtx_header)

if __name__ == "__main__":
    make_sge_by_hexagon(sys.argv[1:])