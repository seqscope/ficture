import sys, os, gzip, gc, argparse, warnings, logging
import numpy as np
import pandas as pd
import sklearn.neighbors
import sklearn.mixture

from ficture.utils.filter_fn import filter_by_density_mixture

def filter_by_density_v1(_args):

    parser = argparse.ArgumentParser(prog = "filter_by_density")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--ref_pt', type=str, help='')
    parser.add_argument('--filter_based_on', type=str, default="Count", help='')
    parser.add_argument('--feature', type=str, default='', help='')
    parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
    parser.add_argument('--filter_batch_size', type=float, default=5000, help='In um, along the streaming axis')
    parser.add_argument('--filter_batch_ncut', type=int, default=1, help='')
    parser.add_argument('--major_axis', type=str, default="X", help='')
    parser.add_argument('--precision_um', type=float, default=2, help='')
    parser.add_argument('--log', default = '', type=str, help='files to write log to')
    parser.add_argument('--max_npts_to_fit_model', type=float, default=1e6, help='')
    parser.add_argument('--min_abs_mol_density_squm_dense', type=float, default=0.2, help='Lowerbound for dense tissue region')
    parser.add_argument('--min_abs_mol_density_squm', type=float, default=0.02, help='A safe lowerbound to remove very sparse technical noise')
    parser.add_argument('--hard_threshold', type=float, default=-1, help='If provided, filter by hard threshold (number of molecules per squared um)')
    parser.add_argument('--radius', type=float, default=7, help='')
    parser.add_argument('--hex_n_move', type=int, default=6, help='')

    parser.add_argument('--xmin', type=float, default=-1, help='In um')
    parser.add_argument('--xmax', type=float, default=np.inf, help='In um')
    parser.add_argument('--ymin', type=float, default=-1, help='In um')
    parser.add_argument('--ymax', type=float, default=np.inf, help='In um')

    parser.add_argument('--redo_filter', action='store_true')
    parser.add_argument('--anchor_only', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args(_args)
    if len(_args) == 0:
        parser.print_help()
        return

    if os.path.exists(args.output):
        warnings.warn("Output file already exists")
    if not os.path.exists(args.input):
        sys.exit("Cannot find input file")
    major_axis = args.major_axis.upper()
    if major_axis not in ["X", "Y"]:
        sys.exit("Invalid --major_axis")
    minor_axis = "X" if major_axis == "Y" else "Y"
    if args.xmax <= 0:
        args.xmax = np.inf
    if args.ymax <= 0:
        args.ymax = np.inf

    if args.log != '':
        try:
            logging.basicConfig(filename=args.log, filemode='a', encoding='utf-8', level=logging.INFO)
        except:
            logging.basicConfig(level= getattr(logging, "INFO", None))
    else:
        logging.basicConfig(level= getattr(logging, "INFO", None))

    key      = args.filter_based_on
    mu_scale = 1./args.mu_scale
    n_move   = args.hex_n_move
    radius   = args.radius
    hex_area = radius**2*3*np.sqrt(3)/2
    chunk_size = 1000000

    gene_kept = set()
    if os.path.exists(args.feature):
        feature = pd.read_csv(args.feature, sep='\t', header=0)
        gene_kept = set(feature.gene.values)

    if os.path.exists(args.ref_pt) and not args.redo_filter:
        pt = pd.read_csv(args.ref_pt, sep='\t')
        logging.info(f"Read existing anchor points")
    else:
        pt = pd.DataFrame()
        df=pd.DataFrame()
        for chunk in pd.read_csv(gzip.open(args.input, 'rb'),\
            sep='\t', usecols=["Y","X","gene",key], chunksize=chunk_size):
            full = chunk.shape[0] == chunk_size
            if len(gene_kept) != 0:
                chunk = chunk.loc[chunk.gene.isin(gene_kept)]
            chunk['X'] = chunk.X.values * mu_scale
            chunk['Y'] = chunk.Y.values * mu_scale
            chunk = chunk.loc[(chunk.X >= args.xmin) & (chunk.X <= args.xmax) &\
                            (chunk.Y >= args.ymin) & (chunk.Y <= args.ymax)]
            if chunk.shape[0] == 0:
                continue
            if args.precision_um > 0:
                chunk['X'] = np.around(chunk.X.values/args.precision_um,0).astype(int)*args.precision_um
                chunk['Y'] = np.around(chunk.Y.values/args.precision_um,0).astype(int)*args.precision_um
            chunk = chunk.groupby(by = ["X", "Y"]).agg({key:sum}).reset_index()
            df = pd.concat([df, chunk])
            hst, hed = df[major_axis].iloc[0], df[major_axis].iloc[-1]
            wst, wed = df[minor_axis].min(), df[minor_axis].max()
            logging.info(f"Current chunk size {hed-hst} x {wed-wst}, collapsed into {df.shape[0]} pts")

            if hed - hst < args.filter_batch_size * .9 and full:
                continue

            st = hst
            ed = st + args.filter_batch_size
            wstep = (wed - wst + 1) / args.filter_batch_ncut
            df['win'] = ((df[minor_axis] - wst)/wstep).astype(int).astype(str)
            ### Detect grid points falling inside dense tissue region
            for w in df.win.unique():
                indx = df.win.eq(w)
                xmin, ymin = df.loc[indx, ['X','Y']].min()
                xmax, ymax = df.loc[indx, ['X','Y']].max()
                sub, m0, m1 = filter_by_density_mixture(df.loc[indx, :], key, radius, n_move, args)
                pt = pd.concat([pt, sub])
                logging.info(f"Window {str(w)} ({xmax-xmin:.1f} X {ymax-ymin:.1f} ):\t{m0:.3f} v.s. {m1:.3f}")
            df=pd.DataFrame()
            if args.debug:
                break

        if not args.debug:
            pt.x = np.around(np.clip(pt.x.values,0,np.inf)/args.precision_um,0).astype(int)
            pt.y = np.around(np.clip(pt.y.values,0,np.inf)/args.precision_um,0).astype(int)
            pt.drop_duplicates(inplace=True)
            pt.x = pt.x * args.precision_um
            pt.y = pt.y * args.precision_um
            pt.to_csv(args.ref_pt, sep='\t', index=False, header=True)

        logging.info(f"Finished identifying anchor points in tissue region.")

    if args.anchor_only:
        sys.exit()


    ref=sklearn.neighbors.BallTree(np.array(pt.loc[:, ['x','y']]))
    with gzip.open(args.input, 'rt') as rf:
        header = rf.readline()
    if args.output.endswith(".gz"):
        with gzip.open(args.output, 'wt') as wf:
            _=wf.write(header)
    else:
        with open(args.output, 'w') as wf:
            _=wf.write(header)

    for chunk in pd.read_csv(gzip.open(args.input, 'rb'),\
        sep='\t', header=0, chunksize=chunk_size):
        if len(gene_kept) != 0:
            chunk = chunk.loc[chunk.gene.isin(gene_kept)]
            if chunk.shape[0] == 0:
                continue
        x = chunk.X.values * mu_scale
        y = chunk.Y.values * mu_scale
        chunk = chunk.loc[ (x >= args.xmin) & (x <= args.xmax) &\
                        (y >= args.ymin) & (y <= args.ymax) ]
        if chunk.shape[0] == 0:
            continue
        dv, iv = ref.query(X=np.array(chunk.loc[:, ["X","Y"]]) * mu_scale, \
                        k=1, return_distance=True, sort_results=False)
        dv = dv.squeeze()
        chunk = chunk.loc[dv < radius, :]
        if chunk.shape[0] == 0:
            continue
        logging.info(f"Output {chunk.shape[0]} rows ...")
        chunk.to_csv(args.output, mode='a', sep='\t', index=False, header=False)
        if args.debug:
            break

if __name__ == "__main__":
    filter_by_density_v1(sys.argv[1:])
