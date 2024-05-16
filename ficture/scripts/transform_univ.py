'''
Transform into factor space based on input model
Model contains gene names and either Dirichlet parameters (or probabilities for more general use?)
Input pixel level data will be grouped into (overlapping) hexagons
'''
import sys, os, copy, gzip, time, logging, pickle, argparse
import numpy as np
import pandas as pd
import random as rng
from sklearn.preprocessing import normalize
from sklearn.decomposition import LatentDirichletAllocation as LDA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.decomposition._online_lda_fast import _dirichlet_expectation_2d

from ficture.loaders.pixel_to_unit_loader import PixelToUnit
from ficture.utils.utilt import init_latent_vars

def transform(_args):

    parser = argparse.ArgumentParser(prog = "transform")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', '--output_pref', type=str, help='')
    parser.add_argument('--model', type=str, help='')
    parser.add_argument('--feature', type=str, default='', help='')
    parser.add_argument('--key', type=str, default = 'gn', help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
    parser.add_argument('--major_axis', type=str, default=None, help='X or Y')
    parser.add_argument('--region_id', type=str, default=None, help='')
    parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
    parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
    parser.add_argument('--thread', type=int, default=-1, help='')
    parser.add_argument('--n_move', type=int, default=3, help='')
    parser.add_argument('--hex_width', type=float, default=24, help='')
    parser.add_argument('--hex_radius', type=float, default=-1, help='')
    parser.add_argument('--precision', type=int, default=1, help='Number of digits to store spatial location (in um), 0 for integer.')
    parser.add_argument('--chunksize', type=int, default=1000000, help='Number of lines to read at a time')
    parser.add_argument('--xy_median', action='store_true', help='Output the median of pixel cooredinates inside each hexagon, default is to output hexagon centers exactly as lattice points')
    parser.add_argument('--log_norm', action='store_true', help='')
    parser.add_argument('--log_norm_size_factor', action='store_true', help='')
    parser.add_argument('--scale_const', type=float, default=-1, help='')
    parser.add_argument('--unit_sum_mean', type=float, default=-1, help='')
    parser.add_argument('--debug', type=int, default=0, help='')

    args = parser.parse_args(_args)
    if len(_args) == 0:
        parser.print_help()
        return

    if not os.path.exists(args.input):
        sys.exit("ERROR: cannot find input file.")
    if not os.path.exists(args.model):
        sys.exit("ERROR: cannot find model file.")
    if args.hex_width <= 0 and args.hex_radius <= 0:
        sys.exit("ERROR: invalid hex_width or hex_radius")
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    logging.basicConfig(level= getattr(logging, "INFO", None))

    ### Input
    with gzip.open(args.input, 'rt') as rf:
        header=rf.readline().strip().split('\t')
    key = args.key.lower()
    header=[x.upper() if len(x) == 1 else x.lower() for x in header]
    usecol = ['X','Y','gene',key]
    adt={'X':float,'Y':float,'gene':str,args.key:int}
    if args.region_id is not None:
        usecol.append(args.region_id)
        adt[args.region_id] = str
    miss_header = [x for x in usecol if x not in header]
    if len(miss_header) > 0:
        sys.exit(f"ERROR: {miss_header} is not in the input file")
    reader = pd.read_csv(gzip.open(args.input, 'rt'), sep='\t',\
                        chunksize=args.chunksize, skiprows=1, names=header,\
                        usecols=usecol, dtype=adt)

    ### Model
    factor_header = []
    if args.model.endswith('.tsv.gz') or args.model.endswith('tsv'):
        model_mtx = pd.read_csv(args.model, sep='\t', index_col = 0)
        factor_header = list(model_mtx.columns)
        feature_kept =list(model_mtx.index)
        M, K = model_mtx.shape
        model = LDA(n_components=K, learning_method='online', batch_size=512, n_jobs = args.thread, verbose = 0)
        init_latent_vars(model, n_features = M, gamma = np.array(model_mtx).T)
    else:
        try:
            model = pickle.load(open(args.model, 'rb'))
            model.n_jobs = args.thread
            feature_kept = model.feature_names_in_
            K, M = model.components_.shape
            model.feature_names_in_ = None
            factor_header = list(np.arange(K).astype(str) )
        except:
            sys.exit("ERROR: --model must be either a tsv file containing gene name and factor-gene loadings or a pickle model object from sklearn LDA")
    if os.path.isfile(args.feature):
        feature = pd.read_csv(args.feature, sep='\t')
        kept_idx = np.where(np.in1d(feature_kept, feature.gene.values))[0]
        feature_kept = np.array(feature_kept)[kept_idx]
        model.components_ = model.components_[:, kept_idx]
        model.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(model.components_)
        )
        M = len(feature_kept)

    ft_dict = {x:i for i,x in enumerate(feature_kept)}
    logging.info(f"Model loaded with {M} features and {K} factors")

    ### Basic parameterse
    radius = args.hex_radius
    if radius < 0:
        radius = args.hex_width / np.sqrt(3)
    b_size = radius * 20
    scale_const = args.scale_const
    unit_sum_mean = args.unit_sum_mean
    if args.log_norm_size_factor:
        args.log_norm = True
    if args.log_norm:
        if hasattr(model, 'log_norm_scaling_const_'):
            scale_const = model.log_norm_scaling_const_
        if scale_const < 0:
            sys.exit("ERROR: --log_norm is specified but the normalization constant used in model fitting is unknown.")
        if args.log_norm_size_factor:
            if hasattr(model, 'unit_sum_mean_'):
                unit_sum_mean = model.unit_sum_mean_
            if unit_sum_mean < 0:
                sys.exit("ERROR: --log_norm_size_factor is specified but the size constant used in model fitting is unknown.")

    # Pixel reader
    batch_obj = PixelToUnit(reader, ft_dict, key, radius,\
                scale=1./args.mu_scale, min_ct_per_unit=args.min_ct_per_unit,\
                sliding_step=args.n_move, major_axis=args.major_axis,\
                xy_lattice=(not args.xy_median))

    # Transform
    post_count = np.zeros((K, M))
    n_unit = 0
    n_batch= 0
    oheader = ["unit",key,"x","y","topK","topP"]+[str(x) for x in range(K)]
    out_f = args.output + ".fit_result.tsv.gz"
    with gzip.open(out_f, 'wt') as wf:
        wf.write('\t'.join(oheader) + '\n')
    t0 = time.time()
    last_batch = set()
    while batch_obj.read_chunk(min_size=b_size):
        if args.log_norm_size_factor:
            rsum = batch_obj.mtx.sum(axis=1) / unit_sum_mean
            mtx = batch_obj.mtx / rsum.reshape((-1,1))
            mtx.data = np.log(mtx.data + 1) / scale_const
            mtx = mtx.tocsr()
        elif args.log_norm:
            mtx = normalize(batch_obj.mtx, norm='l1', axis=1)
            mtx.data = np.log(mtx.data + 1) / scale_const
        else:
            mtx = batch_obj.mtx
        theta = model.transform(mtx)
        post_count += np.array(theta.T @ batch_obj.mtx)
        n_batch += 1
        n_unit  += theta.shape[0]
        t1 = time.time() - t0
        batch_obj.brc['topK'] = np.argmax(theta, axis = 1)
        batch_obj.brc['topP'] = theta.max(axis = 1)
        batch_obj.brc = pd.concat([batch_obj.brc, pd.DataFrame(theta, columns=[str(x) for x in range(K)] )], axis=1)
        batch_obj.brc['x'] = batch_obj.brc.x.map(lambda x : f"{x:.{args.precision}f}")
        batch_obj.brc['y'] = batch_obj.brc.y.map(lambda x : f"{x:.{args.precision}f}")
        batch_obj.brc.rename(columns = {'hex_id':'unit'}, inplace=True)
        if len(last_batch) > 0:
            batch_obj.brc.drop(index = batch_obj.brc.index[batch_obj.brc.unit.isin(last_batch)], inplace=True)
        last_batch = set(batch_obj.brc.unit.values)
        batch_obj.brc[oheader].to_csv(out_f, sep='\t', index=False, header=False, float_format='%.3e', mode='a', compression={"method":"gzip"})
        logging.info(f"Transformed {n_batch} batches with total {n_unit} units, {t1/60:2f}min")
        if (args.debug > 0) and (n_unit >= args.debug):
            break

    out_f = args.output + ".posterior.count.tsv.gz"
    pd.concat([pd.DataFrame({'gene': feature_kept}),\
            pd.DataFrame(post_count.T, dtype='float64',\
                            columns = factor_header)],\
                axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})

if __name__ == '__main__':
    transform(sys.argv[1:])
