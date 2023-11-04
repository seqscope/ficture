'''
Transform into factor space based on input model
Model contains gene names and either Dirichlet parameters (or probabilities for more general use?)
Input pixel level data will be grouped into (overlapping) hexagons
'''
import sys, os, copy, gzip, time, logging, pickle, argparse
import numpy as np
import pandas as pd
import random as rng

from ficture.loaders.pixel_to_unit_loader import PixelToUnit
from ficture.models.online_lda import OnlineLDA

def transform(_args):

    parser = argparse.ArgumentParser(prog = "transform")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output_pref', type=str, help='')
    parser.add_argument('--model', type=str, help='')

    parser.add_argument('--key', type=str, default = 'gn', help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
    parser.add_argument('--major_axis', type=str, default=None, help='X or Y')
    parser.add_argument('--region_id', type=str, default=None, help='')
    parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
    parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
    parser.add_argument('--thread', type=int, default=-1, help='')
    parser.add_argument('--n_move', type=int, default=3, help='')
    parser.add_argument('--hex_width', type=int, default=24, help='')
    parser.add_argument('--hex_radius', type=int, default=-1, help='')
    parser.add_argument('--precision', type=int, default=1, help='Number of digits to store spatial location (in um), 0 for integer.')
    parser.add_argument('--chunksize', type=int, default=100000, help='Number of lines to read at a time')

    args = parser.parse_args(_args)
    logger = logging.getLogger(__name__)
    if not os.path.exists(args.input):
        sys.exit("ERROR: cannot find input file.")
    if not os.path.exists(args.model):
        sys.exit("ERROR: cannot find model file.")
    if args.hex_width <= 0 and args.hex_radius <= 0:
        sys.exit("ERROR: invalid hex_width or hex_radius")
    if not os.path.exists(os.path.dirname(args.output_pref)):
        os.makedirs(os.path.dirname(args.output_pref))

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
    if args.model.endswith('.tsv.gz') or args.model.endswith('tsv'):
        model_mtx = pd.read_csv(args.model, sep='\t')
        feature_kept=list(model_mtx.gene)
        model_mtx = np.array(model_mtx.iloc[:, 1:])
        M, K = model_mtx.shape
        model = OnlineLDA(vocab=feature_kept,K=K,N=1e4,thread=args.thread,tol=1e-3)
        model.init_global_parameter(model_mtx.T)
    else:
        try:
            model = pickle.load(open(args.model, 'rb'))
            model.n_jobs = args.thread
            feature_kept = model.feature_names_in_
            K, M = model.components_.shape
            model.feature_names_in_ = None
        except:
            sys.exit("ERROR: --model must be either a tsv file containing gene name and factor-gene loadings or a pickle model object from sklearn LDA")
    ft_dict = {x:i for i,x in enumerate(feature_kept)}
    logger.info(f"Model loaded with {M} features and {K} factors")

    ### Basic parameterse
    factor_header = [str(x) for x in range(K)]
    radius = args.hex_radius
    if radius < 0:
        radius = args.hex_width / np.sqrt(3)
    b_size = radius * 10

    # Pixel reader
    batch_obj = PixelToUnit(reader, ft_dict, key, radius,\
                scale=1./args.mu_scale, min_ct_per_unit=args.min_ct_per_unit,\
                sliding_step=args.n_move, major_axis=args.major_axis)

    # Transform
    post_count = np.zeros((K, M))
    n_unit = 0
    n_batch= 0
    oheader = ["unit",key,"x","y","topK","topP"]+factor_header
    out_f = args.output_pref + ".fit_result.tsv.gz"
    with gzip.open(out_f, 'wt') as wf:
        wf.write('\t'.join(oheader) + '\n')
    t0 = time.time()
    while batch_obj.read_chunk(min_size=b_size):
        theta = model.transform(batch_obj.mtx)
        post_count += np.array(theta.T @ batch_obj.mtx)
        n_batch += 1
        n_unit  += theta.shape[0]
        t1 = time.time() - t0
        batch_obj.brc['topK'] = np.argmax(theta, axis = 1)
        batch_obj.brc['topP'] = theta.max(axis = 1)
        batch_obj.brc = pd.concat([batch_obj.brc, pd.DataFrame(theta, columns=factor_header)], axis=1)
        batch_obj.brc['x'] = batch_obj.brc.x.map(lambda x : f"{x:.{args.precision}f}")
        batch_obj.brc['y'] = batch_obj.brc.y.map(lambda x : f"{x:.{args.precision}f}")
        batch_obj.brc.rename(columns = {'hex_id':'unit'}, inplace=True)
        batch_obj.brc[oheader].to_csv(out_f, sep='\t', index=False, header=False, float_format='%.3e', mode='a', compression={"method":"gzip"})
        logger.info(f"Transformed {n_batch} batches with total {n_unit} units, {t1/60:2f}min")

    out_f = args.output_pref + ".posterior.count.tsv.gz"
    pd.concat([pd.DataFrame({'gene': feature_kept}),\
            pd.DataFrame(post_count.T, dtype='float64',\
                            columns = [str(k) for k in range(K)])],\
                axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})
