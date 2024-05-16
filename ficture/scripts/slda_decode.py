import sys, os, argparse, logging, gzip, copy, re, time, warnings, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from datetime import datetime

from ficture.models.online_slda import OnlineLDA
from ficture.loaders.pixel_loader import PixelMinibatch

def slda_decode(_args):

    parser = argparse.ArgumentParser(prog="slda_decode")
    # Innput and output info
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--model', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--anchor', type=str, help='')
    parser.add_argument('--anchor_in_um', action='store_true')
    parser.add_argument('--feature', type=str, default='', help='')

    # Data realted parameters
    parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
    parser.add_argument('--key', type=str, default = 'gn', help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
    parser.add_argument('--batch_id', type=str, default = 'random_index', help='Input has to have a column with this name indicating the minibatch id')
    parser.add_argument('--precision', type=float, default=.1, help='If positive, collapse pixels within X um.')

    # Learning related parameters
    parser.add_argument('--thread', type=int, default=1, help='')
    parser.add_argument('--neighbor_radius', type=float, default=25, help='The radius (um) of each anchor point\'s territory')
    parser.add_argument('--halflife', type=float, default=0.7, help='Control the decay of distance-based weight')
    parser.add_argument('--theta_init_bound_multiplier', type=float, default=.2, help='')
    parser.add_argument('--inner_max_iter', type=int, default=30, help='')
    parser.add_argument('--model_scale', type=float, default=-1, help='')
    parser.add_argument('--seed', type=int, default=-1, help='')

    # Other
    parser.add_argument('--lite_topk_output_pixel', type=int, default=-1)
    parser.add_argument('--lite_topk_output_anchor', type=int, default=-1)
    parser.add_argument('--log', type=str, default = '', help='files to write log to')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args(_args)
    if len(_args) == 0:
        parser.print_help()
        return

    if args.log != '':
        try:
            logging.basicConfig(filename=args.log, filemode='a', encoding='utf-8', level=logging.INFO)
        except:
            logging.basicConfig(level= getattr(logging, "INFO", None))
    else:
        logging.basicConfig(level= getattr(logging, "INFO", None))

    if not os.path.exists(args.model):
        sys.exit("ERROR: cannot find model file")
    if not os.path.exists(args.input):
        sys.exit("ERROR: cannot find input file")
    if not os.path.exists(args.anchor):
        sys.exit("ERROR: cannot find anchor file")

    ## obtain seed if not provided
    seed = int(args.seed)
    if seed <= 0:
        seed = int(datetime.now().timestamp()) % 2147483648

    ### Basic parameterse
    mu_scale = 1./args.mu_scale
    radius = args.neighbor_radius
    precision = args.precision
    key = args.key.lower()
    batch_id = args.batch_id.lower()
    chunk_size = 500000

    ### Load model
    factor_names = []
    if args.model.endswith(".tsv.gz") or args.model.endswith(".tsv"):
        # If input is a gzip tsv file
        model = pd.read_csv(args.model, sep='\t')
        factor_names = list(model.columns)[1:]
        feature_kept = model.gene.values
        model = np.array(model.iloc[:,1:]).T
        model = np.clip(model, 0.5, None)
    else:
        # If input is a pickled model object
        try:
            model = pickle.load(open( args.model, "rb" ))
            feature_kept = np.array(model.feature_names_in_)
            model = model.components_
        except:
            sys.exit("ERROR: --model input should be either a tsv file containing gene names and factor profiles, or a pickled model object with at least two attributes, feature_names_in_ and components_, like those defined in scikit-learn LDA model")
    if os.path.isfile(args.feature):
        feature = pd.read_csv(args.feature, sep='\t')
        kept_idx = np.where(np.in1d(feature_kept, feature.gene.values))[0]
        feature_kept = np.array(feature_kept)[kept_idx]
        model = model[:, kept_idx]

    K, M = model.shape
    ft_dict = {x:i for i,x in enumerate( feature_kept ) }
    if args.model_scale > 0:
        model = normalize(model, norm='l1', axis=1) * args.model_scale
    logging.info(f"{M} genes and {K} factors are read from input model")

    slda = OnlineLDA(vocab=feature_kept, K=K, N=1e6, iter_inner=args.inner_max_iter, verbose = 1, seed = seed)
    slda.init_global_parameter(model)
    init_bound = 1./K * args.theta_init_bound_multiplier

    ### Input pixel info (input has to contain certain columns with correct header)
    with gzip.open(args.input, 'rt') as rf:
        oheader = rf.readline().strip().split('\t')
    oheader = [x.lower() if len(x) > 1 else x.upper() for x in oheader]
    input_header = [batch_id,"X","Y","gene",key]
    dty = {x:float for x in ['X','Y']}
    dty[key] = int
    dty.update({x:str for x in [batch_id, 'gene']})
    mheader = [x for x in input_header if x not in oheader]
    if len(mheader) > 0:
        mheader = ", ".join(mheader)
        sys.exit(f"Input misses the following column: {mheader}.")

    pixel_reader = pd.read_csv(args.input, sep='\t', chunksize=chunk_size, \
                skiprows=1, names=oheader, usecols=input_header, dtype=dty)

    pixel_obj = PixelMinibatch(pixel_reader, ft_dict, \
                            batch_id, key, mu_scale, \
                            radius=radius, halflife=args.halflife,\
                            precision=args.precision, thread=args.thread)
    ### anchor info
    pixel_obj.load_anchor(args.anchor, args.anchor_in_um)
    logging.info(f"Read {pixel_obj.grid_info.shape[0]} grid points")
    factor_header = pixel_obj.factor_header

    post_count = np.zeros((K, M))
    n_batch = 0
    while True:
        read_n_batch = pixel_obj.read_chunk(args.thread)
        logging.info(f"Read {read_n_batch} batches ({pixel_obj.dge_mtx.shape})")
        pcount, pixel, anchor  = pixel_obj.run_chunk(slda, init_bound)
        pixel.X = pixel.X.map('{:.2f}'.format)
        pixel.Y = pixel.Y.map('{:.2f}'.format)
        if args.lite_topk_output_pixel > 0 and args.lite_topk_output_pixel < K:
            X = pixel[factor_header].values
            partial_indices = np.argpartition(X, -args.lite_topk_output_pixel, axis=1)[:, -args.lite_topk_output_pixel:]
            sorted_top_indices = np.argsort(X[np.arange(X.shape[0])[:, None], partial_indices], axis=1)[:, ::-1]
            top_indices = partial_indices[np.arange(partial_indices.shape[0])[:, None], sorted_top_indices]
            top_values = X[np.arange(X.shape[0])[:, None], top_indices]
            for k in range(args.lite_topk_output_pixel):
                pixel[f"K{k+1}"] = top_indices[:, k]
            for k in range(args.lite_topk_output_pixel):
                pixel[f"P{k+1}"] = np.clip(top_values[:, k], 0, 1)
            pixel.drop(columns = factor_header, inplace=True)
        write_mode = 'w' if n_batch == 0 else 'a'
        header_include = True if n_batch == 0 else False
        pixel.to_csv(args.output+".pixel.tsv.gz", sep='\t', index=False, header=header_include, mode=write_mode, float_format="%.2e", compression={"method":"gzip"})
        logging.info(f"Output {pixel.shape[0]} pixels and {anchor.shape[0]} anchors")
        anchor.X = anchor.X.map('{:.2f}'.format)
        anchor.Y = anchor.Y.map('{:.2f}'.format)
        if args.lite_topk_output_anchor > 0 and args.lite_topk_output_anchor < K:
            X = anchor[factor_header].values
            partial_indices = np.argpartition(X, -args.lite_topk_output_anchor, axis=1)[:, -args.lite_topk_output_anchor:]
            sorted_top_indices = np.argsort(X[np.arange(X.shape[0])[:, None], partial_indices], axis=1)[:, ::-1]
            top_indices = partial_indices[np.arange(partial_indices.shape[0])[:, None], sorted_top_indices]
            top_values = X[np.arange(X.shape[0])[:, None], top_indices]
            for k in range(args.lite_topk_output_anchor):
                anchor[f"K{k+1}"] = top_indices[:, k]
            for k in range(args.lite_topk_output_anchor):
                anchor[f"P{k+1}"] = np.clip(top_values[:, k], 0, 1)
            anchor.drop(columns = factor_header, inplace=True)
        anchor.to_csv(args.output+".anchor.tsv.gz", sep='\t', index=False, header=header_include, mode=write_mode, float_format="%.2e", compression={"method":"gzip"})
        n_batch += read_n_batch
        post_count += pcount
        if not pixel_obj.file_is_open:
            break

    ### Output posterior summaries
    if len(factor_names) == K:
        factor_header = factor_names
    out_f = args.output + ".posterior.count.tsv.gz"
    pd.concat([pd.DataFrame({'gene': feature_kept}),\
            pd.DataFrame(post_count.T, dtype='float64',\
                            columns = factor_header)],\
            axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})

if __name__ == "__main__":
    slda_decode(sys.argv[1:])
