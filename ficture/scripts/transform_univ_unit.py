'''
Transform into factor space based on input model
Model contains gene names and either Dirichlet parameters (or probabilities for more general use?)
Input pixel level data will be grouped into (overlapping) hexagons
'''
import sys, os, copy, gzip, time, logging
import pickle, argparse
import numpy as np
import pandas as pd
import random as rng

from ficture.loaders.unit_loader import UnitLoader
from ficture.models.online_lda import OnlineLDA

def transform_unit(_args):

    parser = argparse.ArgumentParser(prog = "transform_unit")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output_pref', type=str, help='')
    parser.add_argument('--model', type=str, help='')

    parser.add_argument('--key', type=str, default = 'gn', help='gt: genetotal, gn: gene, spl: velo-spliced, unspl: velo-unspliced')
    parser.add_argument('--feature_id_map', type=str, default='', help='')
    parser.add_argument('--feature_id', type=str, default = 'gene', help='')
    parser.add_argument('--unit_id', type=str, default='random_index', help='')
    parser.add_argument('--unit_attr', type=str, nargs='+', default=[], help='')
    parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')

    parser.add_argument('--thread', type=int, default=1, help='')
    parser.add_argument('--b_size', type=int, default=512, help='')
    parser.add_argument('--chunksize', type=int, default=500000, help='Number of lines to read at a time')
    parser.add_argument('--debug', action='store_true', help='')

    args = parser.parse_args(_args)
    logger = logging.getLogger(__name__)
    if not os.path.exists(args.input):
        sys.exit("ERROR: cannot find input file.")
    if not os.path.exists(args.model):
        sys.exit("ERROR: cannot find model file.")
    if not os.path.exists(os.path.dirname(args.output_pref)):
        os.makedirs(os.path.dirname(args.output_pref))

    ### Input
    with gzip.open(args.input, 'rt') as rf:
        header=rf.readline().strip().split('\t')
    key = args.key
    unit_attr = args.unit_attr
    col_rename = {args.unit_id:'unit', args.feature_id:'gene'}
    header = [col_rename[x] if x in col_rename else x for x in header]
    usecol = [key, 'unit', 'gene'] + unit_attr
    for x in usecol:
        if x not in header:
            sys.exit(f"ERROR: {x} is not in the input file")
    adt = {key:int, 'unit':str, 'gene':str}
    adt.update({x:str for x in unit_attr})
    reader = pd.read_csv(gzip.open(args.input, 'rt'), sep='\t',\
                        chunksize=args.chunksize, skiprows=1, names=header,
                        usecols=usecol, dtype=adt)
    if args.debug:
        print("Input file reader")
        peek = pd.read_csv(gzip.open(args.input, 'rt'), sep='\t',\
                        chunksize=2, skiprows=1, names=header,
                        usecols=usecol, dtype=adt)
        print(next(peek))

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

    factor_header = [str(x) for x in range(K)]
    ft_dict = {x:i for i,x in enumerate( feature_kept ) }
    if os.path.exists(args.feature_id_map):
        feature_name_map = {}
        with open(args.feature_id_map, 'r') as rf:
            for line in rf:
                line = line.strip().split('\t')
                feature_name_map[line[1]] = line[0]
        ft_dict = {feature_name_map[x]:i for x,i in ft_dict.items() if x in feature_name_map}
        logger.info(f"{len(ft_dict)} features from input model exist in the input data")
    logger.info(f"Model loaded with {M} features and {K} factors")



    # Unit reader
    batch_obj =  UnitLoader(reader, ft_dict, key, min_ct_per_unit=args.min_ct_per_unit, unit_attr=unit_attr)

    # Transform
    post_count = np.zeros((K, M))
    n_unit = 0
    n_batch= 0
    oheader = [args.unit_id, key] + unit_attr + ["topK", "topP"] + factor_header
    out_f = args.output_pref + ".fit_result.tsv.gz"
    with gzip.open(out_f, 'wt') as wf:
        wf.write('\t'.join(oheader) + '\n')
    t0 = time.time()
    while batch_obj.update_batch(args.b_size):
        if args.debug:
            print(type(batch_obj.mtx))
            print(batch_obj.mtx.shape)
        theta = model.transform(batch_obj.mtx)
        post_count += np.array(theta.T @ batch_obj.mtx)
        n_batch += 1
        n_unit  += theta.shape[0]
        t1 = time.time() - t0
        if args.debug:
            print(batch_obj.brc[:2])
        batch_obj.brc['topK'] = np.argmax(theta, axis = 1)
        batch_obj.brc['topP'] = theta.max(axis = 1)
        batch_obj.brc = pd.concat([batch_obj.brc, pd.DataFrame(theta, columns=factor_header)], axis=1)
        batch_obj.brc.rename(columns = {'unit':args.unit_id}, inplace=True)
        batch_obj.brc[oheader].to_csv(out_f, sep='\t', index=False, header=False, float_format='%.3e', mode='a', compression={"method":"gzip"})
        logger.info(f"Transformed {n_batch} batches with total {n_unit} units, {t1/60:2f}min")
        if args.debug:
            break

    if args.debug:
        sys.exit()

    out_f = args.output_pref + ".posterior.count.tsv.gz"
    pd.concat([pd.DataFrame({'gene': feature_kept}),\
            pd.DataFrame(post_count.T, dtype='float64',\
                            columns = [str(k) for k in range(K)])],\
                axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})
