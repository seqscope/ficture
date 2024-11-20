import sys, io, os, gzip, glob, copy, re, time, warnings, pickle, argparse, logging
from collections import defaultdict,Counter
import numpy as np
import pandas as pd
from datetime import datetime

from scipy.sparse import *
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
from sklearn.decomposition import LatentDirichletAllocation as LDA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ficture.utils.utilt import make_mtx_from_dge, init_latent_vars
from ficture.loaders.unit_loader import UnitLoader

def init_model_from_pseudobulk(_args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--model', type=str, help='')
    parser.add_argument('--feature', type=str, default = '', help='') # if provided the output will only contains these features
    parser.add_argument('--key', type=str, default='gn', help='')
    parser.add_argument('--thread', type=int, default=1, help='')
    parser.add_argument('--seed', type=int, default=-1, help='')

    parser.add_argument('--scale_model_rel', type=float, default=0, help='Scale the total magnitude of the input pseudobulk model relative to the total count of one epoch of the input. Set to <=0 to use the input model as is.')
    parser.add_argument('--epoch', type=int, default=0, help='Epoch for model training, 0 for using the input model as is')
    parser.add_argument('--alpha', type=float, default=1, help='')
    parser.add_argument('--tau', type=int, default=9, help='')
    parser.add_argument('--kappa', type=float, default=0.7, help='')

    parser.add_argument('--unit_label', default = 'random_index', type=str, help='Which column to use as unit identifier')
    parser.add_argument('--feature_label', default = "gene", type=str, help='Which column to use as feature identifier')
    parser.add_argument('--epoch_id_length', type=int, default=2, help='')
    parser.add_argument('--min_ct_per_unit', type=int, default=50, help='')
    parser.add_argument('--min_ct_per_feature', type=int, default=50, help='')
    parser.add_argument('--reorder_factors', action='store_true', help='')
    parser.add_argument('--debug', action='store_true', help='')

    args = parser.parse_args(_args)
    if len(_args) == 0:
        parser.print_help()
        return

    logging.basicConfig(level= getattr(logging, "INFO", None))

    seed = int(args.seed)
    if seed <= 0:
        seed = int(datetime.now().timestamp()) % 2147483648
    rng = check_random_state(seed)

    key = args.key
    thread = args.thread
    unit_key = args.unit_label.lower()
    gene_key = args.feature_label.lower()
    b_size = 512
    min_ct_model_feature = 50

    # Read model
    model_mtx = pd.read_csv(args.model, sep='\t', index_col = 0)
    model_gene_sum = model_mtx.sum(axis = 1)
    model_mtx.drop(index = model_gene_sum[model_gene_sum<min_ct_model_feature].index, inplace=True )
    factor_header = list(model_mtx.columns)
    feature_model =list(model_mtx.index)
    logging.info(f"Read model matrix {model_mtx.shape}")

    # Read input
    feature_list = None
    if os.path.isfile(args.feature):
        feature_list = pd.read_csv(args.feature, sep='\t', dtype={gene_key:str})[gene_key].tolist()
        feature_list = list(set(feature_list))
    feature, brc, mtx_org, ft_dict, bc_dict = make_mtx_from_dge(args.input,\
        min_ct_per_unit = args.min_ct_per_unit, \
        min_ct_per_feature = args.min_ct_per_feature, \
        feature_list = feature_list, \
        unit = args.unit_label, key = key)
    N, M0 = mtx_org.shape
    logging.info(f"Read data with {N} units, {M0} features")

    # Intersection of features
    feature_model_idx = [ft_dict[x] for x in set(feature_model) & ft_dict.keys()]
    feature_unused_idx = [ft_dict[x] for x in ft_dict.keys() - set(feature_model)]
    if len(feature_model_idx) <= 2:
        sys.exit("ERROR: too few features in common between the model and the input data")
    feature_model = feature.loc[feature_model_idx, "gene"].values
    model_mtx = model_mtx.loc[feature_model, :].values + 0.1
    M, K = model_mtx.shape
    N = mtx_org.shape[0]
    logging.info(f"Use {M} overlapping features for model fitting")

    mtx_org = mtx_org.tocsc()
    M1 = len(feature_unused_idx)
    mtx_unused = mtx_org[:, feature_unused_idx].tocsr()
    mtx_org  = mtx_org[:, feature_model_idx].tocsr()

    if args.scale_model_rel > 0:
        raw_sum = mtx_org.sum()
        model_mtx = model_mtx * (args.scale_model_rel * raw_sum / model_mtx.sum())
    model = LDA(n_components=K, learning_method='online', batch_size=b_size, total_samples = N, learning_offset = args.tau, learning_decay = args.kappa, doc_topic_prior = args.alpha, n_jobs = args.thread, verbose = 0, random_state=seed )
    init_latent_vars(model, n_features = M, gamma = model_mtx.T)

    # Update model
    e = 0
    while e < args.epoch:
        idx = rng.permutation(N)
        _ = model.partial_fit(mtx_org[idx,:])
        e += 1

    # Fit all units
    oheader = ["unit",key,"x","y","topK","topP"]+factor_header
    dtp = {'topK':int,key:int,"unit":str}
    dtp.update({x:float for x in ['topP']+factor_header})
    theta = model.transform(mtx_org)
    post_count = np.array(theta.T @ mtx_org)
    feature_output = feature_model
    if M1 > 0:
        post_count_unused = np.array(theta.T @ mtx_unused)
        post_count = np.concatenate((post_count, post_count_unused), axis = 1)
        feature_output = feature.loc[feature_model_idx + feature_unused_idx, "gene"].values

    # Relabel factors
    if args.reorder_factors:
        weight = post_count.sum(axis=1)
        ordered_k = np.argsort(weight)[::-1]
        model.components_ = model.components_[ordered_k,:]
        model.exp_dirichlet_component_ = model.exp_dirichlet_component_[ordered_k,:]
        theta = theta[:, ordered_k]
        post_count = post_count[ordered_k,:]
        factor_header = [factor_header[x] for x in ordered_k]

    if args.epoch > 0:
        out_f = args.output + ".model_matrix.tsv.gz"
        pd.concat([pd.DataFrame({gene_key: feature_model}),\
            pd.DataFrame(model.components_.T, columns = factor_header, dtype='float64')], axis = 1\
        ).to_csv(out_f, sep='\t', index=False, float_format='%.4e', compression={"method":"gzip"})
        model.feature_names_in_ = feature_model
        pickle.dump(model, open(args.output + ".model.p", 'wb'))

    # Unit assignment
    brc.rename(columns = {'j':'unit', 'X':'x', 'Y':'y'}, inplace = True)
    brc['topK'] = np.argmax(theta, axis = 1).astype(int)
    brc['topP'] = np.max(theta, axis = 1)
    brc = pd.concat((brc, pd.DataFrame(theta, columns = factor_header )), axis = 1)
    brc = brc.astype(dtp)
    res_f = args.output+".fit_result.tsv.gz"
    brc[oheader].to_csv(res_f, sep='\t', float_format="%.4e", index=False, header=True, compression={"method":"gzip"})
    logging.info(f"Result file {res_f}")

    # Posterior count
    out_f = args.output+".posterior.count.tsv.gz"
    pd.concat([pd.DataFrame({gene_key: feature_output}),\
            pd.DataFrame(post_count.T, columns = factor_header, dtype='float64')],\
        axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})

if __name__ == "__main__":
    init_model_from_pseudobulk(sys.argv[1:])
