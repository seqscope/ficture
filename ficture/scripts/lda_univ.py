import sys, os, copy, gzip, logging
import pickle, argparse
import numpy as np
import pandas as pd

from scipy.sparse import *
import sklearn.neighbors
import sklearn.preprocessing
from sklearn.decomposition import LatentDirichletAllocation as LDA

from ficture.loaders.unit_loader import UnitLoader
from ficture.utils.utilt import init_latent_vars

def lda(_args):

    parser = argparse.ArgumentParser(prog = "lda")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', '--output_pref', type=str, help='')
    parser.add_argument('--feature', type=str, help='')
    parser.add_argument('--unit_label', default = 'random_index', type=str, help='Which column to use as unit identifier')
    parser.add_argument('--unit_attr', type=str, nargs='+', default=[], help='')
    parser.add_argument('--feature_label', default = "gene", type=str, help='Which column to use as feature identifier')
    parser.add_argument('--key', default = 'count', type=str, help='')
    parser.add_argument('--train_on', default = '', type=str, help='')
    parser.add_argument('--log', default = '', type=str, help='files to write log to')
    parser.add_argument('--shift_log_transform', action='store_true')
    parser.add_argument('--fix_scaling', type = float, default = -1,)

    parser.add_argument('--nFactor', type=int, default=10, help='')
    parser.add_argument('--minibatch_size', type=int, default=512, help='')
    parser.add_argument('--min_ct_per_feature', type=int, default=1, help='')
    parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
    parser.add_argument('--thread', type=int, default=1, help='')
    parser.add_argument('--epoch', type=int, default=1, help='How many times to loop through the full data')
    parser.add_argument('--epoch_id_length', type=int, default=-1, help='')
    parser.add_argument('--prior', type=str, default='', help="Dirichlet parameters for the global parameter beta (factor x gene)")
    parser.add_argument('--rescale_prior', type = float, default = -1,)
    parser.add_argument('--alpha', type=float, default=1, help='')
    parser.add_argument('--tau', type=int, default=9, help='')
    parser.add_argument('--kappa', type=float, default=0.7, help='')
    parser.add_argument('--N', type=float, default=1e4, help='')
    parser.add_argument('--seed', type=int, default=-1, help='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--overwrite', action='store_true')

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

    seed = args.seed
    if seed <= 0:
        seed = int(datetime.now().timestamp() )

    unit_attr = [x.lower() for x in args.unit_attr]
    key = args.key.lower()
    train_on = args.train_on.lower()
    unit_key = args.unit_label.lower()
    gene_key = args.feature_label.lower()
    if train_on == '':
        train_on = key
    adt = {unit_key:str, gene_key:str, key:int, train_on:int}
    adt.update({x:str for x in unit_attr})
    print(unit_attr)

    ### Basic parameterse
    b_size = args.minibatch_size
    K = args.nFactor

    ### Input
    # Required columns: unit ID, gene, key
    required_header = [unit_key,gene_key,train_on]
    if not os.path.exists(args.input):
        sys.exit("ERROR: cannot find input file.")
    with gzip.open(args.input, 'rt') as rf:
        header = rf.readline().strip().split('\t')
    header = [x.lower() for x in header]
    for x in required_header:
        if x not in header:
            sys.exit("Input file must have at least 3 columns: unit label, feature label, count, matching the customized column names (case insensitive) --unit_label, --feature_label, and --key/--train_on")

    # To be safe
    model_f = args.output+".model.p"
    if os.path.exists(model_f) and not args.overwrite:
        sys.exit("Model file exists, use --overwrite to allow overwriting")

    # Set up model
    fix_scaling = args.fix_scaling
    factor_header = [str(x) for x in range(K)]
    lda = LDA(n_components=K, learning_method='online', batch_size=b_size, total_samples = args.N, learning_offset = args.tau, learning_decay = args.kappa, doc_topic_prior = args.alpha, n_jobs = args.thread, random_state=seed, verbose = args.verbose)
    lda.shift_log_transform_scale = fix_scaling

    # Genes to use
    with gzip.open(args.feature, 'rt') as rf:
        fheader = rf.readline().strip().split('\t')
    fheader = [x.lower() for x in fheader]
    feature=pd.read_csv(args.feature, sep='\t', skiprows=1, names=fheader, dtype={gene_key:str, key:int})
    feature = feature[feature[key] >= args.min_ct_per_feature]
    feature.sort_values(by=key,ascending=False,inplace=True)
    feature.drop_duplicates(subset=gene_key,keep='first',inplace=True)
    feature_kept = list(feature[gene_key].values)

    # If use prior
    if os.path.isfile(args.prior):
        prior = pd.read_csv(args.prior, sep='\t', header=0)
        if "gene" not in prior.columns:
            sys.exit("ERROR: prior file must have a column named 'gene'")
        if prior.shape[1] != K + 1:
            logging.warn(f"Number of factors in --prior file does not match --nFactor ({K}), will use all factors in the prior file")
        K = prior.shape[1] - 1
        prior = prior[prior.gene.isin(feature_kept)]
        feature_kept = list(prior.gene.values)
        prior.drop(columns=['gene'], inplace=True)
        factor_header = list(prior.columns)
        M = len(feature_kept)
        prior = np.array(prior).T # K x M
        if args.rescale_prior > 0:
            w = prior.sum(axis=1)
            target_w = w * (args.N * args.rescale_prior / w.sum())
            prior = normalize(prior, norm='l1', axis=1) * target_w.reshape((-1, 1))
            print("Scaled prior")
            print(prior.sum(axis = 1).round(2))
        init_latent_vars(lda, n_features = M, gamma = prior)
        mt = prior.sum(axis =1)
        mt = " ".join([f"{x:.2e}" for x in mt])
        logging.info(f"Read prior for global parameters. Prior magnitude: {mt}")

    ft_dict = {x:i for i,x in enumerate( feature_kept ) }
    M = len(ft_dict)
    logging.info(f"Start fitting model ... {M} genes will be used")

    epoch = 0
    n_unit = 0
    chunksize=100000 if args.debug else 2000000
    while epoch < args.epoch:
        reader = pd.read_csv(gzip.open(args.input, 'rt'), \
                sep='\t',chunksize=chunksize, skiprows=1, names=header, \
                usecols=[unit_key,gene_key,train_on], dtype=adt)
        batch_obj =  UnitLoader(reader, ft_dict, train_on, \
            batch_id_prefix=args.epoch_id_length, \
            min_ct_per_unit=args.min_ct_per_unit,
            unit_id=unit_key,unit_attr=[])
        if args.shift_log_transform and n_unit == 0 and fix_scaling < 0:
            N = batch_obj.read_one_epoch();
            if N == 0:
                sys.exit("ERROR: no unit read from input file")
            unit_sum = batch_obj.mtx.sum(axis=1)
            qt_sum = np.quantile(unit_sum, q = .98)
            fix_scaling = np.log(1. / qt_sum + 1)
            logging.info(f"Shift log transform scaling: {fix_scaling:.3e} based on .98 quantile of unit sum {qt_sum}")
            batch_obj.mtx = normalize(batch_obj.mtx, norm='l1', axis=1)
            batch_obj.mtx.data = np.log(batch_obj.mtx.data + 1) / fix_scaling
            _ = lda.partial_fit(batch_obj.mtx)
            n_unit += N
            if args.verbose or args.debug:
                logl = lda.score(batch_obj.mtx) / N
                logging.info(f"Epoch {epoch}, finished {n_unit} units. batch logl: {logl:.4f}")
            if args.epoch == 1:
                break
        while batch_obj.update_batch(b_size):
            N = batch_obj.mtx.shape[0]
            if args.verbose or args.debug:
                x1 = np.median(batch_obj.brc[train_on].values)
                x2 = np.mean(batch_obj.brc[train_on].values)
                logging.info(f"Made DGE {N}, median/mean count: {x1:.1f}/{x2:.1f}")
            if args.shift_log_transform:
                batch_obj.mtx = normalize(batch_obj.mtx, norm='l1', axis=1)
                batch_obj.mtx.data = np.log(batch_obj.mtx.data + 1) / fix_scaling
            _ = lda.partial_fit(batch_obj.mtx)
            n_unit += N
            if args.verbose or args.debug:
                logl = lda.score(batch_obj.mtx) / N
                logging.info(f"Epoch {epoch}, finished {n_unit} units. batch logl: {logl:.4f}")
            if len(batch_obj.batch_id_list) > args.epoch:
                break
        if args.epoch_id_length > 0:
            epoch += len(batch_obj.batch_id_list)
        else:
            epoch += 1

    lda.feature_names_in_ = feature_kept
    # Relabel factors based on (approximate) descending abundance
    weight = lda.components_.sum(axis=1)
    ordered_k = np.argsort(weight)[::-1]
    lda.components_ = lda.components_[ordered_k,:]
    lda.exp_dirichlet_component_ = lda.exp_dirichlet_component_[ordered_k,:]
    # Store model
    out_f = args.output + ".model.p"
    pickle.dump( lda, open( out_f, "wb" ) )
    post_mtx = lda.components_.T
    out_f = args.output + ".model_matrix.tsv.gz"
    pd.concat([pd.DataFrame({gene_key: feature_kept}),\
                pd.DataFrame(post_mtx,\
                columns = [str(k) for k in range(K)], dtype='float64')],\
                axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.4e', compression={"method":"gzip"})

    ### Rerun all units once and store results
    oheader = ["unit",key,"x","y","topK","topP"]+factor_header
    dtp = {'topK':int,key:int,"unit":str}
    dtp.update({x:float for x in ['topP']+factor_header})
    res_f = args.output+".fit_result.tsv.gz"
    nbatch = 0
    logging.info(f"Result file {res_f}")

    ucol = [unit_key,gene_key,key] + unit_attr
    if key != train_on:
        ucol += [train_on]
    reader = pd.read_csv(gzip.open(args.input, 'rt'), \
            sep='\t',chunksize=chunksize, skiprows=1, names=header, \
            usecols=ucol, dtype=adt)
    batch_obj =  UnitLoader(reader, ft_dict, key, \
        batch_id_prefix=args.epoch_id_length, \
        min_ct_per_unit=args.min_ct_per_unit, \
        unit_id=unit_key, unit_attr=unit_attr, train_key=train_on)
    post_count = np.zeros((K, M))
    while batch_obj.update_batch(b_size):
        N = batch_obj.mtx.shape[0]
        if args.shift_log_transform:
            mtx = normalize(batch_obj.mtx, norm='l1', axis=1)
            mtx.data = np.log(mtx.data + 1) / fix_scaling
        else:
            mtx = batch_obj.mtx
        theta = lda.transform(mtx)
        if key != train_on:
            post_count += np.array(theta.T @ batch_obj.test_mtx)
        else:
            post_count += np.array(theta.T @ batch_obj.mtx)
        brc = pd.concat((batch_obj.brc.reset_index(), pd.DataFrame(theta, columns = factor_header)), axis = 1)
        brc['topK'] = np.argmax(theta, axis = 1).astype(int)
        brc['topP'] = np.max(theta, axis = 1)
        brc = brc.astype(dtp)
        logging.info(f"{nbatch}-th batch with {brc.shape[0]} units")
        mod = 'w' if nbatch == 0 else 'a'
        hdr = True if nbatch == 0 else False
        brc[oheader].to_csv(res_f, sep='\t', mode=mod, float_format="%.4e", index=False, header=hdr, compression={"method":"gzip"})
        nbatch += 1
        if args.epoch_id_length > 0 and len(batch_obj.batch_id_list) > 1:
            break

    logging.info(f"Finished ({nbatch})")

    out_f = args.output+".posterior.count.tsv.gz"
    pd.concat([pd.DataFrame({gene_key: feature_kept}),\
            pd.DataFrame(post_count.T, dtype='float64',\
                            columns = [str(k) for k in range(K)])],\
            axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})

if __name__ == "__main__":
    lda(sys.argv[1:])
