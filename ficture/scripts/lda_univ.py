import sys, os, copy, gzip, logging
import pickle, argparse
import numpy as np
import pandas as pd

from scipy.sparse import *
import sklearn.neighbors
import sklearn.preprocessing
from sklearn.decomposition import LatentDirichletAllocation as LDA

def lda(_args):

    parser = argparse.ArgumentParser(prog = "lda")

    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--unit_label', default = 'random_index', type=str, help='Which column to use as unit identifier')
    parser.add_argument('--unit_attr', type=str, nargs='+', default=[], help='')
    parser.add_argument('--feature', type=str, default='', help='')
    parser.add_argument('--feature_label', default = "gene", type=str, help='Which column to use as feature identifier')
    parser.add_argument('--key', default = 'count', type=str, help='')
    parser.add_argument('--train_on', default = '', type=str, help='')

    parser.add_argument('--nFactor', type=int, default=10, help='')
    parser.add_argument('--minibatch_size', type=int, default=512, help='')
    parser.add_argument('--min_ct_per_feature', type=int, default=1, help='')
    parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
    parser.add_argument('--thread', type=int, default=1, help='')
    parser.add_argument('--epoch', type=int, default=1, help='How many times to loop through the full data')
    parser.add_argument('--epoch_id_length', type=int, default=-1, help='')
    parser.add_argument('--use_model', type=str, default='', help="Use provided model to transform input data")
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args(_args)
    logger = logging.getLogger(__name__)

    if args.use_model != '' and not os.path.exists(args.use_model):
        sys.exit("Invalid model file")

    unit_attr = [x.lower() for x in args.unit_attr]
    key = args.key.lower()
    unit_key = args.unit_label.lower()
    gene_key = args.feature_label.lower()
    train_on = args.train_on.lower()
    if train_on == '':
        train_on = key
    adt = {unit_key:str, gene_key:str, key:int}
    adt.update({x:str for x in unit_attr})

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

    use_existing_model = False
    model_f = args.output+".model.p"
    if os.path.exists(args.use_model):
        model_f = args.use_model
    if not args.overwrite and os.path.exists(model_f):
        lda = pickle.load( open( model_f, "rb" ) )
        feature_kept = lda.feature_names_in_
        lda.feature_names_in_ = None
        ft_dict = {x:i for i,x in enumerate( feature_kept ) }
        K, M = lda.components_.shape
        use_existing_model = True
        logger.warning(f"Read existing model from\n{model_f}\n use --overwrite to allow the model files to be overwritten\n{M} genes will be used")

    factor_header = [str(x) for x in range(K)]
    if not use_existing_model:
        if not os.path.exists(args.feature):
            sys.exit("Unable to read feature list")
        ### Use only the provided list of features
        with gzip.open(args.feature, 'rt') as rf:
            fheader = rf.readline().strip().split('\t')
        fheader = [x.lower() for x in fheader]
        feature=pd.read_csv(args.feature, sep='\t', skiprows=1, names=fheader, dtype={gene_key:str, key:int})
        feature = feature[feature[key] >= args.min_ct_per_feature]
        feature.sort_values(by=key,ascending=False,inplace=True)
        feature.drop_duplicates(subset=gene_key,keep='first',inplace=True)
        feature_kept = list(feature[gene_key].values)
        ft_dict = {x:i for i,x in enumerate( feature_kept ) }
        M = len(feature_kept)

        logger.info(f"Start fitting model ... model will be stored in\n{model_f}\n{M} genes will be used")
        lda = LDA(n_components=K, learning_method='online', batch_size=b_size, n_jobs = args.thread, verbose = 0)
        feature_mf = np.array(feature[key].values).astype(float)
        feature_mf/= feature_mf.sum()
        epoch = 0
        n_unit = 0
        epoch_id = set()
        while epoch < args.epoch:
            df = pd.DataFrame()
            for chunk in pd.read_csv(gzip.open(args.input, 'rt'), \
                    sep='\t',chunksize=1000000, skiprows=1, names=header, \
                    usecols=[unit_key,gene_key,train_on], dtype=adt):
                chunk = chunk[chunk[gene_key].isin(feature_kept)]
                if args.epoch_id_length > 0:
                    v = set(chunk[unit_key].map(lambda x: x[:args.epoch_id_length]).unique())
                    epoch_id.update(v)
                    epoch = len(epoch_id) - 1
                chunk.rename(columns = {train_on:key}, inplace=True)
                if chunk.shape[0] == 0:
                    continue
                last_indx = chunk[unit_key].iloc[-1]
                df = pd.concat([df, chunk[~chunk[unit_key].eq(last_indx)]])
                if len(df[unit_key].unique()) < b_size * 1.5: # Left to next chunk
                    df = pd.concat((df, chunk[chunk[unit_key].eq(last_indx)]))
                    continue
                # Total mulecule count per unit
                brc = df.groupby(by = [unit_key]).agg({key: sum}).reset_index()
                brc = brc[brc[key] > args.min_ct_per_unit]
                brc.index = range(brc.shape[0])
                df = df[df[unit_key].isin(brc[unit_key].values)]
                # Make DGE
                barcode_kept = list(brc[unit_key].values)
                bc_dict  = {x:i for i,x in enumerate( barcode_kept ) }
                indx_row = [ bc_dict[x] for x in df[unit_key]]
                indx_col = [ ft_dict[x] for x in df[gene_key]]
                N = len(barcode_kept)
                mtx = coo_matrix((df[key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
                x1 = np.median(brc[key].values)
                x2 = np.mean(brc[key].values)
                logger.info(f"Made DGE {mtx.shape}, median/mean count: {x1:.1f}/{x2:.1f}")
                _ = lda.partial_fit(mtx)

                df = copy.copy(chunk[chunk[unit_key].eq(last_indx)] )
                logl = lda.score(mtx) / mtx.shape[0]
                n_unit += N
                logger.info(f"Epoch {epoch}, finished {n_unit} units. batch logl: {logl:.4f}")
                if epoch >= args.epoch:
                    break

            # Leftover
            if len(df[unit_key].unique()) > b_size:
                brc = df.groupby(by = [unit_key]).agg({key: sum}).reset_index()
                brc = brc[brc[key] > args.min_ct_per_unit]
                brc.index = range(brc.shape[0])
                df = df[df[unit_key].isin(brc[unit_key].values)]
                barcode_kept = list(brc[unit_key].values)
                bc_dict  = {x:i for i,x in enumerate( barcode_kept ) }
                indx_row = [ bc_dict[x] for x in df[unit_key]]
                indx_col = [ ft_dict[x] for x in df[gene_key]]
                N = len(barcode_kept)
                mtx = coo_matrix((df[key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
                x1 = np.median(brc[key].values)
                x2 = np.mean(brc[key].values)
                logger.info(f"Made DGE {mtx.shape}, median/mean count: {x1:.1f}/{x2:.1f}")
                _ = lda.partial_fit(mtx)
                logl = lda.score(mtx) / mtx.shape[0]
                logger.info(f"logl: {logl:.4f}")

            if args.epoch_id_length <= 0:
                epoch += 1

        lda.feature_names_in_ = feature_kept
        # Relabel factors based on (approximate) descending abundance
        weight = lda.components_.sum(axis=1)
        ordered_k = np.argsort(weight)[::-1]
        lda.components_ = lda.components_[ordered_k,:]
        lda.exp_dirichlet_component_ = lda.exp_dirichlet_component_[ordered_k,:]
        # Store model
        pickle.dump( lda, open( model_f, "wb" ) )
        out_f = model_f.replace("model.p", "model_matrix.tsv.gz")
        pd.concat([pd.DataFrame({gene_key: lda.feature_names_in_}),\
                    pd.DataFrame(lda.components_.T,\
                    columns = [str(k) for k in range(K)], dtype='float64')],\
                    axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.4e', compression={"method":"gzip"})


    ###
    ### Rerun all units once and store results
    ###
    dtp = {'topK':int,key:int,unit_key:str}
    dtp.update({x:float for x in ['topP']+factor_header})
    res_f = args.output+".fit_result.tsv.gz"
    nbatch = 0
    logger.info(f"Result file {res_f}")

    if train_on != key:
        post_count = np.zeros((K, M))
        epoch_id = ''
        end_of_epoch = False
        df = pd.DataFrame()
        for chunk in pd.read_csv(gzip.open(args.input, 'rt'), \
                sep='\t',chunksize=1000000, skiprows=1, names=header, \
                usecols=[unit_key,gene_key,key,train_on]+unit_attr, dtype=adt):
            chunk = chunk[chunk[gene_key].isin(feature_kept)]
            if chunk.shape[0] == 0:
                continue
            if args.epoch_id_length > 0:
                v = chunk[unit_key].map(lambda x: x[:args.epoch_id_length]).values
                if epoch_id == '':
                    epoch_id = v[0]
                if v[-1] != v[0]:
                    end_of_epoch = True
                    chunk = chunk.loc[v == epoch_id, :]
                if chunk.shape[0] == 0:
                    continue
            last_indx = chunk[unit_key].iloc[-1]
            df = pd.concat([df, chunk[~chunk[unit_key].eq(last_indx)]])
            if len(df[unit_key].unique()) < b_size * 1.5: # Left to next chunk
                df = pd.concat((df, chunk[chunk[unit_key].eq(last_indx)]))
                continue
            # Total mulecule count per unit
            brc = df.groupby(by = [unit_key]).agg({key: sum}).reset_index()
            brc = brc.merge(right=df[[unit_key]+unit_attr].drop_duplicates(subset=unit_key),on=unit_key,how='inner')
            brc = brc[brc[key] > args.min_ct_per_unit]
            brc.index = range(brc.shape[0])
            df = df[df[unit_key].isin(brc[unit_key].values)]

            # Make training DGE
            barcode_kept = list(brc[unit_key].values)
            bc_dict  = {x:i for i,x in enumerate( barcode_kept ) }
            indx_row = [ bc_dict[x] for x in df[unit_key]]
            indx_col = [ ft_dict[x] for x in df[gene_key]]
            N = len(barcode_kept)
            mtx = coo_matrix((df[train_on].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
            x1 = np.median(brc[key].values)
            x2 = np.mean(brc[key].values)
            logger.info(f"Made DGE {mtx.shape}, median/mean count: {x1:.1f}/{x2:.1f}")

            theta = lda.transform(mtx)

            # Aggregate testing counts
            test_ct = df[key].values - df[train_on].values
            if np.min(test_ct) < 0:
                sys.exit("Total count cannot be smaller than training count")
            mtx = coo_matrix((test_ct, (indx_row, indx_col)), shape=(N, M)).tocsr()
            post_count += np.array(theta.T @ mtx)

            brc = pd.concat((brc, pd.DataFrame(theta, columns = factor_header)), axis = 1)
            brc['topK'] = np.argmax(theta, axis = 1).astype(int)
            brc['topP'] = np.max(theta, axis = 1)
            brc = brc.astype(dtp)
            logger.info(f"{nbatch}-th batch with {brc.shape[0]} units")
            if nbatch == 0:
                brc.to_csv(res_f, sep='\t', mode='w', float_format="%.4e", index=False, header=True, compression={"method":"gzip"})
            else:
                brc.to_csv(res_f, sep='\t', mode='a', float_format="%.4e", index=False, header=False, compression={"method":"gzip"})
            nbatch += 1
            df = copy.copy(chunk[chunk[unit_key].eq(last_indx)] )
            if end_of_epoch:
                break

        # Leftover
        brc = df.groupby(by = [unit_key]).agg({key: sum}).reset_index()
        brc = brc.merge(right=df[[unit_key]+unit_attr].drop_duplicates(subset=unit_key),on=unit_key,how='inner')
        brc = brc[brc[key] > args.min_ct_per_unit]
        brc.index = range(brc.shape[0])
        # print(brc.shape)
        if brc.shape[0] > 0:
            df = df[df[unit_key].isin(brc[unit_key].values)]
            # Make training DGE
            barcode_kept = list(brc[unit_key].values)
            bc_dict  = {x:i for i,x in enumerate( barcode_kept ) }
            indx_row = [ bc_dict[x] for x in df[unit_key]]
            indx_col = [ ft_dict[x] for x in df[gene_key]]
            N = len(barcode_kept)
            mtx = coo_matrix((df[train_on].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
            x1 = np.median(brc[key].values)
            x2 = np.mean(brc[key].values)
            logger.info(f"Made DGE {mtx.shape}, median/mean count: {x1:.1f}/{x2:.1f}")

            theta = lda.transform(mtx)

            # Aggregate testing counts
            test_ct = df[key].values - df[train_on].values
            if np.min(test_ct) < 0:
                sys.exit("Total count cannot be smaller than training count")
            mtx = coo_matrix((test_ct, (indx_row, indx_col)), shape=(N, M)).tocsr()
            post_count += np.array(theta.T @ mtx)

            brc = pd.concat((brc, pd.DataFrame(theta, columns = factor_header)), axis = 1)
            brc['topK'] = np.argmax(theta, axis = 1).astype(int)
            brc['topP'] = np.max(theta, axis = 1)
            brc = brc.astype(dtp)
            logger.info(f"{nbatch}-th batch with {brc.shape[0]} units")
            if nbatch == 0:
                brc.to_csv(res_f, sep='\t', mode='w', float_format="%.4e", index=False, header=True, compression={"method":"gzip"})
            else:
                brc.to_csv(res_f, sep='\t', mode='a', float_format="%.4e", index=False, header=False, compression={"method":"gzip"})

        logger.info(f"Finished ({nbatch})")

        out_f = args.output+".posterior.count.tsv.gz"
        pd.concat([pd.DataFrame({gene_key: feature_kept}),\
                pd.DataFrame(post_count.T, dtype='float64',\
                                columns = [str(k) for k in range(K)])],\
                axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})

        sys.exit()





    post_count = np.zeros((K, M))
    epoch_id = ''
    df = pd.DataFrame()
    end_of_epoch = False
    for chunk in pd.read_csv(gzip.open(args.input, 'rt'), \
            sep='\t',chunksize=1000000, skiprows=1, names=header, \
            usecols=[unit_key,gene_key,key]+unit_attr, dtype=adt):
        chunk = chunk[chunk[gene_key].isin(feature_kept)]
        if chunk.shape[0] == 0:
            continue
        if args.epoch_id_length > 0:
            v = chunk[unit_key].map(lambda x: x[:args.epoch_id_length]).values
            if epoch_id == '':
                epoch_id = v[0]
            if v[-1] != v[0]:
                end_of_epoch = True
                chunk = chunk.loc[v == epoch_id, :]
            if chunk.shape[0] == 0:
                continue
        last_indx = chunk[unit_key].iloc[-1]
        df = pd.concat([df, chunk[~chunk[unit_key].eq(last_indx)]])
        if len(df[unit_key].unique()) < b_size * 1.5: # Left to next chunk
            df = pd.concat((df, chunk[chunk[unit_key].eq(last_indx)]))
            continue
        # Total mulecule count per unit
        brc = df.groupby(by = [unit_key]).agg({key: sum}).reset_index()
        brc = brc.merge(right=df[[unit_key]+unit_attr].drop_duplicates(subset=unit_key),on=unit_key,how='inner')
        brc = brc[brc[key] > args.min_ct_per_unit]
        brc.index = range(brc.shape[0])
        df = df[df[unit_key].isin(brc[unit_key].values)]

        # Make training DGE
        barcode_kept = list(brc[unit_key].values)
        bc_dict  = {x:i for i,x in enumerate( barcode_kept ) }
        indx_row = [ bc_dict[x] for x in df[unit_key]]
        indx_col = [ ft_dict[x] for x in df[gene_key]]
        N = len(barcode_kept)
        mtx = coo_matrix((df[key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
        x1 = np.median(brc[key].values)
        x2 = np.mean(brc[key].values)
        logger.info(f"Made DGE {mtx.shape}, median/mean count: {x1:.1f}/{x2:.1f}")

        theta = lda.transform(mtx)
        post_count += np.array(theta.T @ mtx)

        brc = pd.concat((brc, pd.DataFrame(theta, columns = factor_header)), axis = 1)
        brc['topK'] = np.argmax(theta, axis = 1).astype(int)
        brc['topP'] = np.max(theta, axis = 1)
        brc = brc.astype(dtp)
        logger.info(f"{nbatch}-th batch with {brc.shape[0]} units")
        if nbatch == 0:
            brc.to_csv(res_f, sep='\t', mode='w', float_format="%.4e", index=False, header=True, compression={"method":"gzip"})
        else:
            brc.to_csv(res_f, sep='\t', mode='a', float_format="%.4e", index=False, header=False, compression={"method":"gzip"})
        nbatch += 1
        df = copy.copy(chunk[chunk[unit_key].eq(last_indx)] )
        if end_of_epoch:
            break

    # Leftover
    brc = df.groupby(by = [unit_key]).agg({key: sum}).reset_index()
    brc = brc.merge(right=df[[unit_key]+unit_attr].drop_duplicates(subset=unit_key),on=unit_key,how='inner')
    brc = brc[brc[key] > args.min_ct_per_unit]
    brc.index = range(brc.shape[0])
    if brc.shape[0] > 0:
        df = df[df[unit_key].isin(brc[unit_key].values)]
        # Make DGE
        barcode_kept = list(brc[unit_key].values)
        bc_dict  = {x:i for i,x in enumerate( barcode_kept ) }
        indx_row = [ bc_dict[x] for x in df[unit_key]]
        indx_col = [ ft_dict[x] for x in df[gene_key]]
        N = len(barcode_kept)
        mtx = coo_matrix((df[key].values, (indx_row, indx_col)), shape=(N, M)).tocsr()
        x1 = np.median(brc[key].values)
        x2 = np.mean(brc[key].values)
        logger.info(f"Made DGE {mtx.shape}, median/mean count: {x1:.1f}/{x2:.1f}")

        theta = lda.transform(mtx)
        post_count += np.array(theta.T @ mtx)

        brc = pd.concat((brc, pd.DataFrame(theta, columns = factor_header)), axis = 1)
        brc['topK'] = np.argmax(theta, axis = 1).astype(int)
        brc['topP'] = np.max(theta, axis = 1)
        brc = brc.astype(dtp)
        if nbatch == 0:
            brc.to_csv(res_f, sep='\t', mode='w', float_format="%.4e", index=False, header=True, compression={"method":"gzip"})
        else:
            brc.to_csv(res_f, sep='\t', mode='a', float_format="%.4e", index=False, header=False, compression={"method":"gzip"})

    logger.info(f"Finished ({nbatch})")

    out_f = args.output+".posterior.count.tsv.gz"
    pd.concat([pd.DataFrame({gene_key: feature_kept}),\
            pd.DataFrame(post_count.T, dtype='float64',\
                            columns = [str(k) for k in range(K)])],\
            axis = 1).to_csv(out_f, sep='\t', index=False, float_format='%.2f', compression={"method":"gzip"})
