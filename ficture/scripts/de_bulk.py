### Simple differential expression tests

import sys, io, os, gzip, copy, re, time, argparse
import numpy as np
import pandas as pd
import scipy.stats
from scipy.sparse import *
from joblib.parallel import Parallel, delayed

from ficture.utils import utilt

def de_bulk(_args):

    parser = argparse.ArgumentParser(prog="de_bulk")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--feature', type=str, default='', help='')
    parser.add_argument('--feature_label', type=str, default = "gene", help='')
    parser.add_argument('--min_ct_per_feature', default=50, type=int, help='')
    parser.add_argument('--max_pval_output', default=1e-3, type=float, help='')
    parser.add_argument('--min_fold_output', default=1.5, type=float, help='')
    parser.add_argument('--min_output_per_factor', default=10, type=int, help='Even when there are no significant DE genes, output top genes for each factor')
    parser.add_argument('--thread', default=1, type=int, help='')
    parser.add_argument('--use_input_header', action = 'store_true', help='')
    args = parser.parse_args(_args)

    if len(_args) == 0:
        parser.print_help()
        return

    pcut=args.max_pval_output
    fcut=args.min_fold_output
    gene_kept = set()
    if os.path.exists(args.feature):
        feature = pd.read_csv(args.feature, sep='\t', header=0)
        gene_kept = set(feature[args.feature_label].values )

    # Read aggregated count table
    info = pd.read_csv(args.input,sep='\t',header=0)
    oheader = []
    header = []
    if args.use_input_header:
        header = [x for x in info.columns if x != args.feature_label]
        oheader = header
    else:
        for x in info.columns:
            y = re.match('^[A-Za-z]*_*(\d+)$', x)
            if y:
                header.append(y.group(1))
                oheader.append(x)
    K = len(header)
    M = info.shape[0]
    reheader = {oheader[k]:header[k] for k in range(K)}
    reheader[args.feature_label] = "gene"
    info.rename(columns = reheader, inplace=True)
    print(f"Read posterior count over {M} genes and {K} factors")

    if len(gene_kept) > 0:
        info = info.loc[info.gene.isin(gene_kept), :]
    info["gene_tot"] = info.loc[:, header].sum(axis=1).astype(int)
    info = info[info["gene_tot"] > args.min_ct_per_feature]
    info.index = info.gene.values
    total_umi = info.gene_tot.sum()
    total_k = np.array(info.loc[:, header].sum(axis = 0) )
    M = info.shape[0]

    print(f"Testing {M} genes over {K} factors")

    def chisq(k,info,total_k,total_umi):
        res = []
        if total_k <= 0:
            return res
        for name, v in info.iterrows():
            if v[k] <= 0:
                continue
            tab=np.zeros((2,2))
            tab[0,0]=v[str(k)]
            tab[0,1]=v["gene_tot"]-tab[0,0]
            tab[1,0]=total_k-tab[0,0]
            tab[1,1]=total_umi-total_k-v["gene_tot"]+tab[0,0]
            fd=tab[0,0]/total_k/tab[0,1]*(total_umi-total_k)
            if fd < 1:
                continue
            tab = np.around(tab, 0).astype(int) + 1
            chi2, p, dof, ex = scipy.stats.chi2_contingency(tab, correction=False)
            res.append([name,k,chi2,p,fd,v["gene_tot"]])
        return res

    res = []
    if args.thread > 1:
        for k, kname in enumerate(header):
            idx_slices = [idx for idx in utilt.gen_even_slices(M, args.thread)]
            with Parallel(n_jobs=args.thread, verbose=0) as parallel:
                result = parallel(delayed(chisq)(kname, \
                            info.iloc[idx, :].loc[:, [kname, 'gene_tot']],\
                            total_k[k], total_umi) for idx in idx_slices)
            res += [item for sublist in result for item in sublist]
    else:
        for name, v in info.iterrows():
            for k, kname in enumerate(header):
                if total_k[k] <= 0 or v[str(k)] <= 0:
                    continue
                tab=np.zeros((2,2))
                tab[0,0]=v[kname]
                tab[0,1]=v["gene_tot"]-tab[0,0]
                tab[1,0]=total_k[k]-tab[0,0]
                tab[1,1]=total_umi-total_k[k]-v["gene_tot"]+tab[0,0]
                fd=tab[0,0]/total_k[k]/tab[0,1]*(total_umi-total_k[k])
                if fd < 1:
                    continue
                tab = np.around(tab, 0).astype(int) + 1
                chi2, p, dof, ex = scipy.stats.chi2_contingency(tab, correction=False)
                res.append([name,kname,chi2,p,fd,v["gene_tot"]])

    chidf=pd.DataFrame(res,columns=['gene','factor','Chi2','pval','FoldChange','gene_total'])
    chidf["Rank"] = chidf.groupby(by = "factor")["Chi2"].rank(ascending=False)
    chidf = chidf.loc[((chidf.pval<pcut)&(chidf.FoldChange>fcut)) | (chidf.Rank < args.min_output_per_factor), :]
    chidf.sort_values(by=['factor','Chi2'],ascending=[True,False],inplace=True)
    chidf.Chi2 = chidf.Chi2.map(lambda x : "{:.1f}".format(x) )
    chidf.FoldChange = chidf.FoldChange.map(lambda x : "{:.2f}".format(x) )
    chidf.gene_total = chidf.gene_total.astype(int)
    chidf.drop(columns = 'Rank', inplace=True)

    outpath=os.path.dirname(args.output)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    chidf.to_csv(args.output,sep='\t',float_format="%.2e",index=False)

if __name__ == "__main__":
    de_bulk(sys.argv[1:])
