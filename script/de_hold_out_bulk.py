import sys, os, copy, gzip, logging, pickle, argparse
import numpy as np
import pandas as pd
import scipy.stats

from scipy.sparse import *
from sklearn.decomposition import LatentDirichletAllocation

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='')
parser.add_argument('--model', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--hold_out', type=str, default = "0.5", help='')
parser.add_argument('--min_ct_per_unit', type=int, default=20, help='')
parser.add_argument('--min_ct_per_feature', default=50, type=int, help='')
parser.add_argument('--max_pval_output', default=1e-3, type=float, help='')
parser.add_argument('--min_fold_output', default=2, type=float, help='')
parser.add_argument('--debug', action="store_true")
args = parser.parse_args()
logging.basicConfig(level= getattr(logging, "INFO", None))

if not os.path.exists(args.model):
    sys.exit("ERROR: cannot find model file")
if not os.path.exists(args.input):
    sys.exit("ERROR: cannot find input file.")
hold_out = np.clip(float(args.hold_out), 0, 1)
if hold_out >= 1 or hold_out <= 0:
    sys.exit("ERROR: --hold_out has to be between 0 and 1 exclusively, and training data should have enough power")
if hold_out > .9:
    logging.info(f"WARNING: high hold out rate leaves low density data for classification")
pcut=args.max_pval_output
fcut=args.min_fold_output

lda = pickle.load( open( args.model, "rb" ) )
feature_kept = lda.feature_names_in_
lda.feature_names_in_ = None
ft_dict = {x:i for i,x in enumerate( feature_kept ) }
K, M = lda.components_.shape
header = [str(k) for k in range(K)]
bsize = 512

df = pd.read_csv(args.input+"/matrix.mtx.gz",sep=' ',skiprows=3, names=["i","j","Count"], dtype={"i":str, "j":str, "Count":int})
feature = pd.read_csv(args.input+"/features.tsv.gz",sep='\t',usecols=[1],names=["gene"])
feature["i"] = np.arange(feature.shape[0]).astype(str)
feature = feature.loc[feature.gene.isin(ft_dict), :]
brc = pd.read_csv(args.input+"/barcodes.tsv.gz",sep='_',usecols=[0,5],names=["j","offset"],dtype=str)
brc = brc.loc[brc.offset.eq("0.0"), :]

df = df.merge(right = feature, on = "i", how = "inner")
df = df.merge(right = brc, on = "j", how = "inner")
df.drop(columns = "i", inplace=True)

brc = df.groupby(by = "j", as_index=False).agg({"Count":sum})
feature = df.groupby(by = "gene", as_index=False).agg({"Count":sum})
medc = np.median(brc.Count)
logging.info(f"Median count per unit {medc:.2f}, median training count ~{medc * (1-hold_out):.2f}")

df["Test"] = np.random.binomial(n=df.Count.values, p=hold_out)
df["Train"]  = df.Count - df.Test
n1 = df.Train.sum()
n2 = df.Test.sum()
logging.info(f"Total umi {n1} + {n2}")


# Make DGE
barcode_kept = list(brc.j.values)
bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
indx_row = [ bc_dict[x] for x in df['j']]
indx_col = [ ft_dict[x] for x in df['gene']]
N = len(barcode_kept)

mtx_train = coo_array((df.Train.values,\
    (indx_row, indx_col)), shape=(N, M)).tocsr()
mtx_train.eliminate_zeros()

mtx_test = coo_array((df.Test.values,\
    (indx_row, indx_col)), shape=(N, M)).tocsr()
mtx_test.eliminate_zeros()
logging.info(f"Make DGEs {N} x {M}")

post_count = np.zeros((K, M), dtype=float)

st = 0
nbatch = 0
while st < N:
    ed = np.min([st+bsize, N])
    indx = np.arange(st, ed)
    rsum = mtx_train[indx, :].sum(axis = 1)
    indx = indx[rsum > args.min_ct_per_unit]
    if len(indx) == 0:
        logging.info(f"{nbatch}-th batch does not have units meet --min_ct_per_unit")
        nbatch += 1
        continue
    theta = lda.transform(mtx_train[indx, :])
    post_count += np.array(theta.T @ mtx_test[indx, :])
    st = ed
    nbatch += 1
    print(nbatch, len(indx))
    if args.debug and nbatch > 10:
        break

logging.info(f"Finished {nbatch} x {bsize} units")
info = pd.concat([pd.DataFrame({'gene': feature_kept}),\
                  pd.DataFrame(post_count.T, dtype='float64',\
                               columns = header)], axis = 1)
out_f = args.model.replace("model.p", "hold_out.posterior.count.tsv.gz")
info.to_csv(out_f, sep='\t', index=False, float_format="%.2f", compression={"method":"gzip"})

total_k = np.array(info.loc[:, header].sum(axis = 0) )
total_umi = total_k.sum()
info["gene_tot"] = info.loc[:, header].sum(axis = 1)
info = info.loc[info.gene_tot > args.min_ct_per_feature, :]
info.sort_values(by = "gene_tot", ascending=False, inplace=True)
info.index = range(info.shape[0])
logging.info(f"Start testing {info.shape[0]} genes")

if args.debug:
    print(total_umi)
    print(np.around(total_k, 2))
    print(np.around(total_k/total_k.sum(), 2))

res=[]
for k in range(K):
    if total_k[k] <= 0:
        continue
    for i, v in info.iterrows():
        tab=np.zeros((2,2))
        tab[0,0]=v[str(k)]
        tab[0,1]=v["gene_tot"]-tab[0,0]
        tab[1,0]=total_k[k]-tab[0,0]
        tab[1,1]=total_umi-total_k[k]-v["gene_tot"]+tab[0,0]
        fd = tab[0,0]/total_k[k]*(total_umi-total_k[k])/tab[0,1]
        tab = np.around(tab, 0).astype(int) + 1
        chi2, p, dof, ex = scipy.stats.chi2_contingency(tab, correction=False)
        res.append([v["gene"],k,chi2,p,fd,v["gene_tot"]])
        if args.debug:
            if fd > 1.5 and p < 0.05:
                print(tab, fd, p)

chidf=pd.DataFrame(res,columns=['gene','factor','Chi2','pval','FoldChange','gene_total'])
chidf=chidf.loc[(chidf.pval<pcut)&(chidf.FoldChange>fcut), :].sort_values(by=['factor','FoldChange'],ascending=[True,False])
outf = args.output + ".hold_out."+args.hold_out+".bulk_chisq.tsv"
chidf.to_csv(outf,sep='\t',float_format="%.2e",index=False)
logging.info(f"Output {chidf.shape[0]} records")
