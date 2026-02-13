import sys, os, re, copy, gzip, time, logging, pickle, argparse
import numpy as np
import pandas as pd
import sklearn.neighbors
import scipy.optimize
from scipy.sparse import coo_array
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='')
parser.add_argument('--query', type=str, help='')
parser.add_argument('--output', type=str, help='')
parser.add_argument('--K1', type=int, help='')
parser.add_argument('--tol', type=float, default=0.5, help="")
parser.add_argument('--query_scale', type=float, default=-1, help="")
parser.add_argument('--kcol', type=str, default='K1', help='')
args = parser.parse_args()

K1 = args.K1
cluster_list = np.arange(K1)

f = args.path + "/model.true.tsv.gz"
model = pd.read_csv(f,sep='\t',index_col=0)
label_list = list(model.columns)
label_idx = {str(x):i for i,x in enumerate(label_list)}
K0 = len(label_list)

f = args.path + "/model.rgb.tsv"
ctab = pd.read_csv(f,sep='\t',dtype=str)


f = args.path+"/pixel_label.uniq.tsv.gz"
ref = pd.read_csv(f,sep='\t',usecols=['X','Y','cell_label','cell_shape'],dtype={'cell_label':str})
ref['K'] = ref.cell_label.map(label_idx).values
ref.index = range(ref.shape[0])
ref_bt = sklearn.neighbors.BallTree(ref.loc[:, ['X','Y']].astype(float))


dty = {'X':float, 'Y':float, args.kcol:int}
nskip=0
with gzip.open(args.query, 'rt') as rf:
    for line in rf:
        if line[:2] == "##":
            nskip += 1
        # elif line[0] == "#":
        else:
            header = line.strip().split('\t')
            break
header = [x.upper() if x in ['x','y'] else x for x in header]

query = pd.read_csv(args.query,sep='\t',skiprows=nskip+1,names=header,usecols=['X','Y',args.kcol],dtype=dty)
query.rename(columns = {args.kcol:'topK'}, inplace=True)
if args.query_scale > 0:
    query.X = query.X * args.query_scale
    query.Y = query.Y * args.query_scale
if query['topK'].min() != 0: # temporary for Baysor output
    query['topK'] -= query.topK.min()

dist, indx = ref_bt.query( query.loc[:, ['X','Y']].values, k=1, return_distance=True )
dist = dist.reshape(-1)
indx = indx.reshape(-1)
kept = dist < args.tol
brc = copy.copy(query[kept])
brc['cell_label'] = ref.loc[indx[kept], 'cell_label'].values
brc['cell_shape'] = ref.loc[indx[kept], 'cell_shape'].values

confusion = brc.groupby(by=['cell_label','topK']).size().reset_index().rename(columns = {0:'Count'})
f = args.output + ".confusion.tsv"
confusion.to_csv(f,sep='\t',index=False)


confusion['K'] = confusion.cell_label.map(label_idx)
confusion['topK'] = confusion.topK.astype(int)

cost_mtx = coo_array((confusion.Count.values, (confusion.K.values, confusion.topK.values)), shape=(K0, K1)).toarray()
row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mtx, maximize=True)
label_assign = {col_ind[i]:label_list[x] for i,x in enumerate(row_ind) }

ctab_new = pd.DataFrame({'Name':np.arange(K1)})
ctab_new['cell_label'] = ctab_new.Name.map(label_assign)
ctab_new.cell_label.fillna('-', inplace=True)
ctab_new = ctab_new.merge(right=ctab.drop(columns='Name'),on='cell_label',how='left')

cmap = plt.get_cmap("turbo", K1+2)
cmtx = [list(cmap(k+1)) for k in range(K1) ]
cmtx = np.array(cmtx)[:, :3]
cmtx0 = ctab_new.loc[~ctab_new.cell_label.eq("-"), list("RGB")].astype(float).values
dvec = np.zeros(K1)
for i in range(K1):
    d = np.abs(cmtx0 - cmtx[[i], :]).sum(axis = 1)
    dvec[i] = min(d)
keep = np.argsort(-dvec)[:(K1-K0)]
ctab_new.loc[ctab_new.cell_label.eq("-"), list("RGB")] = cmtx[keep, :]

# ctab_new.fillna(0,inplace=True)
ctab_new.sort_values(by='Name', inplace=True)

f = args.output + ".matched.rgb.tsv"
ctab_new.to_csv(f,sep='\t',index=False,header=True,float_format="%.5f")




print(cluster_list)
print(label_assign)

label_assign = {cluster_list[k]:v for k,v in label_assign.items()}
brc['topK_relabel'] = brc['topK'].map(label_assign)
brc['topK_relabel'].fillna('-', inplace=True)

f = args.output + ".bad_pixel.tsv.gz"
brc.loc[brc.topK_relabel != brc.cell_label].to_csv(f,sep='\t',index=False,compression='gzip', float_format='%.2f')

f = args.output + ".bad_pixel.foreground.tsv.gz"
brc.loc[(brc.topK_relabel != brc.cell_label)&(~brc.cell_shape.eq('background'))].to_csv(f,sep='\t',index=False,compression='gzip', float_format='%.2f')
