import sys, io, os, copy, re, time, math, glob
import importlib, warnings, subprocess
import pickle, argparse
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import plotnine
from plotnine import *
from scipy.sparse import *

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexagon_fn import *
from online_slda import *
import utilt

parser = argparse.ArgumentParser()
parser.add_argument('--input_model_template', type=str, help="Template of input LDA model, where a string NFACTOR exists and will be replaced by the number of factors")
parser.add_argument('--nfactor_list', type=str, help="A comma delimited string containing the list of factors to harmonize and plot")
parser.add_argument('--out_pref', type=str, help="Output prefix")
parser.add_argument('--cmap_name', type=str, default="turbo", help="Name of Matplotlib colormap to use")
parser.add_argument('--weight_cut', type=float, default=0.2, help="Weight cutoff in visualization")

args = parser.parse_args()

K_list = sorted([int(x) for x in args.nfactor_list.split(',')])
weight_cut = args.weight_cut

i,j = 0,1
k1,k2=K_list[i],K_list[j]
k1 = int(k1*k2 / math.gcd(k1, k2))
for k2 in K_list[2:]:
    k1 = int(k1*k2 / math.gcd(k1, k2))
n = k1
if n < np.max(K_list) * 4:
    n = np.max(K_list) * 4
print(n)

cmap_name = args.cmap_name
if args.cmap_name not in plt.colormaps():
    cmap_name = "turbo"

cmap = plt.get_cmap("turbo", n)
rgb_map = {x:matplotlib.colors.rgb2hex(cmap(x)) for x in range(n)}

df = pd.DataFrame()
df_arrow = pd.DataFrame()
factor_color_code = {}
i = 0
k1=K_list[i]
f=args.input_model_template.replace("NFACTOR", str(k1) )
while not os.path.exists(f):
    print(k1)
    i += 1
    if i >= len(K_list):
        sys.exit("Cannot find model file")
    k1=K_list[i]
    f=args.input_model_template.replace("NFACTOR", str(k1) )

K_list = K_list[i:]
if len(K_list) <= 1:
    sys.exit()

step = n//k1
c1 = [x for x in range(step//2,n,step)]
print(k1, c1)
factor_color_code[k1] = {i:x for i,x in enumerate(c1)}
i = 0
while k1 < K_list[-1]:
    i += 1
    k2=K_list[i]
    f=args.input_model_template.replace("NFACTOR", str(k1) )
    if not os.path.exists(f):
        print(f"Cannot find model file for {k1}")
        continue
    m1 = pickle.load( open( f, "rb" ) )
    f=args.input_model_template.replace("NFACTOR", str(k2) )
    if not os.path.exists(f):
        print(f"Cannot find model file for {k2}")
        continue
    m2 = pickle.load( open( f, "rb" ) )
    gene_list = sorted(list(set(m1.feature_names_in_).intersection( set(m2.feature_names_in_) )))
    if len(gene_list) == 0:
        print(f"ERROR: cannot find shared genes for {k2} and {k1}")
        continue
    gd = {x:i for i,x in enumerate(m1.feature_names_in_)}
    indx_1=[gd[x] for x in gene_list]
    gd = {x:i for i,x in enumerate(m2.feature_names_in_)}
    indx_2=[gd[x] for x in gene_list]
    mtx1 = m1.components_[:, indx_1]
    mtx2 = m2.components_[:, indx_2]
    print(mtx1.shape, mtx2.shape)

    d1,d2,c1 = utilt.match_factors(mtx1, mtx2, c1, n, cmap, mode='beta')
    factor_color_code[k2] = {i:x for i,x in enumerate(c1)}
    df = pd.concat([df, d1])
    df_arrow = pd.concat([df_arrow, d2])
    k1 = k2

if df.Color.max() > n:
    cmap = plt.get_cmap("turbo", df.Color.max()+1)
    rgb_map = {x:matplotlib.colors.rgb2hex(cmap(x)) for x in range(df.Color.max()+1)}

factor_org_code = {}
for k in K_list:
    factor_org_code = factor_org_code | { '_'.join([str(x) for x in [k, u]]) : rgb_map[v] for u,v in factor_color_code[k].items() }

code_df = [[k,v] for k,v in factor_org_code.items()]
code_df = pd.DataFrame(code_df, columns = ['factor_id', 'color_hex'] )
f = args.out_pref + ".match_factors.color_code.tsv"
code_df.to_csv(f, sep='\t', index=False)


df.Color=pd.Categorical(df.Color)
df_arrow.Color=pd.Categorical(df_arrow.Color)

fig_h = np.max([4, len(K_list)])
fig_w = np.max([fig_h, fig_h * len(K_list) // 3])
plotnine.options.figure_size = (fig_w, fig_h)
with warnings.catch_warnings(record=True):
    ps = (
        ggplot(df, aes(color='Color'))
        +geom_segment(aes(x='xst', y='yst', xend='xed',yend='yed'), size = 5)
        +geom_segment(data=df_arrow[df_arrow.Weight > weight_cut],
                      mapping=aes(x='xst', y='yst', xend='xed',yend='yed', size='Weight'),
                      arrow=arrow(length=0.1, type='open', ends='last'))
        +xlab("")+ylab("")
        +guides(color=None)
        +scale_color_manual(values = rgb_map)
        +theme_bw()
    )
    f = args.out_pref + ".match_factors.png"
    ggsave(filename=f,plot=ps,device='png')
