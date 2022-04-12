import plotnine
import matplotlib
import matplotlib.pyplot as plt
from plotnine import *

import sys, io, os, gzip, glob, copy, re, time
import argparse, importlib, warnings, pickle
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--color_map', default='', type=str, help="Specify harmonized color code")
parser.add_argument('--cmap_name', default='turbo', type=str, help="Specify harmonized color code")
parser.add_argument('--in_pref', type=str, help="Input file prefix, with the number of factors replaced by string NFACTOR")
parser.add_argument('--path', type=str, help="")
parser.add_argument('--figure_width', type=int, default=15, help="Width of the output figure per 1000um")
args = parser.parse_args()

suff = ".png"
if args.color_map != '':
    code_df = pd.read_csv(args.color_map, sep='\t', dtype='str')
    factor_org_code = {code_df.iloc[i,0]:code_df.iloc[i,1] for i in range(code_df.shape[0])}
    suff = ".harmonized.png"

cmap_name = args.cmap_name
if args.cmap_name not in plt.colormaps():
    cmap_name = "nipy_spectral"

figure_path='/'.join([args.path,'analysis/figure'])
pat=args.in_pref.replace("NFACTOR","*")

plotnine.options.dpi=80

files = glob.glob(args.path+"/analysis/"+pat+".fit_result.tsv.gz")
for f in files:
    name = os.path.basename(f).split('.fit_result.tsv.gz')[0]
    print(name)
    wd = re.split('\.|_', name)
    k=int(wd[wd.index("nFactor")+1])
    lda_base_result=pd.read_csv(f,sep='\t')
    lda_base_result['Top_assigned']= lda_base_result.Top_assigned.map(lambda x : str(k)+'_'+str(x))

    if args.color_map != '':
        clist = factor_org_code
    else:
        cmap = plt.get_cmap(cmap_name, k)
        clist = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(k)]

    y_max,y_min = lda_base_result.Hex_center_y.max(), lda_base_result.Hex_center_y.min()
    pt_size = 1000 / (y_max-y_min) * 0.3
    pt_size = np.round(pt_size,2)
    fig_size = int( (y_max - y_min) / 1000 * args.figure_width )
    print(y_max - y_min, fig_size, pt_size)
    plotnine.options.figure_size = (fig_size, fig_size)
    with warnings.catch_warnings(record=True):
        ps = (
            ggplot(lda_base_result,
                   aes(x='Hex_center_y', y='Hex_center_x',
                       color='Top_assigned',alpha='Top_Prob'))
            +geom_point(size = pt_size, shape='o')
            +guides(colour = guide_legend(override_aes = {'size':3,'shape':'o'}))
            +xlab("")+ylab("")
            +guides(alpha=None)
            +coord_fixed(ratio = 1)
            +scale_color_manual(values = factor_org_code)
            +theme_bw()
            +theme(legend_position='bottom')
        )
        fig_f = figure_path + "/" + name + suff
        print(fig_f)
        ggsave(filename=fig_f,plot=ps,device='png',limitsize=False)


files = glob.glob(args.path+"/analysis/"+pat+".refine.pixel.tsv")
for f in files:
    name = os.path.basename(f).split('.refine.pixel.tsv')[0]
    print(name)
    wd = re.split('\.|_', name)
    k=int(wd[wd.index("nFactor")+1])

    pixel_result = pd.read_csv(f,sep='\t')
    pixel_result['Top_assigned'] = pixel_result.Top_assigned.map(lambda x : str(k)+'_'+str(x))
    fig_size = int( (pixel_result.y.max() - pixel_result.y.min()) / 1000 * args.figure_width )

    plotnine.options.figure_size = (fig_size, fig_size)
    with warnings.catch_warnings(record=True):
        ps = (
            ggplot(pixel_result,
                   aes(x='y', y='x', color='Top_assigned',alpha='Top_Prob'))
            +geom_point(size = 0.1, shape='+')
            +guides(colour = guide_legend(override_aes = {'size':4,'shape':'o'}))
            +xlab("")+ylab("")
            +guides(alpha=None)
            +coord_fixed(ratio = 1)
            +scale_color_manual(values = factor_org_code)
            +theme_bw()
            +theme(legend_position='bottom')
        )
        fig_f = figure_path + "/" + name + ".refine.pixel" + suff
        print(fig_f)
        ggsave(filename=fig_f,plot=ps,device='png',limitsize=False)

files = glob.glob(args.path+"/analysis/"+pat+".refine.center.tsv")
for f in files:
    name = os.path.basename(f).split('.refine.center.tsv')[0]
    print(name)
    wd = re.split('\.|_', name)
    k=int(wd[wd.index("nFactor")+1])

    center_result = pd.read_csv(f,sep='\t')
    center_result['Top_assigned'] = center_result.Top_assigned.map(lambda x : str(k)+'_'+str(x))

    pt_size = 1000/(center_result.y.max()-center_result.y.min()) * 0.3
    pt_size = np.round(pt_size,2)
    fig_size = int( (center_result.y.max() - center_result.y.min()) / 1000 * args.figure_width )

    plotnine.options.figure_size = (fig_size, fig_size)
    with warnings.catch_warnings(record=True):
        ps = (
            ggplot(center_result[center_result.Avg_size > 10],
                   aes(x='y', y='x',
                       color='Top_assigned',alpha='Top_Prob'))
            +geom_point(size = pt_size, shape='o')
            +guides(colour = guide_legend(override_aes = {'size':4,'shape':'o'}))
            +xlab("")+ylab("")
            +guides(alpha=None)
            +coord_fixed(ratio = 1)
            +scale_color_manual(values = factor_org_code)
            +theme_bw()
            +theme(legend_position='bottom')
        )
        fig_f = figure_path + "/" + name + ".refine.center" + suff
        print(fig_f)
        ggsave(filename=fig_f,plot=ps,device='png',limitsize=False)
