import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd

from jinja2 import Environment, FileSystemLoader

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='')
parser.add_argument('--pref', type=str, help='')
parser.add_argument('--color_table', type=str, default='', help='')
parser.add_argument('--n_top_gene', type=int, default=20, help='')
parser.add_argument('--min_top_gene', type=int, default=10, help='')
parser.add_argument('--max_pval', type=float, default=0.001, help='')
parser.add_argument('--min_fc', type=float, default=1.5, help='')
args = parser.parse_args()

path=args.path
pref=args.pref
ntop = args.n_top_gene
mtop = args.min_top_gene
pval_max = args.max_pval
fc_min = args.min_fc
ejs = os.path.dirname(os.path.abspath(__file__))+"/factor_report.template.html"


# Color code
if os.path.isfile(args.color_table):
    color_table = pd.read_csv(args.color_table, sep='\t')
else:
    f=path+"/figure/"+pref+".rgb.tsv"
    color_table = pd.read_csv(f, sep='\t')

# DE genes
f=path+"/DE/"+pref+".bulk_chisq.tsv"
de = pd.read_csv(f, sep='\t')

# Posterior count
f=path+"/"+pref+".posterior.count.tsv.gz"
post = pd.read_csv(f, sep='\t')
recol = {}
for u in post.columns:
    v = re.match('^[A-Za-z]*_*(\d+)$', u.strip())
    if v:
        recol[v.group(0)] = v.group(1)
post.rename(columns=recol, inplace=True)

de.factor = de.factor.astype(int)
color_table.Name = color_table.Name.astype(int)
color_table.sort_values(by = 'Name', inplace=True)
color_table['RGB'] = [','.join(x) for x in np.clip((color_table.loc[:, ['R','G','B']].values * 255).astype(int), 0, 255).astype(str) ]

K = color_table.shape[0]
factor_header = np.arange(K).astype(str)

post_umi = post.loc[:, factor_header].sum(axis = 0).astype(int).values
post_weight = post.loc[:, factor_header].sum(axis = 0).values
post_weight /= post_weight.sum()

top_gene = []
de['Rank'] = 0
# Top genes by Chi2
de.sort_values(by=['factor','Chi2'],ascending=False,inplace=True)
for k in range(K):
    de.loc[de.factor.eq(k), 'Rank'] = np.arange(de.factor.eq(k).sum())
    v = de.loc[de.factor.eq(k) & ( (de.Rank < mtop) | \
               ((de.pval <= pval_max) & (de.FoldChange >= fc_min)) ), \
               'gene'].iloc[:ntop].values
    if len(v) == 0:
        top_gene.append([k, '.'])
    else:
        top_gene.append([k, ', '.join(v)])
# Top genes by fold change
de.sort_values(by=['factor','FoldChange'],ascending=False,inplace=True)
for k in range(K):
    de.loc[de.factor.eq(k), 'Rank'] = np.arange(de.factor.eq(k).sum())
    v = de.loc[de.factor.eq(k) & ( (de.Rank < mtop) | \
               ((de.pval <= pval_max) & (de.FoldChange >= fc_min)) ), \
               'gene'].iloc[:ntop].values
    if len(v) == 0:
        top_gene[k].append('.')
    else:
        top_gene[k].append(', '.join(v))
# Top genes by basolute weight
for k in range(K):
    v = post.gene.iloc[np.argsort(post.loc[:, str(k)].values)[::-1][:ntop] ].values
    top_gene[k].append(', '.join(v))

table = pd.DataFrame({'Factor':np.arange(K), 'RGB':color_table.RGB.values,
                      'Weight':post_weight, 'PostUMI':post_umi,
                      'TopGene_pval':[x[1] for x in top_gene],
                      'TopGene_fc':[x[2] for x in top_gene],
                      'TopGene_weight':[x[3] for x in top_gene] })
table.sort_values(by = 'Weight', ascending = False, inplace=True)

f = path+"/"+pref+".factor.info.tsv"
table.to_csv(f, sep='\t', index=False, header=True, float_format="%.5f")

f = path+"/"+pref+".factor.info.tsv"
with open(f, 'r') as rf:
    lines = rf.readlines()
header = lines[0].strip().split('\t')
rows = [ list(enumerate(row.strip().split('\t') )) for row in lines[1:]]

env = Environment(loader=FileSystemLoader(os.path.dirname(ejs)))
# Load template
template = env.get_template(os.path.basename(ejs))
# Render the HTML file
html_output = template.render(header=header, rows=rows)

f=path+"/"+pref+".factor.info.html"
with open(f, "w") as html_file:
    html_file.write(html_output)
