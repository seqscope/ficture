import sys, os, copy, gc, re, gzip, pickle, argparse, logging, warnings
import numpy as np
import pandas as pd
import matplotlib.colors
from jinja2 import Environment, FileSystemLoader
from importlib import resources as impresources

import ficture.scripts
from ficture.utils.visualize_factors import visual_hc, image_to_base64

def factor_report(_args):

    parser = argparse.ArgumentParser(prog="factor_report")
    parser.add_argument('--path', type=str, help='')
    parser.add_argument('--pref', type=str, help='')
    parser.add_argument('--model_id', type=str, default='', help='')
    parser.add_argument('--color_table', type=str, default='', help='')
    parser.add_argument('--n_top_gene', type=int, default=20, help='')
    parser.add_argument('--min_top_gene', type=int, default=10, help='')
    parser.add_argument('--max_pval', type=float, default=0.001, help='')
    parser.add_argument('--min_fc', type=float, default=1.5, help='')
    parser.add_argument('--output_pref', type=str, default='', help='')
    parser.add_argument('--annotation', type=str, default = '', help='')

    parser.add_argument('--hc_tree', action='store_true')
    parser.add_argument('--n_top_gene_on_tree', type=int, default=10, help='')
    parser.add_argument('--tree_figure', type=str, default='', help='')
    parser.add_argument('--cprob_cut', type=str, default='.99', help='Only visualize top factors with cumulative probability > cprob_cut')
    parser.add_argument('--model', type=str, default='', help='')
    parser.add_argument('--anchor', type=str, default='', help='')
    parser.add_argument('--circle_if', type=int, default=24, help='')
    parser.add_argument('--remake_tree', action='store_true')
    parser.add_argument('--circle', action='store_true')
    parser.add_argument('--vertical', action='store_true')
    args = parser.parse_args(_args)

    if len(_args) == 0:
        parser.print_help()
        return

    path=args.path
    pref=args.pref
    ntop = args.n_top_gene
    mtop = args.min_top_gene
    pval_max = args.max_pval
    fc_min = args.min_fc
    ejs = os.path.join(impresources.files(ficture.scripts), "factor_report.template.html")
    if not os.path.isfile(ejs):
        sys.exit(f"Template file {ejs} not found")

    if args.remake_tree or args.circle or args.vertical:
        args.hc_tree = True

    model_id = args.model_id
    if model_id == '':
        model_id = args.pref

    output_pref = args.output_pref
    if output_pref == '':
        output_pref = path+"/"+pref

    # Color code
    color_f = args.color_table
    if not os.path.isfile(color_f):
        color_f = path+"/figure/"+model_id+".rgb.tsv"
    if not os.path.isfile(color_f):
        sys.exit(f"Cannot find color table")
    color_table = pd.read_csv(color_f, sep='\t')
    K = color_table.shape[0]
    logging.info(f"Read color table from {color_f}")

    factor_header = np.arange(K).astype(str)
    factor_name = {}
    if os.path.isfile(args.annotation):
        with open(args.annotation) as f:
            for line in f:
                x = line.strip().split('\t')
                factor_name[x[0]] = x[1]
                factor_header[int(x[0])] = x[1]

    print(factor_header)
    color_table['RGB'] = [','.join(x) for x in np.clip((color_table.loc[:, ['R','G','B']].values * 255).astype(int), 0, 255).astype(str) ]
    color_table['HEX'] = [ matplotlib.colors.to_hex(v) for v in np.array(color_table.loc[:, ['R','G','B']]) ]
    node_color = {factor_header[v['Name']]:v['HEX'] for i,v in color_table.iterrows() }

    # Posterior count
    f=path+"/"+pref+".posterior.count.tsv.gz"
    post = pd.read_csv(f, sep='\t')
    logging.info(f"Read posterior count from {f}")
    recol = {}
    for u in post.columns:
        v = re.match('^[A-Za-z]*_*(\d+)$', u.strip())
        if v:
            recol[v.group(0)] = v.group(1)
    if len(recol) == K:
        post.rename(columns=recol, inplace=True)
    for u in factor_header:
        post[u] = post[u].astype(float)
    post_umi = post.loc[:, factor_header].sum(axis = 0).astype(int).values
    post_weight = post.loc[:, factor_header].sum(axis = 0).values.astype(float)
    post_weight /= post_weight.sum()

    # DE genes
    f=path+"/DE/"+pref+".bulk_chisq.tsv"
    if not os.path.exists(f):
        f=path+"/"+pref+".bulk_chisq.tsv"
        if not os.path.exists(f):
            sys.exit(f"Cannot find DE file")
    de = pd.read_csv(f, sep='\t', dtype={'factor':str})
    logging.info(f"Read DE genes from {f}")
    # de.factor = de.factor.astype(int)
    top_gene = []
    top_gene_anno = []
    de['Rank'] = 0
    # Temporary: shorten unspliced gene names
    de.gene = de.gene.str.replace('unspl_', 'u_')
    # Top genes by Chi2
    de.sort_values(by=['factor','Chi2'],ascending=False,inplace=True)
    de["Rank"] = de.groupby(by = "factor").Chi2.rank(ascending=False, method = "min").astype(int)
    for k, kname in enumerate(factor_header):
        indx = de.factor.eq(kname)
        v = de.loc[indx, 'gene'].iloc[:args.n_top_gene_on_tree].values
        top_gene_anno.append(', '.join(v))
        v = de.loc[indx & ( (de.Rank < mtop) | \
                ((de.pval <= pval_max) & (de.FoldChange >= fc_min)) ), \
                'gene'].iloc[:ntop].values
        if len(v) == 0:
            top_gene.append([kname, '.'])
        else:
            top_gene.append([kname, ', '.join(v)])
    # Top genes by fold change
    de.sort_values(by=['factor','FoldChange'],ascending=False,inplace=True)
    de["Rank"] = de.groupby(by = "factor").FoldChange.rank(ascending=False, method = "min").astype(int)
    for k, kname in enumerate(factor_header):
        indx = de.factor.eq(kname)
        v = de.loc[indx & ( (de.Rank < mtop) | \
                ((de.pval <= pval_max) & (de.FoldChange >= fc_min)) ), \
                'gene'].iloc[:ntop].values
        if len(v) == 0:
            top_gene[k].append('.')
        else:
            top_gene[k].append(', '.join(v))
    # Top genes by absolute weight
    for k, kname in enumerate(factor_header):
        if post_umi[k] < 10:
            top_gene[k].append('.')
            continue
        v = post.gene.iloc[np.argsort(-post.loc[:, kname].values)[:ntop] ].values
        top_gene[k].append(', '.join(v))

    # Summary
    table = pd.DataFrame({'Factor':factor_header, 'RGB':color_table.RGB.values,
                        'Weight':post_weight, 'PostUMI':post_umi,
                        'TopGene_pval':[x[1] for x in top_gene],
                        'TopGene_fc':[x[2] for x in top_gene],
                        'TopGene_weight':[x[3] for x in top_gene] })
    oheader = ["Factor", "RGB", "Weight", "PostUMI", "TopGene_pval", "TopGene_fc", "TopGene_weight"]
    # Anchor genes used for initialization if applicable
    anchor_f = args.anchor
    if not os.path.exists(anchor_f):
        anchor_f = path + "/" + model_id + ".model_anchors.tsv"
    if os.path.exists(anchor_f):
        ak = pd.read_csv(anchor_f, sep='\t', names = ["Factor", "Anchors"], dtype={"Factor":str})
        table = table.merge(ak, on = "Factor", how = "left")
        oheader.insert(4, "Anchors")
        logging.info(f"Read anchor genes from {anchor_f}")

    table.sort_values(by = 'Weight', ascending = False, inplace=True)

    f = output_pref+".factor.info.tsv"
    table.loc[table.PostUMI.ge(10), oheader].to_csv(f, sep='\t', index=False, header=True, float_format="%.5f")

    f = output_pref+".factor.info.tsv"
    with open(f, 'r') as rf:
        lines = rf.readlines()
    header = lines[0].strip().split('\t')
    rows = [ list(enumerate(row.strip().split('\t') )) for row in lines[1:]]

    # Load template
    env = Environment(loader=FileSystemLoader(os.path.dirname(ejs)))
    template = env.get_template(os.path.basename(ejs))

    image_base64 = None
    tree_alt = None
    tree_caption = None

    if args.hc_tree:
        # Hierarchical clustering
        m = re.match("^[0\.]*(\d+)$", args.cprob_cut)
        if m is None:
            sys.exit(f"Invalid --cprob_cut, please use a number between 0 and 1 (e.g. 0.99)")
        cprob_label = m.group(1)
        cprob_cut  = float(args.cprob_cut)

        tree_f = args.tree_figure
        if not os.path.exists(args.tree_figure) or args.remake_tree:
            tree_f = os.path.dirname(color_f) + '/' + pref + ".coshc."+cprob_label+".tree.png"
            if not os.path.exists(tree_f) or args.remake_tree:
                model_f = args.model
                if not os.path.exists(args.model):
                    model_f = path + "/" + model_id + ".model_matrix.tsv.gz"
                if not os.path.exists(model_f):
                    print("Cannot find model file, will cluster based on posterior count")
                    model = post
                else:
                    model = pd.read_csv(model_f, sep='\t')
                # model_prob = np.array(model.iloc[:, 1:]).T + .1
                model_prob = np.array(model.loc[:, factor_header]).T + .1
                model_prob = model_prob / model_prob.sum(axis = 1).reshape((-1,1))

                circle = args.circle
                if not circle and args.circle_if > 0:
                    v = np.argsort(post_weight)[::-1]
                    w = np.cumsum(post_weight[v] )
                    k = np.arange(K)[w > cprob_cut][0]
                    if k > args.circle_if:
                        circle = True
                tree = visual_hc(model_prob, post_weight, top_gene_anno, \
                                node_color = node_color, factor_name = factor_header, circle = circle, \
                                output_f = tree_f, cprob_cut = cprob_cut)
        print(f"Tree figure path: {tree_f}")
        image_base64 = image_to_base64(tree_f)

        tree_alt = "Hierarchical clustering of factors based on pairwise coine distance"
        tree_caption = "Clustering of factors based on pairwise coine distance. Factors with high abundance jointly accounting for " + args.cprob_cut + " of observations are displayed."

    # Render the HTML file
    html_output = template.render(header=header, rows=rows, image_base64=image_base64, tree_image_alt=tree_alt, tree_image_caption=tree_caption)

    f=output_pref+".factor.info.html"
    with open(f, "w") as html_file:
        html_file.write(html_output)

    print(f)

if __name__ == "__main__":
    factor_report(sys.argv[1:])
