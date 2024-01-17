import sys, io, os, gzip, glob, copy, re, time, pickle, warnings, importlib
import numpy as np
import pandas as pd
import base64

from sklearn.preprocessing import normalize
# from sklearn.decomposition import TruncatedSVD, PCA
import scipy.cluster
import scipy.stats
import scipy.spatial
import scipy.linalg

import ete3
from ete3 import Tree, TreeStyle, NodeStyle
os.environ['QT_QPA_PLATFORM']='offscreen'

def logrank(x):
    v = scipy.stats.rankdata(x)
    return - np.log( 1-(v-.5)/len(v) )

def cor_logrank(orgmtx):
    K, M = orgmtx.shape
    rankmtx = np.zeros((K, M), dtype=int)
    for k in range(K):
        rankmtx[k, :] = scipy.stats.rankdata(orgmtx[k, :])
    corlogrank = np.corrcoef(-np.log( 1-(rankmtx-.5)/M ) )
    return corlogrank

def NJ_logrank(orgmtx):
    K, M = orgmtx.shape
    rankmtx = np.zeros((K, M), dtype=int)
    for k in range(K):
        rankmtx[k, :] = scipy.stats.rankdata(orgmtx[k, :])
    corlogrank = np.corrcoef(-np.log( 1-(rankmtx-.5)/M ) )
    node = {} # node id -> [leaves, profile]
    active_node = []
    for k in range(K):
        node[k] = [[k], orgmtx[k, :]]
        active_node.append(k)
    pairwise_dict = {} # (id1, id2) -> score
    for k in range(K-1):
        for l in range(k+1, K):
            pairwise_dict[(k, l)] = corlogrank[k, l]
    Z = []
    for s in range(K-1):
        candi = [] # [(id1, id2), score]
        for i,v1 in enumerate(active_node[:-1]):
            for j in range(i+1, len(active_node)):
                v2 = active_node[j]
                if (v1, v2) not in pairwise_dict:
                    vec1 = logrank(node[v1][1])
                    vec2 = logrank(node[v2][1])
                    score = np.corrcoef(vec1, vec2)[0,1]
                    candi.append([(v1, v2), score])
                    pairwise_dict[ (v1, v2) ] = score
                else:
                    score = pairwise_dict[(v1, v2)]
                    candi.append([(v1, v2), score])
        if len(candi) == 0:
            break
        candi.sort(key = lambda x : x[1], reverse=True)
        v1, v2 = candi[0][0]
        lvs = node[v1][0]+node[v2][0]
        node[K+s] = [lvs, orgmtx[lvs, :].mean(axis = 0)]
        active_node.remove(v1)
        active_node.remove(v2)
        active_node.append(K+s)
        Z.append([ v1, v2, 1-candi[0][1], len(lvs) ])
    return Z

def visual_hc(model_prob, weight, top_gene, node_color=None, factor_name=None, circle=False, vertical=False, output_f=None, cprob_cut=.99):

    K = model_prob.shape[0]
    assert len(weight) == K, "model_prob.shape[0] != len(weight)"
    assert len(top_gene) == K, "len(top_gene) != K"
    if factor_name is None:
        factor_name = [str(x) for x in range(K)]

    model_prob = normalize(np.array(model_prob), norm='l1', axis=1)
    weight = np.array(weight)
    weight /= weight.sum()
    weight_anno = ["%.2e" % x if x < 0.1 else "%.3f" % x for x in weight]
    v = np.argsort(weight)[::-1]
    w = np.cumsum(weight[v] )
    if sum(w > cprob_cut) == 0:
        k = K - 1
    else:
        k = np.arange(K)[w > cprob_cut][0]
    kept_factor = factor_name[v[:(k+1)]]
    kept_idx = v[:(k+1)].astype(str)

    # # Hierarchical clustering
    # cd_dist = scipy.spatial.distance.pdist(model_prob, metric='cosine')
    # Z_hc = scipy.cluster.hierarchy.linkage(cd_dist, method="complete")
    # Z_hc = NJ_logrank(model_prob)

    corlogrank = cor_logrank(model_prob)
    corlogrank = np.nan_to_num(corlogrank, copy=False)
    cd_dist = 1 - .5 * (corlogrank + corlogrank.T)
    np.fill_diagonal(cd_dist, 0)
    cd_dist = scipy.spatial.distance.squareform(cd_dist)
    Z_hc = scipy.cluster.hierarchy.linkage(cd_dist, method="complete")

    # Construct tree object from the clustering
    R, T = scipy.cluster.hierarchy.to_tree(Z_hc, rd=True)
    tr = Tree()
    tr.dist=0
    tr.name='root'
    node_dict = {R.id:tr}
    stack = [R]
    while stack:
        node = stack.pop()
        c_dist = node.dist / 2
        for c in [node.left, node.right]:
            if c:
                ch = Tree()
                ch.dist = c_dist
                ch.name = str(c.id)
                _=node_dict[node.id].add_child(ch)
                node_dict[c.id] = ch
                stack.append(c)

    node_list = [x for x in tr.traverse() if x.is_leaf() and x.name in
    kept_idx]
    subtr = tr.copy()
    subtr.prune( [x.name for x in node_list] )
    for x in subtr.traverse():
        if x.is_leaf():
            x.name = factor_name[int(x.name)]

    if output_f is None:
        return subtr

    ### Visualize tree
    title=f"Hierarchical clustering of {len(node_list)} factors"

    # Node style
    istyle = NodeStyle()
    istyle["size"] = 0
    for n in subtr.traverse():
        n.set_style(istyle)
    if node_color is not None:
        for n in subtr.traverse():
            if n.is_leaf():
                nstyle = NodeStyle()
                nstyle["fgcolor"] = node_color[n.name]
                nstyle['size'] = 10
                n.set_style(nstyle)
    node_anno = {factor_name[k]: " " + factor_name[k] + " ("+weight_anno[k] + "): " + v  for k,v in enumerate(top_gene) }
    def layout(node):
        if node.is_leaf():
            ete3.faces.add_face_to_node(ete3.TextFace(node_anno[node.name]), node, column=0)

    # Tree style
    ts = TreeStyle()
    ts.layout_fn=layout
    ts.show_leaf_name = False
    ts.show_branch_length = False
    ts.show_scale = False
    if circle:
        ts.mode = "c"
        ts.arc_start = 0
        ts.arc_span = 360
    else:
        if vertical:
            ts.rotation = 90
        ts.branch_vertical_margin = 20

    ts.title.add_face(ete3.TextFace(title, fsize=25),column=0)
    subtr.render(output_f, w=2560, units='mm', tree_style=ts)

    return subtr

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
