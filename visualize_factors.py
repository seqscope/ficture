import sys, io, os, gzip, glob, copy, re, time, pickle, warnings, importlib
import numpy as np
import pandas as pd
import base64

from sklearn.preprocessing import normalize
# from sklearn.decomposition import TruncatedSVD, PCA
import scipy.cluster
import scipy.stats
import scipy.spatial

import ete3
from ete3 import Tree, TreeStyle
os.environ['QT_QPA_PLATFORM']='offscreen'

def visual_hc(model_prob, weight, top_gene, output_f=None, cprob_cut=.99):

    K = model_prob.shape[0]
    assert len(weight) == K, "model_prob.shape[0] != len(weight)"
    assert len(top_gene) == K, "len(top_gene) != K"

    model_prob = normalize(np.array(model_prob), norm='l1', axis=1)
    weight = np.array(weight)
    weight /= weight.sum()
    weight_anno = ["%.2e" % x for x in weight ]
    v = np.argsort(weight)[::-1]
    w = np.cumsum(weight[v] )
    k = np.arange(K)[w > cprob_cut][0]
    kept_factor = v[:(k+1)].astype(str)

    node_anno = {k: str(k) + ": " + v + " ("+weight_anno[k] + ")"  for k,v in enumerate(top_gene) }

    # Hierarchical clustering
    cd_cosine = scipy.spatial.distance.pdist(model_prob, metric='cosine')
    Z_cosine = scipy.cluster.hierarchy.average(cd_cosine)

    # Construct tree object from the clustering
    R, T = scipy.cluster.hierarchy.to_tree(Z_cosine, rd=True)
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

    node_list = [x for x in tr.traverse() if x.is_leaf() and x.name in kept_factor]
    subtr = tr.copy()
    subtr.prune( [x.name for x in node_list] )
    title=f"Hierarchical clustering of {len(node_list)} factors"
    if output_f is not None:
        # Visualize tree
        def layout(node):
            if node.is_leaf():
                ete3.faces.add_face_to_node(ete3.TextFace(node_anno[int(node.name)], fsize=20,tight_text=True), node, column=0)
        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.layout_fn=layout
        ts.show_branch_length = False
        ts.mode = "c"
        ts.arc_start = 0
        ts.arc_span = 360
        ts.title.add_face(ete3.TextFace(title, fsize=50),column=0)
        subtr.render(output_f, w=2560, units='mm', tree_style=ts)

    return subtr

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
