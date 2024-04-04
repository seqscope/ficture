#! /usr/bin/python

''' helper functions '''
import numpy as np
import pandas as pd
import copy, re, os, geojson
from scipy import sparse
from scipy.special import gammaln, psi, logsumexp, expit, logit
from sklearn.preprocessing import normalize
import scipy.stats
import scipy.optimize
import sklearn.cluster

from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.path import Path

from sklearn.utils import check_random_state
from sklearn.decomposition._online_lda_fast import (
    _dirichlet_expectation_1d, _dirichlet_expectation_2d,
)

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    assert alpha.min() > 0, "Expecting positive Dirichlet parameters"
    if (len(alpha.shape) == 1):
        return( _dirichlet_expectation_2d(alpha.reshape((-1, 1))) )
    return _dirichlet_expectation_2d(alpha)

def init_latent_vars(model, n_features, dtype=np.float64, gamma = None, ):
    """Initialize latent variables."""
    model.random_state_ = check_random_state(model.random_state)
    model.n_batch_iter_ = 1
    model.n_iter_ = 0
    if model.doc_topic_prior is None:
        model.doc_topic_prior_ = 1.0 / model.n_components
    else:
        model.doc_topic_prior_ = model.doc_topic_prior
    if model.topic_word_prior is None:
        model.topic_word_prior_ = 1.0 / model.n_components
    else:
        model.topic_word_prior_ = model.topic_word_prior

    init_gamma = 100.0
    init_var = 1.0 / init_gamma
    # In the literature, this is called `lambda`
    if gamma is None:
        model.components_ = model.random_state_.gamma(
            init_gamma, init_var, (model.n_components, n_features)
        ).astype(dtype, copy=False)
    else:
        assert gamma.shape == (model.n_components, n_features)
        model.components_ = gamma.astype(dtype, copy=False)

    # In the literature, this is `exp(E[log(beta)])`
    model.exp_dirichlet_component_ = np.exp(
        _dirichlet_expectation_2d(model.components_)
    )

def pg_mean(b,c=0):
    if np.isscalar(c) and c == 0:
        return b/4
    if not np.isscalar(c):
        if np.isscalar(b):
            b = np.ones(c.shape[0]) * b
        v = b/4
        indx = (c != 0)
        v[indx] = b[indx]/2/c[indx] * np.tanh(c[indx]/2)
        return v
    else:
        return (b/2/c * np.tanh(c/2))

def real_to_sb(mtx):
    assert len(mtx.shape) == 2, "Invalid matrix"
    n, K = mtx.shape
    phi = expit(mtx)
    s = phi[:, 0]
    for k in range(1, K-1):
        phi[:, k] = phi[:, k] * (1. - s)
        s +=  phi[:, k]
    phi[:, K - 1] = 1. - s
    phi = np.clip(phi, 1e-8, 1.-1e-8)
    phi = normalize(phi, norm='l1', axis=1)
    return phi

def sb_to_real(phi):
    assert len(phi.shape) == 2, "Invalid matrix"
    phi = np.clip(phi, 1e-8, 1.-1e-8)
    phi = normalize(phi, norm='l1', axis=1) * (1.-1e-6)
    n, K = phi.shape
    mtx = np.zeros((n, K))
    mtx[:, 0] = logit(phi[:, 0])
    s = phi[:, 0]
    for k in range(1, K):
        mtx[:, k] = logit(phi[:, k] / (1. - s) )
        s += phi[:, k]
    return mtx

def logsumexp_csr(X):
    assert sparse.issparse(X), "logsumexp_csr: invalid input"
    X = X.tocsr()
    result = np.zeros(len(X.data))
    for i in range(X.shape[0]):
        result[X.indptr[i]:X.indptr[i+1]] =\
            logsumexp(X.data[X.indptr[i]:X.indptr[i+1]])
    return result

def gen_even_slices(n, n_packs):
    start = 0
    if n_packs < 1:
        raise ValueError("gen_even_slices got n_packs=%s, must be >=1" % n_packs)
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            yield np.arange(start, end)
            start = end

def gen_even_slices_from_list(vec, n_packs):
    start = 0
    vec = np.array(vec)
    n = len(vec)
    if n_packs < 1:
        raise ValueError("gen_even_slices got n_packs=%s, must be >=1" % n_packs)
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            yield vec[start:end]
            start = end

def gen_slices_from_list(vec, bsize):
    start = 0
    vec = np.array(vec)
    n = len(vec)
    while start < n:
        end = start + bsize
        if end > n - bsize//2:
            end = n
        yield vec[start:end]
        start = end

def get_string_with_integer_suff(in_array):
    out = []
    for u in in_array:
        v = re.match('^[A-Za-z]*_*(\d+)$', u.strip())
        if v:
            out.append(v.group(0))
    return out

def get_integer_suff_from_string(in_array):
    out = []
    for u in in_array:
        v = re.match('^[A-Za-z]*_*(\d+)$', u.strip())
        if v:
            out.append(v.group(1))
    return out

def extract_polygons_from_json(gj):
    if isinstance(gj, str):
        assert os.path.exists(gj), f"Invalid file\n{gj}"
        gj = geojson.load(open(gj,'rb'))
    assert gj['type'] == 'FeatureCollection', "Invalid geojson file"
    vertices = []
    for geom in gj['features']:
        if geom['geometry']['type'] == 'Polygon':
            poly = geom['geometry']['coordinates']
            vertices.append(np.array(poly).squeeze())
        elif geom['geometry']['type'] == 'MultiPolygon':
            for poly in geom['geometry']['coordinates']:
                vertices.append(np.array(poly).squeeze())
    return vertices

def svg_parse_list(path):
    commands = {'M': (Path.MOVETO,),'L': (Path.LINETO,),
                'Q': (Path.CURVE3,)*2,'C': (Path.CURVE4,)*3,
                'Z': (Path.CLOSEPOLY,) }
    vertices = []
    codes = []
    for v in path:
        cmd = v[0].upper()
        if len(v) > 1:
            points = [float(x) for x in v[1:]]
            points = np.array(points).reshape((len(points)//2,2))
            vertices.extend(points.tolist())
            codes.extend(commands[cmd])
    return codes, vertices

def match_factors(mtx1, mtx2, c1, n, cmap, mode='beta'):
    """
    Harmonize a pair of LDA results
    """
    k1 = mtx1.shape[0]
    k2 = mtx2.shape[0]
    assert mtx1.shape[1] == mtx2.shape[1], "Invalid input matrices"

    # Try to match two sets of factors
    Q = np.zeros((k1, k2))
    if mode == 'beta':
        beta1 = mtx1 / mtx1.sum(axis = 1).reshape((-1, 1))
        beta2 = mtx2 / mtx2.sum(axis = 1).reshape((-1, 1))
        for k in range(k1):
            q,r = scipy.optimize.nnls(beta2.transpose(), beta1[k, :].transpose())
            Q[k, ] = q
            indx2 = [(-Q[:,k]).argsort() for k in range(k2)]
            indx1 = [(-Q[k,:]).argsort() for k in range(k1)]
    else:
        for k in range(k1):
            q,r = scipy.optimize.nnls(mtx2.transpose(), mtx1[k, :].transpose())
            Q[k, ] = q
            indx2 = [(-Q[:,k]).argsort() for k in range(k2)]
            indx1 = [(-Q[k,:]).argsort() for k in range(k1)]
            Q = Q * mtx2.sum(axis = 1).reshape((1, -1))
            Q = Q / (Q.sum(axis = 1).reshape((-1, 1)))

    model = sklearn.cluster.SpectralBiclustering(n_clusters=(k1,k1), method="bistochastic").fit(Q+1e-5)
    prio = (-Q.sum(axis = 0)).argsort() # sorted from most consistent to less consistent
    step = n//(k1*2)

    dup = [1] * len(c1)
    c1_ref = copy.copy(c1)
    c2 = [-1]*k2
    for i,k in enumerate(prio):
        c2[k] = c1_ref[indx2[k][0]]
        even = dup[indx2[k][0]] % 2
        odd  = dup[indx2[k][0]] + even
        c1_ref[indx2[k][0]] += int(step * (-1)**(even) * (1-1/odd))
        dup[indx2[k][0]] += 1

    indx1 = np.array(c1).argsort()
    v1 = mtx1.sum(axis = 1)
    v1 = v1/v1.sum()
    v1 = v1[indx1]

    indx2 = np.array(c2).argsort()
    v2 = mtx2.sum(axis = 1)
    v2 = v2/v2.sum()
    v2 = v2[indx2]

    st = []
    ed = []
    st_y = []
    ed_y = []
    cl = []

    st_y += [0] + list(np.cumsum(v1)[:-1])
    ed_y += list(np.cumsum(v1))
    cl += [c1[x] for x in indx1]

    offset = 0
    st += [k1+offset]*k1
    ed += [k1+offset]*k1

    st_y += [0] + list(np.cumsum(v2)[:-1])
    ed_y += list(np.cumsum(v2))
    cl += [c2[x] for x in indx2]

    st += [k2-offset]*k2
    ed += [k2-offset]*k2

    df = pd.DataFrame({'xst':st,'yst':st_y,'xed':ed,'yed':ed_y,'Color':cl})

    mid1 = []
    mid2 = []
    v = np.cumsum(v1)
    mid1+=list(v - v1*0.5)
    v = np.cumsum(v2)
    mid2+=list(v - v2*0.5)

    xst = []
    xed = []
    yst = []
    yed = []
    sl = []
    cl = []
    offset = 0.3
    # for ii,i in enumerate(indx1):
    for ii,i in enumerate(indx1):
        for jj,j in enumerate(indx2):
            if Q[i,j] > 1e-3:
                xst.append(k2-offset)
                xed.append(k1+offset)
                yst.append(mid2[jj])
                yed.append(mid1[ii])
                sl.append(Q[i, j])
                cl.append(c1[i])
    df_arrow = pd.DataFrame({'xst':xst,'yst':yst,'xed':xed,'yed':yed,'Weight':sl,'Color':cl})
    return df, df_arrow, c2

def plot_colortable(colors, title, sort_colors=True, ncols=4, dpi = 80,\
                    cell_width = 212, cell_height = 22,\
                    title_fontsize=24, text_fontsize=24,\
                    swatch_width = 48, margin = 12, topmargin = 40):
    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + margin + topmargin


    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=title_fontsize, loc="left", pad=10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=text_fontsize,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

    return fig

def scale_to_prob(W, H):
    uk = H.sum(axis = 1).reshape((-1, 1))
    beta = H / uk
    sn = W @ uk
    theta = (W * uk.T)/sn
    return theta, beta

def read_ct_from_solo_barcodes_tsv(file, key, chunksize=500000, mu_scale=-1):
    ct_idx = "gn,gt,spl,unspl,ambig".split(",").index(key)
    reader = pd.read_csv(gzip.open(f,'rb'), sep='\t', usecols=[1,5,6,7], index_col=0, names=["unit","X","Y",key], dtype={'X':int, 'Y':int}, chunksize=chunksize)
    for chunk in reader:
        if mu_scale > 0:
            chunk.X /= mu_scale
            chunk.Y /= mu_scale
        chunk[key]=chunk[key].map(lambda x : x.split(',')[ct_idx]).astype(int)
        yield chunk

def make_mtx_from_dge(file, min_ct_per_feature = 50, min_ct_per_unit = 100, feature_white_list = None, feature_list = None, unit = "random_index", key = "gn", epoch=1, epoch_id_length=2, return_df = False):
    df = pd.DataFrame()
    epoch_id_list = set()
    for chunk in pd.read_csv(file, sep='\t', usecols = [unit,'X','Y','gene',key], dtype={unit:str}, chunksize=500000):
        unit_list = chunk[unit].str[:epoch_id_length].unique()
        i = 0
        while len(epoch_id_list) < epoch and i < len(unit_list):
            epoch_id_list.add(unit_list[i])
            i += 1
        if len(epoch_id_list) >= epoch and unit_list[-1] not in epoch_id_list:
            pat = '^' + '|'.join(list(epoch_id_list))
            df = pd.concat([df, chunk[chunk[unit].str.contains(pat) & chunk[key].ge(1)] ])
            break
        df = pd.concat([df, chunk[chunk[key].ge(1)] ])
    epoch0 = df[unit].iloc[0][:epoch_id_length]
    one_pass = df[unit].str[:epoch_id_length].eq(epoch0)
    if feature_list is not None:
        feature = pd.DataFrame({"gene": feature_list})
        ct = df[one_pass].groupby(by=['gene']).agg({key:"sum"}).reset_index()
        feature = feature.merge(right = ct, on = 'gene', how = 'left')
        feature[key] = feature[key].fillna(0).astype(int)
    else:
        feature = df[one_pass].groupby(by=['gene']).agg({key:"sum"}).reset_index()
        feature_white_list = set() if feature_white_list is None else set(feature_white_list)
        feature = feature.loc[feature[key].ge(min_ct_per_feature ) | feature.gene.isin(feature_white_list), :]
    M = len(feature)
    feature.index = np.arange(M)
    ft_dict = {x:i for i,x in enumerate(feature.gene)}
    df.drop(index = df.index[~df.gene.isin(ft_dict)], inplace=True)

    brc = df.groupby(by = unit).agg({key:"sum"}).reset_index()
    brc.drop(index = brc.index[brc[key].lt(min_ct_per_unit)], inplace=True)
    N = brc.shape[0]
    brc.index = np.arange(N)
    bc_dict = {x:i for i,x in enumerate(brc[unit])}
    brc['j'] = brc[unit].map(bc_dict)
    df.drop(index = df.index[~df[unit].isin(bc_dict)], inplace=True)
    df['j'] = df[unit].map(bc_dict)
    brc["epoch"] = brc[unit].str[:epoch_id_length]
    df.drop(columns = unit, inplace=True)
    brc = brc.merge(right = df[['j','X','Y' ]].drop_duplicates(subset='j'), on = 'j', how = 'left')

    mtx = sparse.coo_array((df[key].values, (df.j.values, df.gene.map(ft_dict))), shape=(N, M)).tocsr()

    feature["Weight"] = mtx.sum(axis = 0)
    feature.Weight = feature.Weight * 1. / feature.Weight.sum()

    if return_df:
        return df, feature, brc, mtx, ft_dict, bc_dict
    else:
        return feature, brc, mtx, ft_dict, bc_dict


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
