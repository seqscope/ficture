import sys, os, re, copy, subprocess

packages = "numpy,scipy,sklearn,scikit-sparse".split(',')
for pkg in packages:
    if not pkg in sys.modules:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pkg])

import numpy as np
import scipy.sparse
from sksparse.cholmod import cholesky
import sklearn.preprocessing
from scipy.special import expit, logit

class corpus_gp:

    def __init__(self):
        self.doc_pts = 0 # Center positions of patches
        self.mtx = None  # DGE n x M
        self.n = 0  # Number of patches
        self.M = 0  # Number of features (vocabulary)
        self.K = 0
        self.mu_org_mtx = None # n x K
        self.mu_new_mtx = None # n x K
        self.sig_inv_org = [] # list of K n x n sparse matrix
        self.sig_new = []     # list of K n x n sparse matrix
        self.phi = None       # n x K
        self.phi_ik = None    # n x K
        self.phi_mk = None    # M x K
        self.ll = 0

    def init_from_matrix(self, mtx, pts, mu, sig, sig_inv = None, phi = None, features=None, barcodes=None):
        assert scipy.sparse.issparse(mtx), "Invalid mtx"
        self.mtx = mtx.tocsr()
        self.n, self.M = mtx.shape

        assert pts.shape[0] == self.n, "Invalid positions"
        self.doc_pts = pts

        assert mu.shape[0] == self.n, "Invalid mu"
        self.mu_org_mtx = mu
        self.K = mu.shape[1]
        self.mu_new_mtx = copy.copy(self.mu_org_mtx)

        assert type(sig) == list and len(sig) == self.K, "Invalid precision matrix"
        for k in range(self.K):
            assert sig[k].shape == (self.n, self.n) and scipy.sparse.issparse(sig[k]), "Invalid precision matrix"
        self.sig_new = sig
        if sig_inv is None:
            self.sig_inv_org = []
            for k in range(self.K):
                self.sig_inv_org.append(cholesky(self.sig_new[k].tocsc()).inv())
        else:
            self.sig_inv_org = sig_inv

        if phi is not None:
            assert phi.shape == (self.n, self.K), "Invalid phi"
            self.phi = sklearn.preprocessing.normalize(phi, norm='l1', axis=1)
        else:
            self.phi = expit(self.mu_org_mtx)
            s = self.phi[:, 0]
            for k in range(1, self.K-1):
                self.phi[:, k] = self.phi[:, k] * (1. - s)
                s +=  self.phi[:, k]
            self.phi[:, self.K - 1] = 1. - s
            self.phi = np.clip(self.phi, self.eps0, self.eps1)
            self.phi = sklearn.preprocessing.normalize(self.phi, norm='l1', axis=1)

        self.phi_ik = np.multiply(self.mtx.sum(axis = 1), self.phi)
        self.phi_mk = self.mtx.T @ self.phi

        if features is not None:
            self.feature = features
        if barcodes is not None:
            self.barcode = barcodes
