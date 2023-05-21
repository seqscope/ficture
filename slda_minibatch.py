from scipy import sparse
import sklearn.preprocessing
import numpy as np
import copy

class minibatch:

    def __init__(self):
        self.doc_pts = 0  # Positions of anchor points
        self.n = 0        # Number of anchor points
        self.N = 0        # Number of pixels
        self.M = 0        # Number of features (vocabulary)
        self.psi = None   # N x n: P(j|i)
        self.phi = None   # N x K: P(j|i)
        self.gamma = None # n x K: P(k|n)
        self.alpha = None # Prior for gamma
        self.ElogO = None # d_psi E_q[log P(Cij)]
        self.ll = 0       # log likelihood
        self.anchor_adj = None # Adjacency matrix of anchor points

    def init_from_matrix(self, mtx, doc_pts, w, psi = None, phi = None, m_gamma = None, anchor_adj = None, features = None, barcodes = None):
        assert sparse.issparse(mtx), "Invalid matrix - must be sparse"
        assert sparse.issparse(w), "Invalid w - must be sparse"

        self.mtx = mtx.tocsr()
        self.N, self.M = mtx.shape
        self.doc_pts = doc_pts
        self.n = self.doc_pts.shape[0]

        assert np.min(w.data) >= 0, "Invalid w - value must be nonnegative"
        assert w.shape == (self.N, self.n), "Inconsistent dimensions of w"
        self.ElogO = copy.copy(w)
        self.ElogO.eliminate_zeros()
        if psi is not None:
            assert sparse.issparse(psi),  "Invalid psi: must be sparse matrix"
            assert psi.shape  == (self.N, self.n), "Inconsistent dimensions of psi"
            self.psi = psi
            self.psi = (self.ElogO != 0).multiply(self.psi)
        else:
            self.psi = copy.copy(self.ElogO)
        self.ElogO.data = np.log(self.ElogO.data)

        if phi is not None:
            assert phi.shape[0] == self.N, "Inconsistent dimensions of phi"
            self.phi = sklearn.preprocessing.normalize(phi, norm='l1', axis=1)

        if m_gamma is not None:
            assert m_gamma.shape[0] == self.n, "Inconsistent dimensions of gamma"
            self.alpha = m_gamma
            self.gamma = m_gamma

        self.anchor_adj = anchor_adj
        if features is None:
            self.feature = list(range(self.M))
        else:
            self.feature = features
        if barcodes is None:
            self.barcode = list(range(self.N))
        else:
            self.barcode = barcodes
