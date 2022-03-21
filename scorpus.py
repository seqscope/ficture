import os, re, copy
from scipy import sparse
import sklearn.preprocessing

class corpus:

    def __init__(self):
        self.doc_pts = 0 # Positions of anchor points
        self.n = 0  # Number of anchor points
        self.N = 0 # Number of pixels
        self.M = 0 # Number of features (vocabulary)
        self.psi = 0
        self.phi = None
        self.gamma = None
        self.alpha = None
        self.ll = 0

    def init_from_matrix(self, mtx, doc_pts, m_psi, m_phi = None, m_gamma = None, features=None, barcodes=None):
        self.mtx = mtx.tocsr()
        self.N, self.M = mtx.shape
        self.doc_pts = doc_pts
        self.n = self.doc_pts.shape[0]
        assert sparse.issparse(m_psi), "Invalid psi"
        assert m_psi.shape[0] == self.N and m_psi.shape[1] == self.n, "Inconsistent dimensions of psi"
        if m_phi is not None:
            assert m_phi.shape[0] == self.N, "Inconsistent dimensions of phi"
            self.phi = m_phi
            self.phi = self.phi / self.phi.sum(axis = 1).reshape((-1, 1))
        if m_gamma is not None:
            assert m_gamma.shape[0] == self.n, "Inconsistent dimensions of gamma"
            self.alpha = m_gamma

        self.psi = m_psi
        if not sparse.isspmatrix_csr(self.psi):
            self.psi = self.psi.tocsr()
        self.psi_prior = copy.copy(self.psi)
        if features is None:
            self.feature = list(range(self.M))
        else:
            self.feature = features
        if barcodes is None:
            self.barcode = list(range(self.N))
        else:
            self.barcode = barcodes
