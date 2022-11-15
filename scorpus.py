from scipy import sparse
from scipy.special import logit

class corpus:

    def __init__(self):
        self.doc_pts = 0  # Positions of anchor points
        self.n = 0        # Number of anchor points
        self.N = 0        # Number of pixels
        self.M = 0        # Number of features (vocabulary)
        self.psi = None   # N x n: P(j|i)
        self.phi = None   # N x K: P(j|i)
        self.gamma = None # n x K: P(k|n)
        self.alpha = None # Dir prior for gamma
        self.ElogO = None # Prior weight for psi E[log w_{ij}], w_ij \in (0,1]
        self.ll = 0       # log likelihood

    def init_from_matrix(self, mtx, doc_pts, logw, psi = None, phi = None, m_gamma = None, features = None, barcodes = None):
        self.mtx = mtx.tocsr()
        self.N, self.M = mtx.shape
        self.doc_pts = doc_pts
        self.n = self.doc_pts.shape[0]

        assert sparse.issparse(logw), "Invalid logw"
        assert logw.shape == (self.N, self.n), "Inconsistent dimensions of logw"
        self.ElogO = logw
        self.ElogO.eliminate_zeros()
        if psi is not None:
            assert sparse.issparse(psi),  "Invalid psi"
            assert psi.shape  == (self.N, self.n), "Inconsistent dimensions of psi"
            self.psi = psi
            self.psi = (self.ElogO != 0).multiply(self.psi)
        else:
            self.psi = copy.copy(self.ElogO)
            self.psi.data = expit(self.psi.data)

        if phi is not None:
            assert phi.shape[0] == self.N, "Inconsistent dimensions of phi"
            self.phi = phi
            self.phi = self.phi / self.phi.sum(axis = 1).reshape((-1, 1))

        if m_gamma is not None:
            assert m_gamma.shape[0] == self.n, "Inconsistent dimensions of gamma"
            self.alpha = m_gamma

        if features is None:
            self.feature = list(range(self.M))
        else:
            self.feature = features
        if barcodes is None:
            self.barcode = list(range(self.N))
        else:
            self.barcode = barcodes
