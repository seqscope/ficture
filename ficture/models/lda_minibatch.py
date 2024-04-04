from scipy import sparse
import numpy as np

class Minibatch:

    def __init__(self, mtx, features = None, barcodes = None):
        assert sparse.issparse(mtx), "Invalid matrix - must be sparse"
        self.mtx = mtx.tocsr()
        self.n, self.M = mtx.shape
        self.gamma = None # n x K: P(k|n)
        self.alpha = None # Prior for gamma
        self.ll = 0       # log likelihood
        if features is None:
            self.feature = list(range(self.M))
        else:
            self.feature = features
        if barcodes is None:
            self.barcode = list(range(self.n))
        else:
            self.barcode = barcodes

    def init_from_matrix(self, gamma = None, alpha = None):
        if gamma is not None:
            assert gamma.shape[0] == self.n and len(gamma.shape) == 2, "Inconsistent dimensions of gamma"
            self.gamma = gamma
        if alpha is not None:
            self.alpha = alpha

class PairedMinibatch:

    def __init__(self, mtx_focal, mtx_buffer, buffer_weight = None, features = None, barcodes = None):
        assert sparse.issparse(mtx_focal), "Invalid matrix - must be sparse"
        assert sparse.issparse(mtx_buffer), "Invalid matrix - must be sparse"
        self.mtx = mtx_focal.tocsr()
        self.mtx_buffer = mtx_buffer.tocsr()
        self.n, self.M = self.mtx.shape
        self.gamma = None # n x K: P(k|n)
        self.gamma_buffer = None
        self.alpha = None # Prior for gamma
        self.ll = 0       # log likelihood
        if buffer_weight is None:
            self.buffer_weight = np.ones(self.n) * .5
        else:
            self.buffer_weight = np.array(buffer_weight).reshape(-1)
        if features is None:
            self.feature = list(range(self.M))
        else:
            self.feature = features
        if barcodes is None:
            self.barcode = list(range(self.n))
        else:
            self.barcode = barcodes

    def init_from_matrix(self, gamma = None, gamma_buffer = None, alpha = None):
        if gamma is not None:
            assert gamma.shape[0] == self.n and len(gamma.shape) == 2, "Inconsistent dimensions of gamma"
            self.gamma = gamma
        if gamma_buffer is not None:
            assert gamma_buffer.shape[0] == self.n and len(gamma_buffer.shape) == 2, "Inconsistent dimensions of gamma_buffer"
            self.gamma_buffer = gamma_buffer
        if alpha is not None:
            self.alpha = alpha
