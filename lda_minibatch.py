from scipy import sparse

class minibatch:

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
