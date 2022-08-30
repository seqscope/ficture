import sys, os, re, copy, subprocess
from scipy import sparse
import sklearn.preprocessing

class BATCH:

    def __init__(self):
        self.jxy = 0 # Positions of anchor points
        self.n = 0 # Number of anchor points
        self.N = 0 # Number of pixels
        self.M = 0 # Number of features (equal to vocabulary size)
        self.map_ji = {}
        self.map_ij = {}
        self.piwjk = {}
        self.nkw = None
        self.nij = None
        self.njk = None
        self.pij = None
        self.pjk = None
        self.ni  = None
        self.ll = 0
        self.delta_p = 1
        self.iter = 0

    def init_from_matrix(self, mtx, jxy, pij, pjk = None):
        assert sparse.issparse(pij), "Invalid pij"
        assert sparse.issparse(mtx), "Invalid mtx"
        self.mtx = mtx.tocsr()
        self.N, self.M = mtx.shape
        self.n = self.jxy.shape[0]
        assert pij.shape[0] == self.N and pij.shape[1] == self.n, "Inconsistent dimensions of pij"
        if pjk is not None:
            assert pjk.shape[0] == self.n, "Inconsistent dimensions of pjk"
        self.pjk = pjk.tocsr()
        self.pij = pij.tocsr()
        self.jxy = jxy
        iv, jv = self.pij.nonzero()
        for i in range(self.N):
            self.map_ij[i] = []
        for j in range(self.n):
            self.map_ji[j] = []
        for k, i in enumerate(iv):
            map_ji[jv[k]].append(i)
            map_ij[i].append(jv[k])
        self.ni = np.array(self.mtx.sum(axis = 1)).reshape(-1)
