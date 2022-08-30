# Additively regularized topic model applied to spatial transcriptomics

import sys, os, re, copy, warnings
import numpy as np
from scipy.special import logsumexp
from scipy.sparse import *
import sklearn.preprocessing

# Add directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utilt

class SARTM:
    """
    Spatial factor model with additive regularization
    """
    def __init__(self, vocab, K, D, R={},\
                tau0=9, kappa=0.7, iter_inner=50, tol=1e-4, verbose=0):
        """
        Arguments:
        K: Number of topics
        vocab: A set of features.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        R: A dictionary of additive regularizers and their corresponding
            parameters {name: tuple/list of parameters}
            Accepted regularizers include
            sparse_assign: push factor loadings away from unif/flat
            known_marker: each marker is only highly expressed in one factor
            known_profile: full factor specific feature distribution
            sparse_profile: push factor profiles away from unif/flat
            decorr_profile: encourage factors to be dissimilar from each other
            The first parameter is always the scaling coefficient
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate (exponential decay) - should be between
            (0.5, 1.0] to guarantee asymptotic convergence.
        """
        self._vocab = vocab
        self._K = K
        self._M = len(self._vocab)
        self._n = D
        self._R = copy.copy(R);
        self._rhot = 1
        self._tau0 = max([tau0, 0]) + 1
        self._kappa = np.clip(kappa, 0.51, 0.99)
        self._updatect = 0
        self._max_iter_inner = iter_inner
        self._tol = tol
        self._verbose = verbose
        # Parse regularizer
        self._tau = {} # Weights (fixed hyper-parameters) of regularizers
        self.keyR_theta = [k for k in ['sparse_assign'] if k in self._R]
        self.keyR_phi = [k for k in ['known_marker','known_profile','sparse_profile','decorr_profile'] if k in self._R]
        for k in self.keyR_theta + self.keyR_phi:
            self._tau[k] = self._R[k][0]
        self._marker_mtx = None
        self._factor_name = {}
        self._validate_prior()
        self._reg_theta = len(self.keyR_theta) > 0
        self._reg_phi   = len(self.keyR_phi) > 0
        if self._reg_phi:
            self.dRdPhi = np.zeros(self._K, self._M)
        self.phi = np.zeros(self._K, self._M)
        self.nkw = np.zeros(self._K, self._M)


    def init_global_parameter(self, phi=None):
        """
        Initialize factor profiles
        """
        self.phi = phi # K x M
        if self.phi is None:
            self.phi = 1*np.random.gamma(100., 1./100., (self._K, self._M))
        self.phi = sklearn.preprocessing.normalize(sefl.phi, norm='l1', axis=1)


    def do_e_step(self, batch):
        """
        Compute sufficient stats for M step
        """
        batch.ll = 0
        llv = np.zeros(batch.N)
        for k in range(self._K):
            batch.piwjk[k][-1] = csr_array(([],([],[])), shape=(batch.N, self._M), dtype=float)
            for j in range(batch.n):
                batch.piwjk[k][j] = batch.pjk[j][k] * batch.mtx.multiply(batch.pij[:, j])
                batch.piwjk[k][j].eliminate_zeros()
                batch.piwjk[k][-1] += batch.piwjk[k][j]
            llv += batch.piwjk[k][-1].sum(axis = 0)
        batch.ll = logsumexp(llv - np.log(batch.mtx.sum(axis = 1)).sum() ) / batch.N
        batch.nkw = np.zeros((self._K, self._M))
        batch.njk = np.zeros((batch.n, self._K))
        batch.nij.data = 0
        for k in range(self._K):
            batch.nkw[k, :] = batch.piwjk[k][-1].sum(axis = 0).reshape((1, -1))
            for j in range(batch.n):
                batch.njk[j][k] = batch.piwjk[k][j].data.sum()
                batch.nij[:, j]+= csr_matrix(batch.piwjk[k][j].data.sum(axis = 1))

    def do_m_step(self, batch):
        """
        Update local parameter pij and pjk
        """
        batch.pij = sklearn.preprocessing.normalize(batch.nij, norm='l1', axis=1)
        batch.pjk = batch.njk
        if self._reg_theta:
            batch.pjk -= self._tau["sparse_assign"] * self._R["sparse_assign"].reshape((1, -1))
        batch.pjk = np.clip(batch.pjk, 0, np.inf)
        batch.pjk = sklearn.preprocessing.normalize(batch.pjk, norm='l1', axis=1)

    def update_phi(self, batch):
        """
        Update global parameter pkw
        """
        self._compute_dRdPhi()
        delta_phi = batch.nkw
        if self._reg_phi:
            delta_phi -= self.dRdPhi
        delta_phi = np.clip(delta_phi, 0, np.inf)
        delta_phi = sklearn.preprocessing.normalize(delta_phi, norm='l1', axis=1)
        self._rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self.phi = (1-self._rhot) * self.phi + self._rhot * delta_phi
        self._updatect += 1
        if self._verbose > 0:
            print(f"{self._updatect}-th global update, {batch.iter} iterations, ll= {batch.ll:.3e}")

    def minibatch(self, batch):
        assert batch.mtx.shape[1] == self._M, "Incompatible input feature"
        # Initialize
        batch.pij = sklearn.preprocessing.normalize(batch.pij, norm='l1', axis=1)
        batch.nij = batch.pij.multiply( batch.ni.reshape((-1, 1)) )
        if batch.pjk is None:
            njw = batch.pij.T @ batch.mtx
            batch.pjk = np.zeros((batch.n, self._K))
            for k in range(self._K):
                batch.pjk[:, k] = np.exp( ( njw.multiply(np.log(self.phi[k, ])) ).sum(axis = 1) ).reshape((-1, 1))
        batch.pjk = sklearn.preprocessing.normalize(batch.pjk, norm='l1', axis=1)
        batch.piwjk = {}
        for k in range(self._K):
            batch.piwjk[k] = {}
        batch.delta_p = self._tol * 10
        batch.iter = 0
        batch.ll = 1
        dp0 = self._tol * 10
        while batch.delta_p > self._tol and batch.iter < self._max_iter_inner:
            old_pij = copy.copy(batch.pij)
            old_pjk = copy.copy(batch.pjk)
            old_ll  = batch.ll
            self.do_e_step(batch)
            self.do_m_step(batch)
            dp0 = np.abs(old_ll - batch.ll )/batch.ll
            if dp0 < self._tol:
                break
            dp1 = np.abs(old_pij - batch.pij).max()
            dp2 = np.abs(old_pjk - batch.pjk).max()
            batch.delta_p = 0.5 * (dp1 + dp2)
            batch.iter += 1
        self.update_phi(batch)
        return scores


    def _validate_prior(self):
        """
        Check user input prior and hyper-parameters
        """
        # Check if marker info is valid
        if "known_marker" in self.keyR_phi:
            v = self._R.get("known_marker")
            assert len(v) >= 2 and isinstance(v[1], dict), "Invalid marker info"
            v = v[1]
            for i,k in enumerate(v.keys()):
                if i >= self._K:
                    warnings.warm(f"Markers have to be assigned to no more than K factors")
                    break
                if not isinstance(k, int):
                    self._factor_name[i] = k
                    v[i] = v.pop(k)
                    k = i
                if k >= self._K:
                    _ = v.pop(k)
                    warnings.warm(f"Only markers assigned to factor index 0 ~ k-1 or the first K arbitrary names are used, key {k} is invalid")
                    continue
                u = []
                for m in [k]:
                    if isinstance(m, int):
                        if m >= self._M or m < 0:
                            warnings.warm(f"Marker {m} is not in the vocabulary")
                            continue
                        u.append(m)
                    else:
                        if m not in self._vocab:
                            warnings.warm(f"Marker {m} is not in the vocabulary")
                            continue
                        u.append(self._vocab.index(m))
                v[k] = u
            if len(v) == 0:
                self.keyR_phi = [k for k in self.keyR_phi if k != "known_marker"]
                _ = self._tau.pop("known_marker")
            else:
                dv, iv, jv = [], [], []
                i = 0
                for k,u in v.itmes():
                    for w in u:
                        iv += [l for l in range(self._K) if l != k]
                        jv += [w] * self._K - 1
                dv = [1] * len(iv)
                for k,u in v.itmes():
                    dv.append([0] * len(u))
                    iv.append([k] * len(u))
                    jv += u
                self._marker_mtx = coo_matrix((dv, (iv, jv)), shape=(self._K, self._M), dtype=int).tocsr()
                self._marker_mtx.eliminate_zeros()

        # Check prior profiles are valid
        if "known_profile" in self.keyR_phi:
            v = self._R.get("known_profile")
            assert len(v) >= 2 and isinstance(v[1], dict), "Invalid prior profiles"
            v = v[1]
            for i,k in self._factor_name.items():
                if k in v:
                    v[i] = v.pop(k)
            _i = len(self._factor_name)
            for i,k in enumerate(v.keys()):
                if i >= self._K:
                    warnings.warm(f"Markers have to be assigned to no more than K factors")
                    break
                if not isinstance(k, int):
                    self._factor_name[_i] = k
                    v[_i] = v.pop(k)
                    k = _i
                    _i += 1
                if k >= self._K:
                    _ = v.pop(k)
                    warnings.warm(f"Only markers assigned to factor index 0 ~ k-1 or the first K arbitrary names are used, key {k} is invalid")
                    continue
                v[k] = np.asarray(v[k]).reshape(-1)
                assert len(v[k]) == self._M, f"Invalid prior for factor {k}"
                v[k] = v[k] / v[k].sum()
            if len(v) == 0:
                self.keyR_phi = [k for k in self.keyR_phi if k != "known_profile"]
                _ = self._tau.pop("known_profile")

        # Check sparsity parameter
        if "sparse_profile" in self.keyR_phi:
            v = self._R.get("sparse_profile")
            assert len(v) >= 2, "Invalid sparsity parameter for factor profile"
            v = np.asarray(v[1]).reshape(-1)
            assert len(v) == self._M, "Invalid sparsity parameter for factor profile"
            v = v/v.sum()
        if "sparse_assign" in self.keyR_theta:
            v = self._R.get("sparse_assign")
            if len(v) < 2:
                v = np.ones(self._K) / self._K
            else:
                v = np.array(v[1]).reshape(-1)
                assert len(v) == self._K
                v = v / v.sum()

    def _compute_dRdPhi(self):
        """
        dR / dPhi: derivatives of additive regularizers w.r.t. global parameter
        This function actually computes Phi.multiply(dR/dPhi) (for cancellation)
        """
        if not self._reg_phi:
            return
        if "known_profile" in self.keyR_phi:
            for k,v in self._R["known_profile"].items():
                self.dRdPhi[k, :] = v + self.phi[k, :]
                nnz = self.dRdPhi[k, :] > 0
                self.dRdPhi[k][nnz] = 1./self.dRdPhi[k][nnz]
                self.dRdPhi[k, :] = self._tau["known_profile"] * self.dRdPhi[k, :].multiply(v.multiply(self.phi[k, :]) )

        if "known_marker" in self.keyR_phi:
            self.dRdPhi += self._tau["known_marker"] * self.phi.multiply(self._marker_mtx)

        if "sparse_profile" in self.keyR_phi:
            self.dRdPhi += self._tau["sparse_profile"] * self.phi.multiply(self._R["sparse_profile"].reshape((1, -1)) )

        if "decorr_profile" in self.keyR_phi:
            for k in range(self._K):
                v = np.zeros(self._M)
                for l in range(self._K):
                    if l == k:
                        continue
                    v += self.phi[k, :].multiply(self.phi[l, :])
                self.dRdPhi[k, :] += self._tau["decorr_profile"] * v
