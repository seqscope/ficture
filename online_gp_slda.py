import sys, re, time, string, copy

import scorpus_gp, utilt

import numpy as np
import scipy.special as sp
from scipy.special import gammaln, psi, logsumexp, expit
import scipy.sparse
from scipy.sparse import *
import sklearn.preprocessing
import random
from sksparse.cholmod import cholesky



class GPSBLDA:
    """
    Spatial LDA using GP and stick breaking
    """

    def __init__(self, vocab, K, D, alpha = None, eta = None, tau0=9, kappa=0.7,\
                 iter_inner = 50, tol = 1e-4, verbose = 0, seed=1984):
        """
        Arguments:
        K: Number of topics
        vocab: A set of features.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.
        """

        np.random.seed(seed)
        random.seed(seed)

        self._vocab = vocab
        self._K = K
        self._M = len(self._vocab)
        self._n = D
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0
        self._max_iter_inner = iter_inner
        self._tol = tol
        self._verbose = verbose
        self.eps0 = 1e-8
        self.eps1 = 1. - 1e-8

        self._alpha = alpha
        if self._alpha is None:
            self._alpha = np.ones(self._K)/self._K
        elif np.isscalar(self._alpha):
            self._alpha = np.ones(self._K)*self._alpha
        else:
            self._alpha = self._alpha.reshape(-1)
            assert self._alpha.shape[0] == self._K, "Invalid alpha"

        self._eta = eta
        if self._eta is None:
            self._eta = (np.ones(self._M)/self._K).reshape((1,-1))
        elif np.isscalar(self._eta):
            self._eta = (np.ones(self._M)*self._eta).reshape((1,-1))
        else:
            self._eta = self._eta.reshape((1, -1))
            assert self._eta.shape[1] == self._M, "Invalid eta"


    def init_global_parameter(self, m_lambda=None):
        # Initialize the variational distribution q(beta|lambda)
        if m_lambda is None:
            self._lambda = 1*np.random.gamma(100., 1./100., (self._K, self._M))
        else:
            self._lambda = m_lambda
        self._Elog_beta = utilt.dirichlet_expectation(self._lambda)
        self._expElog_beta = np.exp(self._Elog_beta)

    def do_e_step(self, batch):
        """
        Update local parameters, compute sufficient stats for M step
        """
        sig_diag = np.zeros((batch.n, self._K))
        for k in range(self._K):
            sig_diag[:,k] = batch.sig_new[k].diagonal()
        mrg_var = np.power(batch.mu_new_mtx, 2) + sig_diag

        phi_const = 0.5 * batch.mu_new_mtx
        v = np.cumsum(phi_const, axis = 1)
        for k in range(1, self._K-1):
            phi_const[:, k] -= v[:, k-1]

        pg_mean = np.zeros((batch.n, self._K))
        for i in range(batch.n):
            pg_mean[i, ] = utilt.pg_mean(1., batch.mu_new_mtx[i,:].reshape(-1))
        pg_mean[:, self._K - 1] = 0

        for i in range(batch.n):

            nnz = batch.mtx[i, ].nonzero()[1]
            indx_r = list(nnz) * self._K
            indx_c = [x for x in range(self._K) for y in range(len(nnz))]

            pgik = [x for x in pg_mean[i, :] for y in range(len(nnz))]
            pgik = coo_matrix((pgik, (indx_r, indx_c)), shape=(self._M, self._K)).toarray() # M x K

            v = sp.expit(batch.mu_new_mtx[i, :])
            s = 1. - 1e-6 - v[0]
            for j in range(1, self._K-1):
                v[j] = v[j] * s
                s -= v[j]
            v[-1] = s
            v = np.clip(v, self.eps0, self.eps1)
            v = v / v.sum()
            v = [x for x in v for y in range(len(nnz))]
            phi_imk = coo_matrix((v, (indx_r, indx_c)), shape=(self._M, self._K)).toarray() # M x K
            phi_imk = np.multiply(phi_imk, self._expElog_beta.T)
            phi_imk = sklearn.preprocessing.normalize(phi_imk, norm='l1', axis=1)
            phi_fix = copy.copy(phi_imk[:, -1])

            mrg_var_mk = [x for x in mrg_var[i, :] for y in range(len(nnz))]
            mrg_var_mk = coo_matrix((mrg_var_mk, (indx_r, indx_c)), shape=(self._M, self._K)).toarray()
            mrg_mu_mk = [x for x in phi_const[i, :] for y in range(len(nnz))]
            mrg_mu_mk = coo_matrix((mrg_mu_mk, (indx_r, indx_c)), shape=(self._M, self._K)).toarray()

            it = 0
            phi_old = np.ones(self._K) / self._K
            delta_phi = self._tol + 1
            while it < self._max_iter_inner and delta_phi > self._tol:
                pg_imk = np.multiply(pgik, phi_imk)
                for k in range(self._K-2, -1, -1):
                    pg_imk[:, k] += pg_imk[:, k+1]
                phi_imk = np.multiply(mrg_var_mk, pg_imk)
                for k in range(1, self._K-1):
                    phi_imk[:, k] += phi_imk[:, k-1]
                phi_imk = self._Elog_beta.T + mrg_mu_mk - 0.5 * phi_imk
                for k in range(1, self._K):
                    phi_imk[:, k] -= k * np.log(2)
                for j in nnz:
                    v = phi_imk[j, :-1]
                    phi_imk[j, :-1] = np.exp(v - logsumexp(v)) * (1.-phi_fix[j])
                    phi_imk[j,  -1] = phi_fix[j]
                phi_imk = phi_imk / phi_imk.sum(axis = 1)[:, np.newaxis]
                phi_imk = np.multiply(batch.mtx[i, ].toarray().T, phi_imk)
                phi_ik = np.asarray(phi_imk.sum(axis = 0)).squeeze()
                batch.phi_ik[i, :] = copy.copy(phi_ik)
                phi_ik = phi_ik / phi_ik.sum()
                delta_phi = np.max(np.abs(phi_ik - phi_old))
                phi_old = copy.copy(phi_ik)
                it += 1
                if self._verbose > 1 and it % 10 == 0:
                    print(f"Update local parameters. {i}-th unit, {it}-th iteration, {delta_phi:.3e}")

            batch.phi_mk += phi_imk
            batch.phi[i, ] = phi_old
            if self._verbose > 0 and i % 50 == 0:
                print(f"Update local parameters. {i}-th unit")

        batch.ll = logsumexp(batch.mtx @ self._Elog_beta.T + np.log(batch.phi), axis = 1)

    def approx_score(self, batch):

        # E[log p(x | u, beta)]
        score_patch = batch.ll.mean()

        # E[log p(beta | eta) - log q (beta | lambda)]
        score_beta = np.sum((self._eta-self._lambda) * self._Elog_beta)
        score_beta += np.sum(gammaln(self._lambda)) - np.sum(gammaln(np.sum(self._lambda, 1)))

        return (score_patch, score_beta)

    def update_lambda(self, batch):
        """
        Process one minibatch
        """
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # E step to update phi | lambda, u for mini-batch
        self.do_e_step(batch)
        # Estimate likelihood for current values of lambda.
        scores = self.approx_score(batch)
        if self._verbose > 0:
            print(f"{self._updatect}-th global update. Scores: " + ", ".join(['%.2e'%x for x in scores]))
        # Update lambda based on documents.
        it = 0
        meanchange = self._tol + 1.
        pg_b = copy.copy(batch.phi_ik)
        for k in range(self._K-2,-1,-1):
            pg_ik[:, k] += pg_ik[:, k+1]
        mu_old = copy.copy(batch.mu_new_mtx[:, :(self._K-1)])
        while it < self._max_iter_inner and meanchange > self._tol:
            pg_ik = pg_mean(pg_b, np.abs(batch.mu_new_mtx)+self.eps0)
            for k in range(self._K-1):
                batch.mu_new_mtx[:, k] = scipy.sparse.linalg.spsolve(batch.sig_inv_org + scipy.sparse.diags(pg_ik[:, k], 0), batch.phi_ik[:, k])
            meanchange = np.max(np.abs(mu_old - batch.mu_new_mtx[:, :(self._K-1)]), axis = 1).mean(axis = 0)
            if self._verbose > 1:
                print(f"({self._updatect},{it})-th global update. mean change of mu: {meanchange:.2e}")
            it += 1
        self._lambda = (1-rhot) * self._lambda + \
                       rhot * (self._eta + self._n * batch.phi_mk / batch.n)
        self._Elog_beta = utilt.dirichlet_expectation(self._lambda)
        self._expElog_beta = np.exp(self._Elog_beta)
        self._updatect += 1
        return scores
