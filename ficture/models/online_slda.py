# Functions for fitting augmented Latent Dirichlet Allocation
# with online variational Bayes (VB).

import sys, io, os, re, time, copy, subprocess, logging

import numpy as np
from sklearn.utils import check_random_state
from scipy.special import gammaln, psi, logsumexp, expit, logit
from scipy.sparse import *
from sklearn.preprocessing import normalize
from sklearn.decomposition._online_lda_fast import (
    _dirichlet_expectation_1d, _dirichlet_expectation_2d,
)

class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """
    def __init__(self, vocab, K, N, alpha = None, eta = None, tau0=9, kappa=.7, zeta = 0, iter_inner = 50, tol = 1e-4, iter_gamma = 10, verbose = 0, seed = None):
        """
        Arguments:
        K: Number of topics
        vocab: A list of features
        D:     Total number of pixels in the entire dataset.
        alpha: Hyperparameter for the prior on weight vectors theta
        eta:   Hyperparameter for prior on topics beta
        tau0:  A (non-negative) learning parameter that downweights earl iterations
        kappa: Learning rate (exponential decay), should be between
               (0.5, 1.0] to guarantee asymptotic convergence.
        --- Experimental ---
        zeta:  Weight of the proximal contamination penalty
        iter_gamma: Maximum number of iterations when maximizing the "penalized ELBO" w.r.t. gamma (there is no analytical solution)
        """
        self._vocab = vocab
        self._K = K
        self._M = len(self._vocab)
        self._N = N
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._zeta = zeta
        self._updatect = 0
        self._max_iter_inner = iter_inner
        self._max_iter_gamma = iter_gamma
        self._tol = tol
        self._verbose = verbose
        self._Elog_beta = None      # K x M
        self._lambda = None         # K x M
        self.rng_ = check_random_state(seed)

        self._alpha = alpha # 1D array
        if self._alpha is None:
            self._alpha = np.ones(self._K)/self._K
        elif np.isscalar(self._alpha):
            self._alpha = np.ones(self._K)*self._alpha
        else:
            self._alpha = self._alpha.reshape(-1)
            assert self._alpha.shape[0] == self._K, "Invalid alpha"

        self._eta = eta     # 1D array
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
            self._lambda = self.rng_.gamma(100., 1./100., (self._K, self._M))
        else:
            self._lambda = m_lambda
        self._Elog_beta = _dirichlet_expectation_2d(self._lambda)

    def do_e_step(self, batch):
        """
        Update local parameters, compute sufficient stats for M step
        """
        Xb = batch.mtx @ self._Elog_beta.T         # Dense, N x K
        if issparse(Xb):
            Xb = Xb.toarray()
        c_indx, r_indx = batch.psi.nonzero()
        # Initialize the variational distribution q(theta|gamma)
        if batch.alpha is None:
            batch.alpha = np.broadcast_to(self._alpha, (batch.n, self._K))
        if batch.gamma is None:
            batch.gamma = copy.copy(batch.alpha)
        gamma_old = copy.copy(batch.gamma)
        Elog_theta = _dirichlet_expectation_2d(batch.gamma) # n x K
        meanchange = self._tol + 1
        it = 0

        while it < self._max_iter_inner and meanchange > self._tol:

            batch.phi = batch.psi @ Elog_theta + Xb
            batch.phi = np.exp(batch.phi -\
                               logsumexp(batch.phi, axis = 1).reshape((-1, 1)) )
            psi_hat = batch.phi[c_indx, 0] * Elog_theta[r_indx, 0] # \sum_k phi_ik Elog[theta_jk]
            for k in range(1, self._K):
                psi_hat += batch.phi[c_indx, k] * Elog_theta[r_indx, k]
            psi_hat += np.multiply(Xb, batch.phi).sum(axis = 1).reshape(-1)[c_indx] # \sum_k phi_ik log Pik
            batch.psi.data = psi_hat
            batch.psi += batch.ElogO
            batch.psi.data = np.exp(batch.psi.data)
            batch.psi = normalize(batch.psi, norm='l1', axis=1)
            batch.gamma = batch.alpha + batch.psi.T @ batch.phi
            Elog_theta = _dirichlet_expectation_2d(batch.gamma)

            meanchange = np.abs(batch.gamma - gamma_old).max(axis=1).mean()
            gamma_old = copy.copy(batch.gamma)
            it += 1
            # if self._verbose > 2: # debug
            #     print( np.around([batch.psi.sum(axis = 1).min(),\
            #                       batch.phi.sum(axis = 1).min()], 2) )
            if self._verbose > 2 or (self._verbose > 1 and it % 10 == 0):
                logging.info(f"E-step, update phi, psi, gamma: {it}-th iteration, mean change {meanchange:.4f}")

        sstats = batch.phi.T @ batch.mtx # K x M
        batch.ll = batch.psi.T @ batch.mtx @ self._Elog_beta.T
        ll_norm = logsumexp(batch.ll, axis = 1)
        ll_tot = 0
        for k in range(self._K):
            ll_tot += (batch.ll[:, k] - ll_norm).sum()
        batch.ll = ll_tot / batch.n
        return sstats

    def update_lambda_penalized(self, batch):
        assert self._zeta > 0 and self._zeta < 1, "zeta must be in (0, 1) for penalized update"
        assert batch.anchor_adj is not None, "batch.anchor_adj must be provided for penalized update"
        if issparse(batch.anchor_adj):
            assert (batch.anchor_adj.diagonal() > 0).sum() >= batch.n, "anchor_adj must contain self-loops"
        else:
            assert (np.diag(batch.anchor_adj) > 0).sum() >= batch.n, "anchor_adj must contain self-loops"
        # E step to update gamma, phi | lambda for mini-batch
        lambda_org = self._eta + self.do_e_step(batch) # K X M
        theta = normalize(batch.gamma, norm='l1', axis=1) # n x K, E[theta]
        ckl = theta.T @ normalize(batch.anchor_adj, norm='l1', axis=1) @ theta
        ckl /= theta.sum(axis = 0).reshape((-1, 1)) # K x K, factor spatial proximity
        np.fill_diagonal(ckl, 0)
        lam_sum = lambda_org.sum(axis = 1)
        it = 0
        meanchange = self._tol + 1
        while it < self._max_iter_gamma and meanchange > self._tol:
            lambda_new = np.zeros((self._K, self._M))
            for k in range(self._K):
                avg = ckl[[k], :] @ (normalize(lambda_org, norm='l1', axis=1)*lam_sum[k])
                adj = np.clip(lambda_org[k, :] - self._zeta * avg, 0, None)
                lambda_new[k, :] = adj / adj.sum() * lam_sum[k]
            v = np.abs(lambda_new - lambda_org).max(axis = 1) / lambda_org.sum(axis = 1)
            meanchange = v.mean()
            if self._verbose > 2 or (self._verbose > 1 and it % 10 == 0) or (self._verbose > 1 and it == self._max_iter_gamma - 1):
                logging.info(f"Penalized M-step, update lambda : {it}-th iteration, mean max change {meanchange:.4f}")
            lambda_org = copy.copy(lambda_new)
            it += 1

        if self._verbose > 0:
            # Estimate likelihood for current values of lambda.
            scores = self.approx_score(batch)
            logging.info(f"{self._updatect}-th global update. Scores: " + ", ".join(['%.2e'%x for x in scores]))

        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._lambda = (1-rhot) * self._lambda + \
                       rhot * ((self._N / batch.N) * lambda_org)
        self._Elog_beta = _dirichlet_expectation_2d(self._lambda)
        self._updatect += 1
        return 1

    def update_lambda(self, batch):
        """
        Process one minibatch
        """
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.

        # E step to update gamma, phi | lambda for mini-batch
        sstats = self.do_e_step(batch)
        # Estimate likelihood for current values of lambda.
        scores = self.approx_score(batch)
        if self._verbose > 0:
            logging.info(f"{self._updatect}-th global update. Scores: " + ", ".join(['%.2e'%x for x in scores]))

        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._lambda = (1-rhot) * self._lambda + \
                       rhot * ((self._N / batch.N) * (self._eta + sstats) )
        self._Elog_beta = _dirichlet_expectation_2d(self._lambda)
        self._updatect += 1
        return scores

    def approx_score(self, batch):

        score_gamma = 0
        score_beta  = 0
        Elog_theta = _dirichlet_expectation_2d(batch.gamma)

        # E[log p(x | theta, phi, beta)]
        score_pixel = batch.mtx @ self._Elog_beta.T
        score_pixel = np.multiply(batch.phi, score_pixel/batch.N).sum()
        score_patch = batch.ll

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score_gamma += np.sum((batch.alpha - batch.gamma)*Elog_theta)
        score_gamma += np.sum(gammaln(batch.gamma)) - np.sum(gammaln(np.sum(batch.gamma, 1)))
        score_gamma /= batch.n

        # E[log p(beta | eta) - log q (beta | lambda)]
        score_beta += np.sum((self._eta-self._lambda)*self._Elog_beta)
        score_beta += np.sum(gammaln(self._lambda)) - np.sum(gammaln(np.sum(self._lambda, 1)))

        return((score_pixel, score_patch, score_gamma, score_beta, self._N * (score_patch+score_gamma) + score_beta))

    def coherence_pmi(self, pw, pseudo_ct = 200, top_gene_n = 100):
        """
        Topic coherence (pointwise mutual information)
        """
        topic_pmi = []
        top_gene_n = np.min([top_gene_n, self._M])
        pw = pw / pw.sum()
        for k in range(self._K):
            indx = np.argsort(-self._Elog_beta[k, :])[:top_gene_n]
            b = np.exp(self._Elog_beta[k, indx] )
            b = np.clip(b, 1e-6, 1.-1e-6)
            w = 1. - np.power(1.-pw[indx], pseudo_ct)
            w = w.reshape((-1, 1)) @ w.reshape((1, -1))
            p0 = 1.-np.power(1-b, pseudo_ct)
            p0 = p0.reshape((-1, 1)) @ p0.reshape((1, -1))
            pmi = np.log(p0) - np.log(w)
            np.fill_diagonal(pmi, 0)
            pmi = np.round(pmi.mean(), 3)
            topic_pmi.append(pmi)
        return topic_pmi
