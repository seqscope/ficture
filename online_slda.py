# Functions for fitting augmented Latent Dirichlet Allocation
# with online variational Bayes (VB).
# Adapted from Matthew D. Hoffman (2010)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys, io, os, re, time, copy, subprocess

packages = "numpy,scipy,sklearn".split(',')
for pkg in packages:
    if not pkg in sys.modules:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", pkg])


import numpy as np
from scipy.special import gammaln, psi, logsumexp
from scipy.sparse import *
import sklearn.preprocessing

# Add directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import scorpus, utilt

class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """
    def __init__(self, vocab, K, D, alpha = None, eta = None, tau0=9, kappa=0.7, iter_inner = 50, tol = 1e-4, verbose = 0):
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
        ysum = np.asarray(batch.mtx.sum(axis = 1))
        Xb = batch.mtx @ self._Elog_beta.T
        # Initialize the variational distribution q(theta|gamma)
        if batch.alpha is None:
            batch.alpha = np.broadcast_to(self._alpha, (batch.n, self._K))
        Elog_theta = utilt.dirichlet_expectation(batch.alpha)
        expElog_theta = np.exp(Elog_theta)
        if batch.phi is None:
            batch.phi = batch.psi @ expElog_theta
            # batch.phi = sklearn.preprocessing.normalize(batch.phi, norm='l1', axis=1)
            batch.phi = sklearn.preprocessing.normalize(batch.phi, norm='max', axis=1)

        gamma_old = copy.copy(batch.alpha)
        meanchange = self._tol + 1
        it = 0
        while it < self._max_iter_inner and meanchange > self._tol:
            batch.psi = batch.psi_prior.multiply(np.exp(batch.phi @ Elog_theta.T))
            # batch.psi = sklearn.preprocessing.normalize(batch.psi, norm='l1', axis=1)
            batch.psi = sklearn.preprocessing.normalize(batch.psi, norm='max', axis=1)
            batch.phi = np.multiply(ysum, batch.psi @ Elog_theta) + Xb
            batch.phi = np.exp(batch.phi - logsumexp(batch.phi, axis=1)[:, np.newaxis])
            # batch.phi = np.exp(batch.phi - np.max(batch.phi, axis=1)[:, np.newaxis])
            batch.gamma = batch.alpha + batch.psi.T @ batch.phi
            Elog_theta = utilt.dirichlet_expectation(batch.gamma)
            expElog_theta = np.exp(Elog_theta)
            meanchange = np.abs(batch.gamma - gamma_old).mean()
            gamma_old = copy.copy(batch.gamma)
            it += 1
            if self._verbose > 1 and it % 10 == 0:
                print(f"E-step, update phi, psi, gamma: {it}-th iteration, mean change {meanchange:.4f}")

        sstats = batch.phi.T @ batch.mtx
        batch.ll = logsumexp( np.multiply(ysum, np.log(batch.phi+1e-8)) + Xb, axis = 1).sum() / batch.N
        return sstats

    def approx_score(self, batch):

        score_gamma = 0
        score_beta  = 0
        Elog_theta = utilt.dirichlet_expectation(batch.gamma)

        # E[log p(x | theta, beta)]
        score_pixel = batch.ll
        Xtilde = batch.psi.T @ batch.mtx
        ytilde = Xtilde.sum(axis = 1)
        score_patch = logsumexp(np.multiply(ytilde,  Elog_theta) + Xtilde @ self._Elog_beta.T, axis = 1).sum() / batch.n

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score_gamma += np.sum((batch.alpha - batch.gamma)*Elog_theta)
        score_gamma += np.sum(gammaln(batch.gamma)) - np.sum(gammaln(np.sum(batch.gamma, 1)))
        score_gamma /= batch.n

        # E[log p(beta | eta) - log q (beta | lambda)]
        score_beta += np.sum((self._eta-self._lambda)*self._Elog_beta)
        score_beta += np.sum(gammaln(self._lambda)) - np.sum(gammaln(np.sum(self._lambda, 1)))

        return((score_pixel, score_patch, score_gamma, score_beta, self._n * (score_patch+score_gamma) + score_beta))

    def coherence_pmi(self, pw, pseudo_ct = 200, top_gene_n = 100):
        """
        Topic coherence (pointwise mutual information)
        """
        topic_pmi = []
        top_gene_n = np.min([top_gene_n, self._M])
        pw = pw / pw.sum()
        for k in range(self._K):
            indx = np.argsort(-self._expElog_beta[k, :])[:top_gene_n]
            b = self._expElog_beta[k, indx]
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

    def update_lambda(self, batch):
        """
        Process one minibatch
        """
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # E step to update gamma, phi | lambda for mini-batch
        sstats = self.do_e_step(batch)
        # Estimate likelihood for current values of lambda.
        scores = self.approx_score(batch)
        if self._verbose > 0:
            print(f"{self._updatect}-th global update. Scores: " + ", ".join(['%.2e'%x for x in scores]))
        # Update lambda based on documents.
        self._lambda = (1-rhot) * self._lambda + \
                       rhot * (self._eta + self._n * sstats / batch.n)
        self._Elog_beta = utilt.dirichlet_expectation(self._lambda)
        self._expElog_beta = np.exp(self._Elog_beta)
        self._updatect += 1
        return scores
