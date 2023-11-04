# Functions for fitting Latent Dirichlet Allocation
# with online variational Bayes (VB)
# following Hoffman et al. 2010

import sys, io, os, re, time, copy, warnings
import numpy as np
from scipy.special import gammaln, psi, logsumexp, expit, logit
from scipy.sparse import issparse
from sklearn.preprocessing import normalize
from joblib.parallel import Parallel, delayed

from ficture.utils import utilt

class OnlineLDAPenalized:
    """
    Implements online VB for LDA as described in Hoffman et al. 2010.
    """
    def __init__(self, vocab, K, N, alpha = None, eta = None, tau0=9, kappa=.7, zeta = 0, iter_inner = 50, tol = 1e-4, iter_gamma = 10, verbose = 0, thread = 1, rng = None, seed = None, proximal = True):
        """
        Arguments:
        vocab: A list of features
        K:     Number of topics
        N:     Estimated total number of units
        alpha: Hyperparameter for the prior on theta
        eta:   Hyperparameter for prior on beta
        tau0:  A (non-negative) learning parameter that downweights earl iterations
        kappa: Learning rate (exponential decay), should be between
               (0.5, 1.0] to guarantee asymptotic convergence.
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
        self._eps = 1e-16
        self._verbose = verbose
        self._thread = thread
        self._proximal = proximal
        self._Elog_beta = None      # K x M
        self._lambda = None         # K x M
        self._expElog_beta = None   # K x M
        self._sstats = None         # K x M

        if self._zeta < 0:
            warnings.warn("Invalid zeta, fall back to default 0")
            self._zeta = 0

        self.rng = rng
        if self.rng is None:
            if seed is None:
                self.rng = np.random.default_rng(int(time.time()))
            else:
                self.rng = np.random.default_rng(seed)

        self._alpha = alpha # Factor weight prior
        if self._alpha is None:
            self._alpha = np.ones(self._K)/self._K
        elif np.isscalar(self._alpha):
            self._alpha = np.ones(self._K)*self._alpha
        elif len(self._alpha.shape) == 1:
            assert self._alpha.shape[0] == self._K, "Invalid alpha"
        else:
            warnings.warn("Invalid alpha, fall back to default uniform 1/K")
            self._alpha = np.ones(self._K)/self._K

        self._eta = eta  # Expression profile prior
        if self._eta is None:
            self._eta = (np.ones(self._M)/self._K).reshape((1,-1))
        elif np.isscalar(self._eta):
            self._eta = (np.ones(self._M)*self._eta).reshape((1,-1))
        elif len(self._eta.shape) == 1:
            self._eta = np.array(self._eta).reshape((1, -1))
            assert self._eta.shape[1] == self._M, "Invalid eta"
        elif len(self._eta.shape) == 2:
            assert self._eta.shape == (self._K, self._M), "Invalid eta"
        else:
            warnings.warn("Invalid eta, fall back to default uniform 1/K")
            self._eta = (np.ones(self._M)/self._K).reshape((1,-1))

    def init_global_parameter(self, _lambda=None):
        # Initialize the variational distribution q(beta|lambda)
        if _lambda is None:
            self._lambda = np.random.gamma(100., 1./100., (self._K, self._M))
        else:
            self._lambda = _lambda
            assert self._lambda.shape == (self._K, self._M), "Invalid lambda"
        if self._lambda.min() <= 0 :
            warnings.warn("Parameters must be positive, will clip to epsilon")
            self._lambda = np.clip(self._lambda, self._eps, np.inf)
        self._Elog_beta = utilt.dirichlet_expectation(self._lambda)
        self._expElog_beta = np.exp(self._Elog_beta)


    def _update_gamma(self, X, _gamma, alpha):
        gamma = copy.copy(_gamma)
        sstats = np.zeros((self._K, self._M))
        expElog_theta = np.exp(utilt.dirichlet_expectation(gamma))
        for j in range(X.shape[0]):
            maxchange = self._tol + 1
            it = 0
            while it < self._max_iter_inner and maxchange > self._tol:
                old_gamma = copy.copy(gamma[j, :])
                phi_norm = expElog_theta[j, :] @ self._expElog_beta + self._eps # 1 x M
                gamma[j, :] = alpha[j, :] +\
                    np.multiply(expElog_theta[j, :], \
                                (X[[j], :].multiply(1./phi_norm)) @ self._expElog_beta.T) # 1 x K
                expElog_theta[j, :] = np.exp(utilt.dirichlet_expectation(gamma[j, :]))
                maxchange = np.abs(old_gamma - gamma[j, :] ).max()
                it += 1
                if self._verbose > 3 or (self._verbose > 2 and it % 10 == 0):
                    print(f"E-step, {j}-th unit, update gamma: {it}-th iteration, max change {maxchange:.4f}")
            phi_norm = expElog_theta[j, :] @ self._expElog_beta + self._eps
            phi = expElog_theta[j, :].reshape((-1, 1)) * self._expElog_beta / phi_norm.reshape((1, -1)) # K x M
            sstats += np.multiply(X[[j], :].toarray(), phi)
        return sstats, gamma


    def do_e_step(self, batch):
        """
        Update local parameters, compute sufficient stats for M step
        """
        # Initialize the variational distribution q(theta|gamma)
        if batch.alpha is None:
            batch.alpha = np.broadcast_to(self._alpha, (batch.n, self._K))
        if batch.gamma is None:
            batch.gamma = self.rng.gamma(100., 1./100., (batch.n, self._K))
        if self._proximal and hasattr(batch, 'mtx_buffer') and batch.mtx_buffer is not None:
            if batch.gamma_buffer is None:
                batch.gamma_buffer = self.rng.gamma(100., 1./100., (batch.n, self._K))

        # Run E-step in parallel
        if self._thread <= 1:
            self._sstats, batch.gamma = self._update_gamma(batch.mtx, batch.gamma, batch.alpha)
        else:
            idx_slices = [idx for idx in utilt.gen_even_slices(batch.n, self._thread)]
            with Parallel(n_jobs=self._thread, verbose=max(0, self._verbose - 2)) as parallel:
                result = parallel(delayed(self._update_gamma)(batch.mtx[idx, :], batch.gamma[idx, :], batch.alpha[idx, :])
                for idx in idx_slices)
            # Collect sufficient statistics
            self._sstats = np.zeros((self._K, self._M))
            for i, v in enumerate(result):
                self._sstats += v[0]
                batch.gamma[idx_slices[i], :] = v[1]

        batch.ll = np.multiply(normalize(batch.gamma, norm='l1', axis=1), batch.mtx @ self._Elog_beta.T).sum() / batch.n

        # E-step on surrounding buffer region
        if self._proximal and hasattr(batch, 'mtx_buffer') and batch.mtx_buffer is not None:
            if self._thread <= 1:
                _, batch.gamma_buffer = self._update_gamma(batch.mtx_buffer, batch.gamma_buffer, batch.alpha)
            else:
                with Parallel(n_jobs=self._thread, verbose=max(0, self._verbose - 2)) as parallel:
                    result = parallel(delayed(self._update_gamma)(batch.mtx_buffer[idx, :], batch.gamma_buffer[idx, :], batch.alpha[idx, :])
                    for idx in idx_slices)
                for i, v in enumerate(result):
                    batch.gamma_buffer[idx_slices[i], :] = v[1]
        return


    def update_lambda(self, batch):
        """
        Process one minibatch
        """
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.

        if not hasattr(batch, 'gamma_buffer') or batch.gamma_buffer is None:
            self._proximal = False
        # E step to update gamma | lambda for mini-batch
        self.do_e_step(batch)

        # Estimate likelihood for current values of lambda.
        scores = 0
        if self._verbose > 0:
            scores = self.approx_score(batch)
            print(f"{self._updatect}-th global update. Scores: " + ", ".join(['%.2e'%x for x in scores]))

        lam_hat = self._eta + self._sstats # K x M
        if self._verbose > 1:
            ldelta = np.abs(normalize(lam_hat, norm='l1', axis=1) - normalize(self._lambda, norm='l1', axis=1)).max(axis = 1)
            print(f"Max relative change: {ldelta.max():.4f}, median change: {np.median(ldelta):.4f}")
        if self._zeta > 0:
            lam_sum = lam_hat.sum(axis=1).reshape((-1, 1))
            if self._proximal:
                # Penalty weighted by spatial proximity
                theta = normalize(batch.gamma, norm='l1', axis=1) # n x K, E[theta]
                theta_buffer = normalize(batch.gamma_buffer, norm='l1', axis=1) # n x K
                ckl = (theta * batch.buffer_weight.reshape((-1, 1))).T @ theta_buffer +\
                    (theta * (1-batch.buffer_weight).reshape((-1, 1))).T @ theta # K x K
                ckl /= theta.sum(axis = 0).reshape((-1, 1)) # row stochastic
                np.fill_diagonal(ckl, 0) # remove self-loop
                if self._verbose > 2:
                    print("Ckl row sum:")
                    print(np.around(ckl.sum(axis = 1), 3))
            else:
                ckl = np.ones((self._K, self._K)) - np.eye(self._K)
                ckl /= self._K - 1
                # Uniformly penalize all pairwise topic similarity
            lam_delta = ckl @ (normalize(self._lambda, norm='l1', axis=1) * lam_sum)
            lam_hat = np.clip(lam_hat - self._zeta * lam_delta, 0, None)
            lam_hat = normalize(lam_hat, norm='l1', axis=1) * lam_sum
            if self._verbose > 1:
                ldelta = np.abs(lam_hat - self._eta - self._sstats).max(axis = 1)
                ldelta /= lam_sum.reshape(-1)
                print(f"Max relative change due to penalty: {ldelta.max():.4f}, median change: {np.median(ldelta):.4f}")


        # Update global parameters
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        doc_ratio = float(self._N) / batch.n
        self._lambda = (1-rhot) * self._lambda + rhot * (doc_ratio * lam_hat)
        self._Elog_beta = utilt.dirichlet_expectation(self._lambda)
        self._expElog_beta = np.exp(self._Elog_beta)
        self._updatect += 1
        if self._verbose > 0:
            post_weight = self._lambda.sum(axis=1)
            post_weight /= post_weight.sum()
            post_weight.sort()
            k = min(10, self._K)
            print(f"(Top) {k} topic weight {post_weight[-k:].sum():.4f}")
            print(np.around(post_weight[-k:], 3))

        return scores


    def transform(self, X, gamma = None, alpha = None):
        assert X.shape[1] == self._M
        if issparse(X):
            X = X.tocsr()
        n = X.shape[0]
        if gamma is None:
            gamma = self.rng.gamma(100., 1./100., (n, self._K))
        if alpha is None:
            alpha = np.broadcast_to(self._alpha, (n, self._K))
        else:
            if len(alpha.shape) == 1:
                alpha = np.broadcast_to(alpha, (n, self._K))
            else:
                assert alpha.shape == (n, self._K), "Invalid alpha input"
        # Run E-step in parallel
        if self._thread <= 1:
            _, gamma = self._update_gamma(X, gamma, alpha)
        else:
            idx_slices = [idx for idx in utilt.gen_even_slices(n, self._thread)]
            with Parallel(n_jobs=self._thread, verbose=max(0, self._verbose - 1)) as parallel:
                result = parallel(delayed(self._update_gamma)(X[idx, :], gamma[idx, :], alpha[idx, :])
                for idx in idx_slices)
            for i, v in enumerate(result):
                gamma[idx_slices[i], :] = v[1]
        gamma = normalize(gamma, norm='l1', axis=1)
        return gamma


    def approx_score(self, batch):
        """
        ELBO
        """
        score_gamma = 0
        score_beta  = 0
        Elog_theta = utilt.dirichlet_expectation(batch.gamma)
        E_theta = normalize(batch.gamma, norm='l1', axis=1)

        # E[log p(x | theta, beta)]
        score_unit = batch.mtx @ self._Elog_beta.T
        score_unit = np.multiply(E_theta, score_unit/batch.n).sum()

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score_gamma += np.sum((batch.alpha - batch.gamma)*Elog_theta)
        score_gamma += np.sum(gammaln(batch.gamma)) - np.sum(gammaln(np.sum(batch.gamma, 1)))
        score_gamma /= batch.n

        # E[log p(beta | eta) - log q (beta | lambda)]
        score_beta += np.sum((self._eta-self._lambda)*self._Elog_beta)
        score_beta += np.sum(gammaln(self._lambda)) - np.sum(gammaln(np.sum(self._lambda, 1)))

        return((score_unit, score_gamma, score_beta, self._N * (score_unit+score_gamma) + score_beta))

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
