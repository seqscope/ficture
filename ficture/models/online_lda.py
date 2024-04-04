"""

=============================================================
Online Latent Dirichlet Allocation with variational inference
=============================================================

This implementation is modified from Matthew D. Hoffman's onlineldavb code
Link: https://github.com/blei-lab/onlineldavb
"""
# sklearn version: 1.4.1
# Author: Chyi-Kwei Yau
# Author: Matthew D. Hoffman (original onlineldavb implementation)
from numbers import Integral, Real
import sys, os, copy
import numpy as np
import scipy.sparse as sp
from joblib import effective_n_jobs
from scipy.special import gammaln, logsumexp
from sklearn.preprocessing import normalize
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils import check_random_state, gen_batches, gen_even_slices
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted, check_non_negative
from sklearn.decomposition._online_lda_fast import (
    _dirichlet_expectation_1d as cy_dirichlet_expectation_1d,
)
from sklearn.decomposition._online_lda_fast import (
    _dirichlet_expectation_2d,
)
from sklearn.decomposition._online_lda_fast import (
    mean_change as cy_mean_change,
)

from ficture.utils.utilt import gen_even_slices_from_list

EPS = np.finfo(float).eps

def _update_doc_distribution(
    X,
    exp_elog_beta,
    doc_topic_prior,
    max_doc_update_iter,
    mean_change_tol,
    cal_sstats,
    random_state,
):
    """E-step: update document-topic distribution.

    Parameters
    ----------
    X               : N x M
    exp_elog_beta   : K x M `exp(E[log(beta)])`.
    doc_topic_prior : float. Prior of document topic distribution `theta`.
    max_doc_update_iter : int. Max number of iterations for updating q(theta)
    mean_change_tol : float. Stopping tolerance for updating q(theta)
    cal_sstats      : bool. Indicate whether to calculate sufficient statistics
    random_state    : RandomState/Generator instance or None

    Returns
    -------
    (gamma, suff_stats) :
        `gamma` is unnormalized topic distribution for each document.
        In the literature, this is `gamma`. we can calculate `E[log(theta)]`
        from it.
        `suff_stats` is expected sufficient statistics for the M-step.
            When `cal_sstats == False`, this will be None.

    """
    is_sparse_x = sp.issparse(X)
    n_samples, n_features = X.shape
    n_topics = exp_elog_beta.shape[0]

    if random_state:
        gamma = random_state.gamma(100.0, 0.01, (n_samples, n_topics)).astype(
            X.dtype, copy=False
        )
    else:
        gamma = np.ones((n_samples, n_topics), dtype=X.dtype)

    # In the literature, this is `exp(E[log(theta)])`
    exp_elog_theta = np.exp(_dirichlet_expectation_2d(gamma))

    # diff on `component_` (only calculate it when `cal_diff` is True)
    suff_stats = (
        np.zeros(exp_elog_beta.shape, dtype=X.dtype) if cal_sstats else None
    )

    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    # These cython functions are called in a nested loop on usually
    # very small arrays (length=n_topics).
    # In that case, finding the appropriate signature of the
    # fused-typed function can be more costly than its execution,
    # hence the dispatch is done outside of the loop.
    ctype = "float" if X.dtype == np.float32 else "double"
    mean_change = cy_mean_change[ctype]
    dirichlet_expectation_1d = cy_dirichlet_expectation_1d[ctype]
    eps = np.finfo(X.dtype).eps

    for idx_d in range(n_samples):
        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d] : X_indptr[idx_d + 1]] # col index for row idx_d
            cnts = X_data[X_indptr[idx_d] : X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]

        gamma_d = gamma[idx_d, :]
        # The next one is a copy, since the inner loop overwrites it.
        exp_elog_theta_d = exp_elog_theta[idx_d, :].copy()
        exp_elog_beta_d = exp_elog_beta[:, ids]

        # Iterate between `gamma_d` and `norm_phi` until convergence
        for _ in range(0, max_doc_update_iter):
            last_d = gamma_d

            # The optimal phi_{dwk} is proportional to
            # exp(E[log(theta_{dk})]) * exp(E[log(beta_{dw})]).
            norm_phi = np.dot(exp_elog_theta_d, exp_elog_beta_d) + eps

            gamma_d = exp_elog_theta_d * np.dot(cnts / norm_phi, exp_elog_beta_d.T)
            # Note: adds doc_topic_prior to gamma_d, in-place.
            dirichlet_expectation_1d(gamma_d, doc_topic_prior, exp_elog_theta_d)

            if mean_change(last_d, gamma_d) < mean_change_tol:
                break
        gamma[idx_d, :] = gamma_d

        # Contribution of document d to the expected sufficient
        # statistics for the M step.
        if cal_sstats:
            norm_phi = np.dot(exp_elog_theta_d, exp_elog_beta_d) + eps
            suff_stats[:, ids] += np.outer(exp_elog_theta_d, cnts / norm_phi)

    return (gamma, suff_stats)


class LDA(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis:
        A classifier with a linear decision boundary, generated by fitting
        class conditional densities to the data and using Bayes' rule.
    """
    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 0, None, closed="neither")],
        "doc_topic_prior": [None, Interval(Real, 0, 1, closed="both")],
        "topic_word_prior": [None, Interval(Real, 0, 1, closed="both")],
        "learning_method": [StrOptions({"batch", "online"})],
        "learning_decay": [Interval(Real, 0, 1, closed="both")],
        "learning_offset": [Interval(Real, 1.0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "batch_size": [Interval(Integral, 0, None, closed="neither")],
        "evaluate_every": [Interval(Integral, None, None, closed="neither")],
        "total_samples": [Interval(Real, 0, None, closed="neither")],
        "perp_tol": [Interval(Real, 0, None, closed="left")],
        "mean_change_tol": [Interval(Real, 0, None, closed="left")],
        "max_doc_update_iter": [Interval(Integral, 0, None, closed="left")],
        "n_jobs": [None, Integral],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
        "intercept": [None, "array-like"],
        "penalty": [None, Interval(Real, 0, 1, closed="left")],
        "debug" : [None, Integral],
    }

    def __init__(
        self,
        n_components=10,
        *,
        doc_topic_prior=None,
        topic_word_prior=None,
        learning_method="batch",
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=10,
        batch_size=128,
        evaluate_every=-1,
        total_samples=1e6,
        perp_tol=1e-1,
        mean_change_tol=1e-3,
        max_doc_update_iter=100,
        n_jobs=None,
        verbose=0,
        random_state=None,
        intercept=None,
        penalty=0,
        penalize_ambian_only = False,
        debug = 0,
    ):
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = total_samples
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.intercept = intercept
        self.penalty = penalty
        self.penalize_ambian_only = penalize_ambian_only
        self.debug = debug

    def _init_latent_vars(self, n_features, dtype=np.float64, lambda_=None):
        """Initialize latent variables."""

        self.random_state_ = check_random_state(self.random_state)
        self.n_batch_iter_ = 1
        self.n_iter_ = 0
        self.n_features = n_features

        if self.intercept is not None:
            self.intercept_ = np.asarray(self.intercept).reshape(-1)
            assert len(self.intercept_) == self.n_features
            self.intercept_ /= self.intercept_.sum()
            self.intercept = True
        else:
            self.intercept = False

        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = 1.0 / self.n_components
        else:
            self.doc_topic_prior_ = self.doc_topic_prior

        if self.topic_word_prior is None:
            self.topic_word_prior_ = 1.0 / self.n_components
        else:
            self.topic_word_prior_ = self.topic_word_prior

        init_gamma = 100.0
        init_var = 1.0 / init_gamma
        # In the literature, this is called `lambda`
        if lambda_ is None:
            self.components_ = self.random_state_.gamma(
                init_gamma, init_var, (self.n_components, n_features)
            ).astype(dtype, copy=False)
        else:
            assert lambda_.shape == (self.n_components, n_features)
            self.components_ = lambda_.astype(dtype)

        # In the literature, this is `exp(E[log(beta)])`
        self.exp_elog_beta = np.exp(
            _dirichlet_expectation_2d(self.components_)
        )
        if self.intercept:
            self.exp_elog_beta = np.vstack(
                (self.exp_elog_beta, self.intercept_)
            )

    def _e_step(self, X, cal_sstats, random_init, parallel=None):
        """E-step in EM update.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        cal_sstats : bool
            Parameter that indicate whether to calculate sufficient statistics
            or not. Set ``cal_sstats`` to True when we need to run M-step.

        random_init : bool
            Parameter that indicate whether to initialize document topic
            distribution randomly in the E-step. Set it to True in training
            steps.

        parallel : joblib.Parallel, default=None
            Pre-initialized instance of joblib.Parallel.

        Returns
        -------
        (gamma, suff_stats) :
            `gamma` is unnormalized topic distribution for each
            document. In the literature, this is called `gamma`.
            `suff_stats` is expected sufficient statistics for the M-step.
            When `cal_sstats == False`, it will be None.

        """

        # Run e-step in parallel
        random_state = self.random_state_ if random_init else None

        n_jobs = effective_n_jobs(self.n_jobs)
        if parallel is None:
            parallel = Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1))
        results = parallel(
            delayed(_update_doc_distribution)(
                X[idx_slice, :],
                self.exp_elog_beta,
                self.doc_topic_prior_,
                self.max_doc_update_iter,
                self.mean_change_tol,
                cal_sstats,
                random_state,
            )
            for idx_slice in gen_even_slices(X.shape[0], n_jobs)
        )

        # merge result
        doc_topics, sstats_list = zip(*results)
        gamma = np.vstack(doc_topics)

        if cal_sstats:
            # This step finishes computing the sufficient statistics for the
            # M-step.
            suff_stats = np.zeros(self.exp_elog_beta.shape, dtype=self.exp_elog_beta.dtype)
            for sstats in sstats_list:
                suff_stats += sstats
            suff_stats *= self.exp_elog_beta
        else:
            suff_stats = None

        return (gamma, suff_stats)

    def _em_step(self, X, total_samples, batch_update, parallel=None):
        """EM update for 1 iteration.

        update `_component` by batch VB or online VB.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        total_samples : int
            Total number of documents. It is only used when
            batch_update is `False`.

        batch_update : bool
            Parameter that controls updating method.
            `True` for batch learning, `False` for online learning.

        parallel : joblib.Parallel, default=None
            Pre-initialized instance of joblib.Parallel

        Returns
        -------
        gamma : ndarray of shape (n_samples, n_components)
            Unnormalized document topic distribution.
        """

        # E-step
        _, suff_stats = self._e_step(
            X, cal_sstats=True, random_init=True, parallel=parallel
        )
        if self.intercept:
            suff_stats = suff_stats[:-1, :]

        # M-step
        if batch_update:
            self.components_ = self.topic_word_prior_ + suff_stats
        else:
            # online update
            # In the literature, the weight is `rho`
            weight = np.power(
                self.learning_offset + self.n_batch_iter_, -self.learning_decay
            )
            doc_ratio = float(total_samples) / X.shape[0]
            if self.penalty > 0 and self.n_batch_iter_ > 3:
                korder = np.arange(self.n_components)
                self.random_state_.shuffle(korder)
                w = suff_stats.sum(axis = 1) * self.penalty
                if self.intercept and self.penalize_ambian_only:
                    for k in korder:
                        suff_stats[k, :] = np.clip((suff_stats[k, :] - self.intercept_ * w[k]) / (1-self.penalty), EPS, None)
                else:
                    w0 = self.components_.sum(axis = 1)
                    w0 /= w0.sum()
                    beta0 = normalize(self.components_, axis = 1, norm = 'l1')
                    if self.intercept:
                        w0 = np.concatenate((w0, [1]))
                        w0 /= w0.sum()
                        beta0 = np.vstack((beta0, self.intercept_))
                    for k in korder:
                        lam0 = copy.copy(suff_stats[k, :])
                        for l in range(beta0.shape[0]):
                            if l == k:
                                continue
                            suff_stats[k, :] -= beta0[l, :] * w[k] * w0[l]
                        suff_stats[k, :] = np.clip(suff_stats[k, :] / (1-self.penalty), EPS, None)
                        delta = np.abs(suff_stats[k, :] - lam0).sum()
                        if self.debug:
                            print(f"{k}, {delta * self.penalty / w[k]:.3f}")
            self.components_ *= 1 - weight
            self.components_ += weight * (
                self.topic_word_prior_ + doc_ratio * suff_stats
            )

        # update `component_` related variables
        self.exp_elog_beta = np.exp(
            _dirichlet_expectation_2d(self.components_)
        )
        if self.intercept:
            self.exp_elog_beta = np.vstack(\
                (self.exp_elog_beta, self.intercept_))
        self.n_batch_iter_ += 1
        return

    def _more_tags(self):
        return {
            "preserves_dtype": [np.float64, np.float32],
            "requires_positive_X": True,
        }

    def _check_non_neg_array(self, X, reset_n_features, whom):
        """check X format

        check X format and make sure no negative value in X.

        Parameters
        ----------
        X :  array-like or sparse matrix

        """
        dtype = [np.float64, np.float32] if reset_n_features else self.components_.dtype

        X = self._validate_data( # from sklearn.base.BaseEstimator
            X,
            reset=reset_n_features,
            accept_sparse="csr",
            dtype=dtype,
        )
        check_non_negative(X, whom)

        return X

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        """Online VB with Mini-Batch update.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Partially fitted estimator.
        """
        first_time = not hasattr(self, "components_")

        X = self._check_non_neg_array(
            X, reset_n_features=first_time, whom="LatentDirichletAllocation.partial_fit"
        )
        n_samples, n_features = X.shape
        batch_size = self.batch_size

        # initialize parameters or check
        if first_time:
            self._init_latent_vars(n_features, dtype=X.dtype)

        if n_features != self.components_.shape[1]:
            raise ValueError(
                "The provided data has %d dimensions while the model was trained with feature size %d."
                % (n_features, self.components_.shape[1])
            )

        n_jobs = effective_n_jobs(self.n_jobs)
        idx_randomize = np.arange(n_samples)
        self.random_state_.shuffle(idx_randomize)
        with Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1)) as parallel:
            for idx_slice in gen_even_slices_from_list(idx_randomize, n_samples // batch_size):
                self._em_step(
                    X[idx_slice, :],
                    total_samples=self.total_samples,
                    batch_update=False,
                    parallel=parallel,
                )

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Learn model for the data X with variational Bayes method.

        When `learning_method` is 'online', use mini-batch update.
        Otherwise, use batch update.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """
        X = self._check_non_neg_array(
            X, reset_n_features=True, whom="LatentDirichletAllocation.fit"
        )
        n_samples, n_features = X.shape
        max_iter = self.max_iter
        evaluate_every = self.evaluate_every
        learning_method = self.learning_method

        batch_size = self.batch_size

        # initialize parameters
        self._init_latent_vars(n_features, dtype=X.dtype)
        # change to perplexity later
        last_bound = None
        n_jobs = effective_n_jobs(self.n_jobs)
        with Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1)) as parallel:
            for i in range(max_iter):
                idx_randomize = np.arange(n_samples)
                self.random_state_.shuffle(idx_randomize)
                if learning_method == "online":
                    n_batch = n_samples // batch_size
                    for idx_slice in gen_even_slices_from_list(idx_randomize, n_batch):
                        self._em_step(
                            X[idx_slice, :],
                            total_samples=n_samples,
                            batch_update=False,
                            parallel=parallel,
                        )
                else:
                    # batch update
                    self._em_step(
                        X, total_samples=n_samples, batch_update=True, parallel=parallel
                    )

                # check perplexity
                if evaluate_every > 0 and (i + 1) % evaluate_every == 0:
                    doc_topics_distr, _ = self._e_step(
                        X, cal_sstats=False, random_init=False, parallel=parallel
                    )
                    bound, ll = self._perplexity_precomp_distr(
                        X, doc_topics_distr, sub_sampling=False
                    )
                    if self.verbose:
                        print(
                            "iteration: %d of max_iter: %d, perplexity: %.4f"
                            % (i + 1, max_iter, bound)
                        )
                    if last_bound and abs(last_bound - bound) < self.perp_tol:
                        break
                    last_bound = bound
                elif self.verbose:
                    print("iteration: %d of max_iter: %d" % (i + 1, max_iter))
                self.n_iter_ += 1

        # calculate final perplexity value on train set
        doc_topics_distr, _ = self._e_step(
            X, cal_sstats=False, random_init=False, parallel=parallel
        )
        self.bound_, self.ll_ = self._perplexity_precomp_distr(
            X, doc_topics_distr, sub_sampling=False
        )

        return self

    def _unnormalized_transform(self, X):
        """Transform data X according to fitted model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        Returns
        -------
        gamma : ndarray of shape (n_samples, n_components)
            Document topic distribution for X.
        """
        gamma, _ = self._e_step(X, cal_sstats=False, random_init=False)

        return gamma

    def transform(self, X):
        """Transform data X according to the fitted model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        Returns
        -------
        gamma : ndarray of shape (n_samples, n_components)
            Document topic distribution for X.
        """
        check_is_fitted(self)
        X = self._check_non_neg_array(
            X, reset_n_features=False, whom="LatentDirichletAllocation.transform"
        )
        gamma = self._unnormalized_transform(X)
        gamma /= gamma.sum(axis=1)[:, np.newaxis]
        return gamma

    def _approx_bound(self, X, gamma, sub_sampling):
        """Estimate the variational bound.

        Estimate the variational bound over "all documents" using only the
        documents passed in as X. Since log-likelihood of each word cannot
        be computed directly, we use this bound to estimate it.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        gamma : ndarray of shape (n_samples, n_components)
            Document topic distribution. In the literature, this is called
            gamma.

        sub_sampling : bool, default=False
            Compensate for subsampling of documents.
            It is used in calculate bound in online learning.

        Returns
        -------
        score : float

        """

        def _loglikelihood(prior, distr, dirichlet_distr):
            # calculate log-likelihood
            size = distr.shape[1]
            score = np.sum((prior - distr) * dirichlet_distr)
            score += np.sum(gammaln(distr) - gammaln(prior))
            score += np.sum(gammaln(prior * size) - gammaln(np.sum(distr, 1)))
            return score

        is_sparse_x = sp.issparse(X)
        n_samples, n_components = gamma.shape
        n_features = self.components_.shape[1]
        score = 0

        Elog_theta = _dirichlet_expectation_2d(gamma)
        Elog_beta = _dirichlet_expectation_2d(self.components_)
        if self.intercept:
            Elog_beta = np.vstack((Elog_beta, np.log(self.intercept_)))

        if is_sparse_x:
            X_data = X.data
            X_indices = X.indices
            X_indptr = X.indptr

        # E[log p(docs | theta, beta)]
        for idx_d in range(0, n_samples):
            if is_sparse_x:
                ids = X_indices[X_indptr[idx_d] : X_indptr[idx_d + 1]]
                cnts = X_data[X_indptr[idx_d] : X_indptr[idx_d + 1]]
            else:
                ids = np.nonzero(X[idx_d, :])[0]
                cnts = X[idx_d, ids]
            temp = (
                Elog_theta[idx_d, :, np.newaxis] + Elog_beta[:, ids]
            )
            norm_phi = logsumexp(temp, axis=0)
            score += np.dot(cnts, norm_phi)
        elogl = score

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        score += _loglikelihood(self.doc_topic_prior_, gamma, Elog_theta)

        # Compensate for the subsampling of the population of documents
        if sub_sampling:
            doc_ratio = float(self.total_samples) / n_samples
            score *= doc_ratio

        # E[log p(beta | eta) - log q (beta | lambda)]
        if self.intercept:
            scale = self.components_.sum()
            lambda_ = np.vstack((self.components_, self.intercept_ * scale))
            score += _loglikelihood(self.topic_word_prior_, lambda_, Elog_beta)
        else:
            score += _loglikelihood(self.topic_word_prior_, self.components_, Elog_beta)

        return score, elogl

    def score(self, X, y=None):
        """Calculate approximate log-likelihood as score.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        score : float
            Use approximate bound as score.
        """
        check_is_fitted(self)
        X = self._check_non_neg_array(
            X, reset_n_features=False, whom="LatentDirichletAllocation.score"
        )

        gamma = self._unnormalized_transform(X)
        score, ll = self._approx_bound(X, gamma, sub_sampling=False)
        return score, ll

    def _perplexity_precomp_distr(self, X, gamma=None, sub_sampling=False):
        """Calculate approximate perplexity for data X with ability to accept
        precomputed gamma

        Perplexity is defined as exp(-1. * log-likelihood per word)

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        gamma : ndarray of shape (n_samples, n_components), \
                default=None
            Document topic distribution.
            If it is None, it will be generated by applying transform on X.

        Returns
        -------
        score : float
            Perplexity score.
        """
        if gamma is None:
            gamma = self._unnormalized_transform(X)
        else:
            n_samples, n_components = gamma.shape
            if n_samples != X.shape[0]:
                raise ValueError(
                    "Number of samples in X and gamma do not match."
                )

            if n_components != self.n_components:
                raise ValueError("Number of topics does not match.")

        current_samples = X.shape[0]
        bound, ll = self._approx_bound(X, gamma, sub_sampling)

        if sub_sampling:
            word_cnt = X.sum() * (float(self.total_samples) / current_samples)
        else:
            word_cnt = X.sum()
        perword_bound = bound / word_cnt

        return np.exp(-1.0 * perword_bound), ll/word_cnt

    def perplexity(self, X, sub_sampling=False):
        """Calculate approximate perplexity for data X.

        Perplexity is defined as exp(-1. * log-likelihood per word)

        .. versionchanged:: 0.19
           *gamma* argument has been deprecated and is ignored
           because user no longer has access to unnormalized distribution

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        sub_sampling : bool
            Do sub-sampling or not.

        Returns
        -------
        score : float
            Perplexity score.
        """
        check_is_fitted(self)
        X = self._check_non_neg_array(
            X, reset_n_features=True, whom="LatentDirichletAllocation.perplexity"
        )
        return self._perplexity_precomp_distr(X, sub_sampling=sub_sampling)

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.exp_elog_beta.shape[0]
