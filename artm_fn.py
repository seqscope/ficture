import sys, io, os, copy, time
import numpy as np
import pandas as pd
import sklearn
from random import shuffle

class ARTM:
    '''
    '''
    def __init__(self, K, tau = None, ker_thres=-1,\
                 nB=1, nS=None, B=None, S=None):
        self.K = K
        self.nS = nS
        self.nB = nB
        self.T = set(range(self.K))
        self.tau = {'smooth_phi':0, 'sparse_phi':0, 'smooth_theta':0, 'sparse_theta':0, 'decorr_phi':0}
        if tau is not None:
            self.tau.update(tau)
        self.kernel_thres = ker_thres
        if S is None and nS is not None and nS > 0:
            self.S = set(range(nS))
        elif S is not None:
            self.S = set([int(i) for i in S if i >= 0])
        else:
            self.S = set()
        self.nS = len(self.S)
        if B is None and nB is not None and nB > 0:
            self.B = set(range(nS, (nB+nS)))
        elif B is not None:
            self.B = set([int(i) for i in B if i >= 0])
        else:
            self.B = set()
        self.nB = len(self.B)
        assert len(self.B - self.T) == 0 and len(self.S - self.T) == 0, "Illegal input B and/or S set"
        assert len(self.B.intersection(self.S)) == 0, "Input B and S cannot intersect"
        self.topic_class = {'S':self.S, 'B':self.B, 'Other':self.T-self.S-self.B}
        self.mask_s = np.array([1 if t in self.topic_class['S'] else 0 for t in range(self.K)]).reshape((-1, 1))
        self.mask_b = np.array([1 if t in self.topic_class['B'] else 0 for t in range(self.K)]).reshape((-1, 1))
        self.verbose = False
        self.verbose_inner = False
        self.init_model = False
        self.init_data = False

    def initialize_model(self, M = None, vocab = None, vocab_freq = None,\
                         phi = None, tol_theta_max = 1e-4, tol_perp_rel = 1e-4,\
                         kappa = 0.7, rho_offset = 10,\
                         max_iter = 20, min_iter = 5,\
                         kernel_thres = 0.25, verbose = False):
        assert M is not None or vocab is not None, "Please input vocabulary list or its size"
        self.M = M
        if phi is not None:
            self.M = phi.shape[1]
            assert phi.shape[0] == self.K, "Inconsistent dimension of phi with model size"
        self.vocab = vocab
        if vocab is None:
            self.vocab = list(range(self.M))
        self.fw = vocab_freq
        self.phi = phi
        self.ntw = 0
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.verbose = verbose
        self.tol_theta_max = tol_theta_max
        self.tol_perp_rel = tol_perp_rel
        self.rho_offset = rho_offset
        self.kappa = kappa
        self.kernel_thres = kernel_thres
        self.ct_minibatch = 0
        self.kl_ref = {'phi':self.fw, 'theta':1./self.K}
        score_names = "perplexity,background_ratio,purity,contrast,kernel_size".split(',')
        self.score_tracker = {x:[] for x in score_names}
        self.init_model = True


    def _fit_batch(self, ndw, theta):

        N = ndw.sum()
        n, M = ndw.shape
        pdw = np.einsum('tw,dt->dw', self.phi, theta) # \hat P(w|d), n x M
        logl = (ndw[pdw>0] * np.log(pdw[pdw>0])).sum()
        perplexity = np.exp(-logl/ N)
        it = 0
        theta_org = 0
        while it < self.max_iter:

            pre_perp = perplexity
            ### E step P(t|d,w), fit local parameter
            pdwt = np.einsum('tw,dt->dwt', self.phi, theta) # P(t|d,w) n x M x K
            pdw = pdwt.sum(axis = 2)
            pdw[pdw == 0] = 1
            pdwt = pdwt / pdw[:, :, np.newaxis]
            ndt = np.einsum('dw,dwt->dt', ndw, pdwt) # P(t|d) n x K
            ### M step, local parameter
            theta = ndt + self.tau['smooth_theta'] * (self.kl_ref['theta']*self.mask_b).T + self.tau['sparse_theta'] * (self.kl_ref['theta'] * self.mask_s).T
            theta = np.clip(theta, 0, np.inf)
            theta = theta / theta.sum(axis = 1).reshape((-1, 1))

            # Loglikelihood
            pdw = np.einsum('tw,dt->dw', self.phi, theta) # \hat P(w|d) n x M
            logl = (ndw[pdw>0] * np.log(pdw[pdw>0])).sum()
            # Perplexity score
            perplexity = np.exp(-logl/ N)
            delta_perp = np.abs(perplexity - pre_perp)/pre_perp
            if self.verbose_inner and it % self.verbose_inner == 0:
                v = theta.max(axis = 1)
                print(f"{self.ct_minibatch}-{it}. {perplexity:.2e}, {np.min(v):.2f}, {np.median(v):.2f}, {np.max(v):.2f}")
            it += 1
            if it == 1:
                perp0 = perplexity
            if it == self.max_iter:
                perp1 = perplexity
            if it > self.min_iter and (np.abs(theta_org - theta).max() < self.tol_theta_max or delta_perp < self.tol_perp_rel):
                perp1 = perplexity
                break
            theta_org = theta

        # Background ratio
        if self.nB > 0:
            br = ((ndw[:,:,np.newaxis] * pdwt).sum(axis = (0,1)).reshape((-1, 1)) * self.mask_b).sum() / N
            self.score_tracker['background_ratio'].append(br)
        if self.verbose and self.ct_minibatch % self.verbose == 0:
            print(f"{self.ct_minibatch}-th minibatch. {perp0:.2e} -> {perp1:.2e}")
        ntw_tilde = np.einsum('dw,dwt->tw', ndw, pdwt)
        return ntw_tilde, theta

    def fit_stochastic(self, mtx, theta=None, batch_size=256, verbose_inner=False):
        assert self.init_model, "Please run initialize_model before fitting data"
        self.b_size = batch_size
        shuffle_indx = np.arange(mtx.shape[0])
        shuffle(shuffle_indx)
        st = 0
        self.theta = theta
        if theta is None:
            self.theta = np.clip(np.abs(np.random.normal(0, 1, (n, self.T))*np.sqrt(np.pi/2/self.K)), 0.05, 0.8)
            self.theta = self.theta / self.theta.sum(axis = 1).reshape((-1, 1))
        while st < mtx.shape[0]:
            ed = min([st + self.b_size, mtx.shape[0]])
            subset_indx = np.arange(st, ed)
            ndw = mtx[subset_indx, :].toarray()
            N = ndw.sum()
            n, M = ndw.shape
            ### Local parameter and sufficient statistics
            ntw_tilde, theta = self._fit_batch(ndw, self.theta[subset_indx, :])
            ### M-step, global parameter
            if self.ct_minibatch == 0:
                self.ntw = ntw_tilde
            else:
                rhot = (self.rho_offset + self.ct_minibatch)**(-self.kappa)
                self.ntw = (1. - rhot) * self.ntw + rhot * ntw_tilde # K x M
            self.phi = self.ntw + self.tau['smooth_phi'] * self.kl_ref['phi'] * self.mask_b + self.tau['sparse_phi'] * self.kl_ref['phi'] * self.mask_s
            if self.tau['decorr_phi'] != 0:
                self.phi += self.tau['decorr_phi'] * (self.mask_s * np.einsum('tw,s,sw->tw',self.phi, self.mask_s.reshape(-1), self.phi))
            self.phi = np.clip(self.phi, 0, np.inf)
            rsum = self.phi.sum(axis = 1)
            indx = rsum > 0
            self.phi[indx, :] = self.phi[indx, :] / rsum[indx].reshape((-1, 1))
            # self.phi = self.phi / self.phi.sum(axis = 1).reshape((-1, 1))
            ### Evaluation scores
            pt = theta.sum(axis = 0) + 1e-4
            pt = (pt / pt.sum()).reshape((-1, 1)) # P(t) K x 1
            pwt = self.phi * pt # P(t|w) K x M
            pw = pwt.sum(axis = 0)
            indx = pw != 0
            pwt[:, indx] = pwt[:, indx] / pw[indx].reshape((1, -1))
            # Kernel size
            maskWt = pwt > self.kernel_thres
            kersize = maskWt.sum(axis = 1)
            for k in range(self.K):
                maskWt[k, np.argmax(pwt[k, :])] = 1
            # Contrast
            contrast = (pwt * maskWt).sum(axis = 1) / maskWt.sum(axis = 1)
            # Purity
            purity = (self.phi * maskWt).sum(axis = 1)
            # Perplexity score
            pdw = np.einsum('tw,dt->dw', self.phi, theta) # \hat P(w|d) n x M
            logl = (ndw[pdw>0] * np.log(pdw[pdw>0])).sum()
            perplexity = np.exp(-logl/ N)
            self.score_tracker['perplexity'].append(perplexity)
            self.score_tracker['purity'].append(purity)
            self.score_tracker['contrast'].append(contrast)
            self.score_tracker['kernel_size'].append(kersize)
            if self.verbose and self.ct_minibatch % self.verbose == 0:
                print( list(kersize), '\t', list(np.around(purity, 2)) )
                print((self.phi > 0).sum(axis = 1))
            self.ct_minibatch += 1
            st += self.b_size

    def fit(mtx, theta, stochastic = True, batch_size = 256, clear = False, max_iter = None, min_iter = None, verbose = None, phi = None):
        # initialize model based on data if not initialized
        assert mtx.shape[1] == self.M, "Input does not match vocabulary"
        self.n = mtx.shape[0]
        if self.fw is None:
            self.fw = np.asarray(mtx.sum(axis = 0)).reshape(-1)
            self.fw = self.fw / self.fw.sum()
        if verbose is not None:
            self.verbose = verbose
        if max_iter is not None:
            self.max_iter = max_iter
        if min_iter is not None:
            self.min_iter = min_iter
        if clear:
            self.phi = phi
        if not clear and phi is not None:
            print("Warning: if clear=False, input phi is ignored")
        if self.phi is None:
            self.phi = np.abs(np.random.normal(0, 1, size=(self.K, self.M))) *\
                        (fw*0.1).reshape((1, -1)) + fw.reshape((1, -1))
            self.phi = np.clip(self.phi, 0, 1)
            self.phi = self.phi / self.phi.sum(axis = 1).reshape((-1, 1))
        if stochastic:
            self.fit_stochastic(self, mtx, theta, batch_size)
        else:
            self.fit_stochastic(self, mtx, theta, batch_size=mtx.shape[0])
