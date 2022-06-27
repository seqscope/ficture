import sys, io, os, copy, time
import numpy as np
import pandas as pd
import sklearn
import scipy.optimize
import heapq

class SNMF:
    """
    Sparse NMF implemented with MM algorithm and hard thresholding
    X \approx YB, ||B[k, :]||_\infty <= sparse_s \forall k \in [K]
    """
    def __init__(self, sparse_s=50, faster=True, start_acceleration = 10):
        self.sparse_s = sparse_s
        self.faster = faster
        self.start_acceleration = start_acceleration

    def initialize(self, X, Y, B = None, feature_list = None):
        """
        Initialization.
        Require two numpy ndarray
        X: units \times features
        Y: initialized units \times factor probabilities
        Optional
        B: factors \times features, factor loading
        Will be initialized by NNLS
        feature_list: a list of names for input features
        """
        assert Y.shape[0] == X.shape[0], "Incompatible dimensions"
        self.X = X
        self.Y = Y
        self.N, self.M = self.X.shape
        self.Y = sklearn.preprocessing.normalize(self.Y, norm='l1', axis = 1)
        self.Y = np.concatenate((np.ones(self.N).reshape((self.N, 1)), self.Y), axis=1) #$
        self.Xf = np.linalg.norm(self.X, ord='fro')
        self.K = self.Y.shape[1]
        # self.Bc = np.asarray(self.X.mean(axis = 0)).reshape((1, -1))
        if self.M < self.sparse_s:
            self.sparse_s = self.M
        self.log = []
        self.features = feature_list
        self.iter = 0
        self.total_iter = 0
        # Initialize loading matrix
        if B is None:
            self.B = np.zeros((self.K, self.M))
            for m in range(self.M):
                # self.B[:, m], _ = scipy.optimize.nnls(self.Y, (self.X[:,m]-self.Bc[0, m]).reshape(-1))
                self.B[:, m], _ = scipy.optimize.nnls(self.Y, self.X[:, m].reshape(-1))
            self.B = np.clip(self.B, 0, np.inf)
        else:
            self.B = B
        for k in range(1, self.K): #$
            indx = heapq.nlargest(self.sparse_s, range(self.M), key = lambda x : self.B[k, x])
            mask = np.zeros(self.M, dtype=bool)
            mask[indx] = 1
            self.B[k, ~mask] = 0

    def update(self, X, Y, stop_abs = None, stop_rel = None, res_flat_max = None, max_iter = None, verbose = None, faster = False):
        assert X.shape[1] == self.M or Y.shape[1] == self.K-1, "Incompatible with existing dimensions" #$
        assert X.shape[0] == Y.shape[0], "Incompatible dimensions"
        self.X = X
        self.Y = Y
        self.N = self.X.shape[0]
        self.Y = sklearn.preprocessing.normalize(self.Y, norm='l1', axis = 1)
        self.Y = np.concatenate((np.ones(self.N).reshape((self.N, 1)), self.Y), axis=1) #$
        self.iter = 0
        self.mut = self.mut_initial
        for k,v in vars().items():
            if k not in ['self', 'X', 'Y'] and v is not None:
                setattr(self,k,v)
        self.fit_inner()

    def fit(self, kappa=1.0001, c=0.9999**2, mut=1, nu=1/2, stop_abs = 1e-4, stop_rel = 1e-4, res_flat_max = 5, max_iter = 50, start_acceleration = None, faster = None, verbose = False):
        for k,v in vars().items():
            if k != 'self' and v is not None:
                setattr(self,k,v)
        self.mut_initial = self.mut
        self.fit_inner()

    def fit_inner(self):
        """
        Run SNMF
        """
        # for k,v in vars().items():
        #     if v != 'None' and k != 'self':
        #         setattr(self,k,v)

        # First iteration
        t0 = time.time()
        # B1 = self.B + self.Bc
        # res0 = np.linalg.norm(self.X - self.Y @ B1, ord='fro') / self.Xf
        res0 = np.linalg.norm(self.X - self.Y @ self.B, ord='fro') / self.Xf #$
        Y0 = copy.copy(self.Y)
        B0 = copy.copy(self.B)
        betaBt = 0
        betaYt = 0
        deltaY = 0
        deltaB = 0

        # BBt = B1 @ B1.T
        BBt = self.B @ self.B.T #$
        YtY = self.Y.T @ self.Y
        Ly = np.linalg.norm(YtY, ord='fro')
        Lb = np.linalg.norm(BBt, ord='fro')
        # Ytilde = self.Y+betaBt*deltaY
        # Btilde = B1+betaYt*deltaB
        # nablaYf = self.Y @ BBt - self.X @ B1.T
        # nablaBf = YtY @ B1 - self.Y.T @ self.X
        nablaYf = self.Y @ BBt - self.X @ self.B.T #$
        nablaBf = YtY @ self.B - self.Y.T @ self.X #$

        self.Y = np.clip(self.Y - 1./Lb * nablaYf, 0, np.inf)
        self.Y[:, 1:] = sklearn.preprocessing.normalize(self.Y[:, 1:], norm='l1', axis = 1)
        # self.B = np.clip(B1 - 1./self.kappa/Ly * nablaBf - self.Bc, 0, np.inf)
        self.B = np.clip(self.B - 1./self.kappa/Ly * nablaBf, 0, np.inf) #$
        for k in range(1, self.K):
            indx = heapq.nlargest(self.sparse_s, range(self.M), key=lambda x : self.B[k, x])
            mask = np.zeros(self.M, dtype=bool)
            mask[indx] = 1
            self.B[k, ~mask] = 0
        # B1 = self.B + self.Bc
        # res = np.linalg.norm(self.X - self.Y @ B1, ord='fro') / self.Xf
        res = np.linalg.norm(self.X - self.Y @ self.B, ord='fro') / self.Xf #$

        deltaB = self.B - B0
        deltaY = self.Y - Y0
        deltab = np.linalg.norm(deltaB, ord='fro')
        deltay = np.linalg.norm(deltaY, ord='fro')
        deltabi = (deltaB!=0).sum()
        nabbmax = np.abs(nablaBf).max()
        nabymax = np.abs(nablaYf).max()
        self.log.append([self.iter, res, deltab, deltay, deltabi, nabbmax, nabymax])

        t0 = time.time() - t0
        if self.verbose:
            print(f"{self.iter}-st iteration, relative residual (F norm) change {res0:.3f}->{res:.3f}. Maximum derivative {nabbmax:.1e}, {nabymax:.1e}")

        res_flat_ct = 0
        t0 = time.time()
        while self.iter < self.max_iter:
            self.iter += 1
            self.total_iter += 1
            res0 = res
            mut0 = self.mut
            Y0 = copy.copy(self.Y)
            B0 = copy.copy(self.B)
            Ly0 = copy.copy(Ly)
            Lb0 = copy.copy(Lb)

            # BBt = B1 @ B1.T
            BBt = self.B @ self.B.T #$
            YtY = self.Y.T @ self.Y
            Ly = np.linalg.norm(YtY, ord='fro')
            Lb = np.linalg.norm(BBt, ord='fro')

            if self.faster and self.total_iter >= self.start_acceleration:
                self.mut = 0.5 * (1+np.sqrt(1+4*mut0**2))
                betaBt = np.min([ (mut0-1)/self.mut, (self.kappa-1)/self.kappa * np.sqrt((self.c*self.nu*(1-self.nu)*Ly0)/(Ly)) ])
                betaYt = np.min([ (mut0-1)/self.mut,  np.sqrt((self.c*Lb0)/(Lb)) ])
                Ytilde = self.Y+betaBt*deltaY
                # Btilde = B1+betaYt*deltaB
                Btilde = self.B+betaYt*deltaB #$
                # nablaYf = Ytilde @ BBt - self.X @ B1.T
                nablaYf = Ytilde @ BBt - self.X @ self.B.T #$
                nablaBf = YtY @ Btilde - self.Y.T @ self.X
                self.Y = np.clip(Ytilde - 1./Lb * nablaYf, 0, np.inf)
                # self.B = Btilde - self.Bc - 1./self.kappa/Ly * nablaBf
                self.B = Btilde - 1./self.kappa/Ly * nablaBf #$
            else:
                # nablaYf = self.Y @ BBt - self.X @ B1.T
                # nablaBf = YtY @ B1 - self.Y.T @ self.X
                nablaYf = self.Y @ BBt - self.X @ self.B.T #$
                nablaBf = YtY @ self.B - self.Y.T @ self.X #$
                self.Y = np.clip(self.Y - 1./Lb * nablaYf, 0, np.inf)
                self.B = B0 - 1./self.kappa/Ly * nablaBf

            self.B = np.clip(self.B, 0, np.inf)
            self.Y[:, 1:] = sklearn.preprocessing.normalize(self.Y[:, 1:], norm='l1', axis = 1) #$
            for k in range(1, self.K):
                indx = heapq.nlargest(self.sparse_s, range(self.M), key=lambda x : self.B[k, x])
                mask = np.zeros(self.M, dtype=bool)
                mask[indx] = 1
                self.B[k, ~mask] = 0
            # B1 = self.B + self.Bc
            # res = np.linalg.norm(self.X - self.Y @ B1, ord='fro') / self.Xf
            res = np.linalg.norm(self.X - self.Y @ self.B, ord='fro') / self.Xf #$
            # maxres = np.abs(self.X - self.Y @ B1).max()
            maxres = np.abs(self.X - self.Y @ self.B).max() #$

            deltaB = self.B - B0
            deltaY = self.Y - Y0
            deltab = np.linalg.norm(deltaB, ord='fro')
            deltay = np.linalg.norm(deltaY, ord='fro')
            deltabi = np.abs((self.B!=0)^(B0!=0)).sum()
            nabbmax = np.abs(nablaBf).max()
            nabymax = np.abs(nablaYf).max()

            self.log.append([self.iter, res, deltab, deltay, deltabi, nabbmax, nabymax])

            if np.abs(res - res0)/res0 < self.stop_rel:
                res_flat_ct += 1
                if res_flat_ct > self.res_flat_max:
                    break
            else:
                res_flat_ct = 0

            t1 = time.time() - t0
            if self.verbose:
                print(f"[{self.iter} {t1/60:.3f} min]. {res_flat_ct}-th flat iterations, {deltabi} l-\infty factor loading elements change.\nRelative total residual {res:.3f}, max residual {maxres:.1e}, maximun derivative {nabbmax:.3f}, {nabymax:.3f}.")
            if nabbmax < self.stop_abs and nabymax < self.stop_abs:
                break
            if maxres < self.stop_abs:
                break

    def feature_enrichment(self):
        # B_fold = np.around(self.B / self.Bc, 2)
        B_fold = np.around(self.B[1:,:] / self.B[0, :], 2) #$
        rnnz = (self.B[1:, :] != 0).sum(axis = 1) #$
        cnnz = (self.B[1:, :] != 0).sum(axis = 0) #$
        # snmf_res = pd.DataFrame({"NNZ":cnnz, "Beta_0":self.Bc.reshape(-1)})
        snmf_res = pd.DataFrame({"NNZ":cnnz, "Beta_0":self.B[0, :].reshape(-1)}) #$
        if self.features is not None:
            snmf_res['feature'] = self.features
        for k in range(1, self.K): #$
            snmf_res['Beta_'+str(k)] = self.B[k, :] #$
            snmf_res['Beta_rel_'+str(k)] = B_fold[k, :] #$
        # snmf_res['MaxBeta'] = self.B.max(axis = 0)
        snmf_res['MaxBeta'] = self.B[1:, :].max(axis = 0)
        snmf_res['MaxFoldChange'] = B_fold.max(axis = 0) + 1
        # st = np.asarray(-self.B).argsort(axis = 0)
        st = np.asarray(-self.B[1:, :]).argsort(axis = 0)
        snmf_res['Primary'] = st[0, :]
        snmf_res['Secondary'] = st[1, :]
        snmf_res.loc[snmf_res.NNZ<2, 'Secondary'] = -1
        snmf_res.loc[snmf_res.NNZ.eq(0), 'Primary'] = -1

        rk = np.zeros((self.K-1, self.M), dtype=int) #$
        for k in range(1, self.K): #$
            v = np.argsort(-B_fold[k, :])
            for i,x in enumerate(v):
                rk[k, x] = i
        snmf_res['RK'] = [rk[st[0, i], i] for i in range(self.M)]
        return snmf_res
