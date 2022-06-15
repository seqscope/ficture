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
    def __init__(self, sparse_s=50, kappa=1.0001, c=0.9999**2, mut0=1, nu=1/2, max_iter = 50, stop_abs = 1e-4, stop_rel = 1e-4, res_flat_max= 5):
        for k,v in vars().items():
            if k != 'self':
                setattr(self,k,v)

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
        self.X = X
        self.Y = Y
        assert self.Y.shape[0] == self.X.shape[0], "Incompatible dimensions"
        self.Y = sklearn.preprocessing.normalize(self.Y, norm='l1', axis = 1)
        self.Xf = np.linalg.norm(self.X, ord='fro')
        self.N, self.M = self.X.shape
        self.K = self.Y.shape[1]
        self.Bc = np.asarray(self.X.mean(axis = 0)).reshape((1, -1))
        if self.M < self.sparse_s:
            self.sparse_s = self.M
        self.log = []
        self.features = feature_list
        # Initialize loading matrix
        if B is None:
            self.B = np.zeros((self.K, self.M))
            for m in range(self.M):
                self.B[:, m], _ = scipy.optimize.nnls(self.Y, (self.X[:,m]-self.Bc[0,m]).reshape(-1))
            self.B = np.clip(self.B, 0, np.inf)
        else:
            self.B = B
        for k in range(self.K):
            indx = heapq.nlargest(self.sparse_s, range(self.M), key=lambda x : self.B[k, x])
            mask = np.zeros(self.M, dtype=bool)
            mask[indx] = 1
            self.B[k, ~mask] = 0

    def fit(self, verbose = False):
        """
        Run SNMF
        """
        # First iteration
        t0 = time.time()
        t = 1
        B1 = self.B + self.Bc
        res0 = np.linalg.norm(self.X - self.Y @ B1, ord='fro') / self.Xf
        Y0 = copy.copy(self.Y)
        B0 = copy.copy(self.B)
        betaBt = 0
        betaYt = 0
        deltaY = 0
        deltaB = 0

        BBt = B1 @ B1.T
        YtY = self.Y.T @ self.Y
        mut = 0.5 * (1+np.sqrt(1+4*self.mut0**2))

        Ly = np.linalg.norm(YtY, ord='fro')
        Lb = np.linalg.norm(BBt, ord='fro')
        Ytilde = self.Y+betaBt*deltaY
        Btilde = B1+betaYt*deltaB
        nablaYf = Ytilde @ BBt - self.X @ B1.T
        nablaBf = YtY @ Btilde - self.Y.T @ self.X

        self.Y = np.clip(Ytilde - 1./Lb * nablaYf, 0, np.inf)
        self.B = Btilde-self.Bc - 1./self.kappa/Ly * nablaBf
        self.B = np.clip(self.B, 0, np.inf)
        for k in range(self.K):
            indx = heapq.nlargest(self.sparse_s, range(self.M), key=lambda x : self.B[k, x])
            mask = np.zeros(self.M, dtype=bool)
            mask[indx] = 1
            self.B[k, ~mask] = 0
        B1 = self.B + self.Bc
        res = np.linalg.norm(self.X - self.Y @ B1, ord='fro') / self.Xf

        deltaB = self.B - B0
        deltaY = self.Y - Y0
        deltab = np.linalg.norm(deltaB, ord='fro')
        deltay = np.linalg.norm(deltaY, ord='fro')
        deltabi = (deltaB!=0).sum()
        nabbmax = np.abs(nablaBf).max()
        nabymax = np.abs(nablaYf).max()
        self.log.append([t, res, deltab, deltay, deltabi, nabbmax, nabymax])

        t0 = time.time() - t0
        if verbose:
            print(f"{t}-st iteration, relative residual (F norm) change {res0:.3f}->{res:.3f}. Maximum derivative {nabbmax:.1e}, {nabymax:.1e}")

        res_flat_ct = 0
        t0 = time.time()
        while t < self.max_iter:
            t += 1
            res0 = res
            mut0 = mut
            Y0 = copy.copy(self.Y)
            B0 = copy.copy(self.B)
            Ly0 = copy.copy(Ly)
            Lb0 = copy.copy(Lb)
            deltab0 = deltab
            deltay0 = deltay

            BBt = B1 @ B1.T
            YtY = self.Y.T @ self.Y
            Ly = np.linalg.norm(YtY, ord='fro')
            Lb = np.linalg.norm(BBt, ord='fro')
            mut = 0.5 * (1+np.sqrt(1+4*mut0**2))
            betaBt = np.min([ (mut0-1)/mut, (self.kappa-1)/self.kappa * np.sqrt((self.c*self.nu*(1-self.nu)*Ly0)/(Ly)) ])
            betaYt = np.min([ (mut0-1)/mut,  np.sqrt((self.c*Lb0)/(Lb)) ])
            Ytilde = self.Y+betaBt*deltaY
            Btilde = B1+betaYt*deltaB
            nablaYf = Ytilde @ BBt - self.X @ B1.T
            nablaBf = YtY @ Btilde - self.Y.T @ self.X

            self.Y = np.clip(Ytilde - 1./Lb * nablaYf, 0, np.inf)
            self.B = Btilde - self.Bc - 1./self.kappa/Ly * nablaBf
            self.B = np.clip(self.B, 0, np.inf)
            for k in range(self.K):
                indx = heapq.nlargest(self.sparse_s, range(self.M), key=lambda x : self.B[k, x])
                mask = np.zeros(self.M, dtype=bool)
                mask[indx] = 1
                self.B[k, ~mask] = 0
            B1 = self.B + self.Bc
            res = np.linalg.norm(self.X - self.Y @ B1, ord='fro') / self.Xf
            maxres = np.abs(self.X - self.Y @ B1).max()

            deltaB = self.B - B0
            deltaY = self.Y - Y0
            deltab = np.linalg.norm(deltaB, ord='fro')
            deltay = np.linalg.norm(deltaY, ord='fro')
            deltabi = np.abs((self.B!=0)^(B0!=0)).sum()
            nabbmax = np.abs(nablaBf).max()
            nabymax = np.abs(nablaYf).max()

            self.log.append([t, res, deltab, deltay, deltabi, nabbmax, nabymax])

            if np.abs(res - res0)/res0 < self.stop_rel:
                res_flat_ct += 1
                if res_flat_ct > self.res_flat_max:
                    break
            else:
                res_flat_ct = 0

            t1 = time.time() - t0
            if verbose:
                print(f"[{t} {t1/60:.3f} min]. {res_flat_ct}-th flat iterations, {deltabi} l-\infty factor loading elements change.\nRelative total residual {res:.3f}, max residual {maxres:.1e}, maximun derivative {nabbmax:.3f}, {nabymax:.3f}.")
            if nabbmax < self.stop_abs and nabymax < self.stop_abs:
                break
            if maxres < self.stop_abs:
                break

    def feature_enrichment(self):
        B_fold = np.around(self.B / self.Bc, 2)
        rnnz = (self.B != 0).sum(axis = 1)
        cnnz = (self.B != 0).sum(axis = 0)
        df = pd.DataFrame({"NNZ":cnnz, "Beta_0":self.Bc.reshape(-1)})
        # if self.features is not None:
        #     df['feature'] = self.features
        for k in range(self.K):
            df['Beta_'+str(k+1)] = self.B[k, :]
            df['Beta_rel_'+str(k+1)] = B_fold[k, :]
        df['MaxBeta'] = self.B.max(axis = 0)
        df['MaxFoldChange'] = B_fold.max(axis = 0) + 1
        st = np.asarray(-self.B).argsort(axis = 0)
        df['Primary'] = st[0, :]
        df['Secondary'] = st[1, :]
        df.loc[df.NNZ<2, 'Secondary'] = -1
        df.loc[df.NNZ.eq(0), 'Primary'] = -1

        rk = np.zeros((self.K, self.M), dtype=int)
        for k in range(self.K):
            v = np.argsort(-B_fold[k, :])
            for i,x in enumerate(v):
                rk[k, x] = i
        df['RK'] = [rk[st[0, i], i] for i in range(self.M)]
        return df
