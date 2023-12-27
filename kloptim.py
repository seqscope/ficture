import sys, os, re, copy, gzip, time, logging, pickle
import numpy as np
from scipy.sparse import *
from sklearn.preprocessing import normalize
import scipy.optimize

def kl_obj(w, x, H):
    q = np.clip(np.dot(w, H), 1e-10, np.inf)
    return (- x * np.log(q) + q).sum()

def kl_jac(w, x, H):
    q = np.clip(np.dot(w, H), 1e-10, np.inf)
    return ((1 - (x / q).reshape((1, -1))) * H).sum(axis = 1)

def kl_hess(w, x, H):
    q = np.clip(np.dot(w, H), 1e-10, np.inf)
    return ((x / q).reshape((1, -1)) * H) @ H.T

def kl_hessp(w, p, x, H):
    Hp = H.T @ p.reshape((-1, 1))
    q = np.clip(np.dot(w, H), 1e-10, np.inf)
    Hp *= (x / q).reshape((-1, 1))
    return np.dot(H, Hp).squeeze()

def optim_kl(X, W, H, tol = 1e-6, verbose = 0, method = 'SLSQP'):
    N,K = W.shape
    M   = H.shape[1]
    assert X.shape == (N,M)
    result = np.zeros(W.shape)
    objval = np.zeros(N)
    constr = scipy.optimize.Bounds(lb=np.zeros(K), keep_feasible=False)
    if issparse(X):
        for i in range(N):
            res = scipy.optimize.minimize(kl_obj, W[i,:], method=method,
                   jac=kl_jac, bounds = constr, \
                   args = (X[[i], :].toarray().squeeze(), H), options={'ftol': tol, 'disp': False})
            if not res.success:
                print(f"Warning: {i}-th subproblem, optim failed")
                result[i, :] = W[i,:]
                objval[i] = kl_obj(result[i, :], X[[i], :].toarray().squeeze(), H)
                continue
            result[i, :] = res.x
            objval[i] = res.fun
            if verbose and i % 100 == 0:
                print(f"{i}-th subproblem, optim converge {res.success}, obj {res.fun:.3f}")
    else:
        for i in range(N):
            res = scipy.optimize.minimize(kl_obj, W[i,:], method=method,
                   jac=kl_jac, bounds = constr, \
                   args = (X[i, :], H), options={'ftol': tol, 'disp': False})
            if not res.success:
                print(f"Warning: {i}-th subproblem, optim failed")
                result[i, :] = W[i,:]
                objval[i] = kl_obj(result[i, :], X[i, :], H)
                continue
            result[i, :] = res.x
            objval[i] = res.fun
            if verbose and i % 100 == 0:
                print(f"{i}-th subproblem, optim converge {res.success}, obj {res.fun:.3f}")
    return (result, objval)

def optim_kl_hess(X, W, H, tol = 1e-6, verbose = 0, method='trust-ncg'):
    N,K = W.shape
    M   = H.shape[1]
    assert X.shape == (N,M)
    result = np.zeros(W.shape)
    objval = np.zeros(N)
    constr = scipy.optimize.Bounds(lb=np.zeros(K), keep_feasible=False)
    if issparse(X):
        for i in range(N):
            res = scipy.optimize.minimize(kl_obj, W[i,:], method=method,
                   jac=kl_jac, hessp=kl_hessp, bounds = constr, \
                   args = (X[[i], :].toarray().squeeze(), H), options={'gtol': tol, 'disp': False})
            if not res.success:
                print(f"Warning: {i}-th subproblem, optim failed")
                result[i, :] = W[i,:]
                objval[i] = kl_obj(result[i, :], X[[i], :].toarray().squeeze(), H)
                # continue
            result[i, :] = res.x
            objval[i] = res.fun
            if verbose and i % 100 == 0:
                print(f"{i}-th subproblem, optim converge {res.success}, obj {res.fun:.3f}")
    else:
        for i in range(N):
            res = scipy.optimize.minimize(kl_obj, W[i,:], method=method,
                   jac=kl_jac, hessp=kl_hessp, bounds = constr, \
                   args = (X[i, :], H), options={'gtol': tol, 'disp': False})
            if not res.success:
                print(f"Warning: {i}-th subproblem, optim failed")
                result[i, :] = W[i,:]
                objval[i] = kl_obj(result[i, :], X[i, :], H)
                continue
            result[i, :] = res.x
            objval[i] = res.fun
            if verbose and i % 100 == 0:
                print(f"{i}-th subproblem, optim converge {res.success}, obj {res.fun:.3f}")
    return (result, objval)
