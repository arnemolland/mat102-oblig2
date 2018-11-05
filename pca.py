# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:16:30 2016

@author: jev
"""

from __future__ import division
import numpy as np
import scipy.linalg as sl

# Principal component analysis
# X = np.array-type, meanCentered, a = antall komponenter
# Kall: [scores, loadings, error] = pca(X)
def pca(X, a = 3, tol = 0.0001):
    X = np.array(X)
    E = X
    [n,m] = X.shape
    T = np.zeros([n,a])
    P = np.zeros([m,a])
    for i in range(a):
        t_new = np.random.rand(n)
        for iter in range(1000):
            t = t_new
            p = np.dot(E.T,t)
            p = p/sl.norm(p)
            t_new = np.dot(E,p)
            if sl.norm(t-t_new) <= tol:
                break
        else:
            raise Exception("Something went wrong")
        T[:,i] = t
        P[:,i] = p
        E = E - np.dot(t.reshape([n,1]),p.reshape([1,m]))
    return [T,P,E] # X = T*P.T + E

# For å sikre at søylene i X har middel = 0
# Typisk kall: X = meanCenter(X)
def meanCenter(X):
    X = np.array(X)
    return X - np.mean(X, axis = 0)

# For å sikre at søylene i X har standardavvik 1
# Typisk kall: X = standardize(X)
def standardize(X):
    # X allerede meanCentered
    stdX = np.std(X, axis=0,ddof=1)
    stdX[stdX == 0] = 1
    return X/stdX
