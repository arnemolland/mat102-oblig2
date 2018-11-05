# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:02:27 2016

@author: jev
"""

from __future__ import division
import numpy as np
import scipy.linalg as sl

# [a,b] = linearRegression(x,y) computes a and b such that y = ax+b is the least squares
# approximation line to the points (x(i),y(i)).
# x and y must be tables of the same length
 
def linearRegression(x,y):
    M = np.ones([len(x),2])
    M[:,0] = x
    p = sl.solve(np.dot(M.T,M),np.dot(y,M))
    return p
    
# [a,b,c] = quadraticRegression(x,y) computes a, b and c such that y = ax^2+bx+c is the least squares
# approximation conic to the points (x(i),y(i)).
# x and y must be tables of the same length
 
def quadraticRegression(x,y):
    M = np.ones([len(x),3])
    M[:,0] = np.multiply(x,x)
    M[:,1] = x
    p = sl.solve(np.dot(M.T,M),np.dot(y,M))
    return p
    
# [a,b,c,d] = cubicRegression(x,y) computes a, b, c and d such that y = ax^3+bx^2+cx+d is the least squares
# approximation cubic to the points (x(i),y(i)).
# x and y must be tables of the same length

def cubicRegression(x,y):
    M = np.ones([len(x),4])
    M[:,0] = np.multiply(x,np.multiply(x,x))
    M[:,1] = np.multiply(x,x)
    M[:,2] = x
    p = sl.solve(np.dot(M.T,M),np.dot(y,M))
    return p
    
# [a0,a1,b1] = sinusoidRegression(x,y,omega) computes a0 a1 b1 such that y = a0+a1cos(2*pi*x/omega)+bsin(2*pi*x/omega) is the least squares
# approximation sinusoid with period omega to the points (x(i),y(i)).
# x and y must be tables of the same length. omega is a scalar.

def sinusoidRegression(x,y,omega):
    M = np.ones([len(x),3])
    M[:,1] = np.cos(np.multiply(2*np.pi/omega,x))
    M[:,2] = np.sin(np.multiply(2*np.pi/omega,x))
    p = sl.solve(np.dot(M.T,M),np.dot(y,M))
    return p
# [a,b,c] = linearRegression2D(x,y,z) computes a,b and c so that
# z = ax + by +c
# is the best linear approximation to the points with coordinates (x(i),y(i),z(i))
# in the least squares sense.
# The inputs must be tables of the same length.
    
def linearRegression2D(x,y,z):
    M = np.ones([len(x),3])
    M[:,0] = x
    M[:,1] = y
    p = sl.solve(np.dot(M.T,M),np.dot(y,M))
    return p