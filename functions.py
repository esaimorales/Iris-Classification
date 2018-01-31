import numpy as np
import math

# returns mean (mu) of matrix
def get_mu(matrix):
    mu = np.mean(np.matrix(matrix), axis=0)
    # print mu.T
    return mu.T

# returns covariance matrix given matrix and mu (divides by N)
def get_sigma(matrix, mu):
    height, width = matrix.shape[0], matrix.shape[1]
    covariance_mtx = np.array([[float(0) for i in range(width)] for j in range(width)])
    for row in matrix:
        covariance_mtx += (row - mu.T).T * (row - mu.T)
    return covariance_mtx/height

# returns covariance matrix given matrix (divides by N -1)
def sigma(matrix):
    return np.cov(matrix.T)

# TODO figure out which sigma function to use
# both provide similar results
