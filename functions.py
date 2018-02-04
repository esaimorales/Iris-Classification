import numpy as np
import math

# returns mean (mu) of matrix
def get_mu(matrix):
    mu = np.mean(np.matrix(matrix), axis=0)
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

# returns identity covariance matrix
def get_id_matrix(matrix):
    mtx = np.zeros((4,4))
    for i in range(matrix.shape[0]):
        mtx[i,i] = matrix[i,i]
    return mtx
