import numpy
import math

def get_mu(matrix):
    mu = numpy.mean(numpy.matrix(matrix), axis=0)
    return mu.T

def get_sigma(matrix, mu):
    height, width = matrix.shape[0], matrix.shape[1]
    covariance_mtx = numpy.array([[float(0) for i in range(width)] for j in range(width)])
    for row in matrix:
        covariance_mtx += (row - mu.T).T * (row - mu.T)
    return covariance_mtx/height

def sigma(matrix):
    return numpy.cov(matrix.T)

def probability(mu, sigma, x):
    a = 1/( (math.sqrt(2*math.pi) ** 4)   )
