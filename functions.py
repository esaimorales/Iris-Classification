import numpy
import math

def get_mu(matrix):
    mu = numpy.mean(numpy.matrix(matrix), axis=0)
    return mu.T

def get_sigma(matrix, mu):
    height, width = matrix.shape[0], matrix.shape[1]
    cov = numpy.array([[float(0) for i in range(width)] for j in range(width)])
    for row in matrix:
        cov += (row - mu.T).T * (row - mu.T)
    return cov/height

def sigma(matrix):
    return numpy.cov(matrix.T)


def pdf(x, mu, sigma):
    P = x.shape[0]

    prob = 1 / math.sqrt(((2 * math.pi) ** P) * numpy.linalg.det(sigma))
    ins = numpy.dot(numpy.dot((x - mu).T, numpy.linalg.inv(sigma)), x - mu)
    exp = math.exp(-0.5 * ins)

    return prob * exp

def probability(mu, sigma, x):
    a = 1/( (math.sqrt(2*math.pi) ** 4) *  )
