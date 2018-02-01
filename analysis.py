import numpy as np
import math

from collections import Counter

# calculates probability density
def probability_density(mu, sigma, data_point):
    data_point = np.array([data_point]).T
    prd = (data_point - mu).T * np.linalg.inv(sigma)
    # print 'dataPoint', data_point
    # print 'mu', mu
    # print 'dataPoint - mu', (data_point - mu)
    # print 'inverse sigma', np.linalg.inv(sigma)
    # print 'product', prd
    # print 'product size', prd.shape

    e_bs = math.exp( (-0.5) * np.dot(prd, (data_point - mu)) )
    frac = 1/( (math.sqrt(2*math.pi) ** data_point.shape[0]) * np.linalg.det(sigma))
    return frac * e_bs

# reports analysis data and errors
def report(results):
    for row in results:
        # print row
        max_pd = row.index(max(row))
        if max_pd == 0:
            print 'Iris-setosa', max(row)
        elif max_pd == 1:
            print 'Iris-versicolor', max(row)
        elif max_pd == 2:
            print 'Iris-virginica', max(row)

# performs linear discriminant analysis on given test datasets
def lda(mu1, mu2, mu3, sigma, test1, test2, test3):

    res1, res2, res3 = [], [], []
    for data_point in test1:
        values = []
        values.append(probability_density(mu1, sigma, data_point))
        values.append(probability_density(mu2, sigma, data_point))
        values.append(probability_density(mu3, sigma, data_point))
        res1.append(values)

    report(res1)

    for data_point in test2:
        values = []
        values.append(probability_density(mu1, sigma, data_point))
        values.append(probability_density(mu2, sigma, data_point))
        values.append(probability_density(mu3, sigma, data_point))
        res2.append(values)

    report(res2)

    for data_point in test3:
        values = []
        values.append(probability_density(mu1, sigma, data_point))
        values.append(probability_density(mu2, sigma, data_point))
        values.append(probability_density(mu3, sigma, data_point))
        res3.append(values)

    report(res3)

# performs quadratic discriminant analysis on given test datasets
def qda(mu1, mu2, mu3, sigma1, sigma2, sigma3, test1, test2, test3):

    res1, res2, res3 = [], [], []
    for data_point in test1:
        values = []
        values.append(probability_density(mu1, sigma1, data_point))
        values.append(probability_density(mu2, sigma2, data_point))
        values.append(probability_density(mu3, sigma3, data_point))
        res1.append(values)

    # report(res1)

    for data_point in test2:
        values = []
        values.append(probability_density(mu1, sigma1, data_point))
        values.append(probability_density(mu2, sigma2, data_point))
        values.append(probability_density(mu3, sigma3, data_point))
        res2.append(values)

    # report(res2)

    for data_point in test3:
        values = []
        values.append(probability_density(mu1, sigma1, data_point))
        values.append(probability_density(mu2, sigma2, data_point))
        values.append(probability_density(mu3, sigma3, data_point))
        res3.append(values)

    # report(res3)
