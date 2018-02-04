from parse import parse_file
from parse import get_train
from parse import get_test

import numpy
import math

from functions import get_mu
from functions import get_sigma
from functions import sigma
from functions import get_id_matrix

from analysis import lda
from analysis import qda

# extract data from text file
iris_data = parse_file('training.txt')

# prepare training dataset
training = get_train(iris_data)
setosa, versicolor, virginica = training[0], training[1], training[2]

# prepare testing dataset
testing = get_test(iris_data)
test_setosa, test_versicolor, test_virginica = testing[0], testing[1], testing[2]

# set numpy matrix
setosa = numpy.array(setosa)
versicolor = numpy.array(versicolor)
virginica = numpy.array(virginica)

# get mu
mu1 = get_mu(setosa)
mu2 = get_mu(versicolor)
mu3 = get_mu(virginica)

# get sigma
sigma1 = get_sigma(setosa, mu1)
sigma2 = get_sigma(versicolor, mu2)
sigma3 = get_sigma(virginica, mu3)

# prepare average sigma for Linear Discriminant Analysis
average_sigma = (sigma1 + sigma2 + sigma3)/3

# do LDA and QDA
print 'Calculating LDA and QDA for test data...'

print 'LDA... '
lda(mu1, mu2, mu3, average_sigma, test_setosa, test_versicolor, test_virginica)
print 'QDA... '
qda(mu1, mu2, mu3, sigma1, sigma2, sigma3, test_setosa, test_versicolor, test_virginica)


# do LDA and QDA for training
print 'Calculating LDA and QDA for training data...'

print 'LDA... '
lda(mu1, mu2, mu3, average_sigma, setosa, versicolor, virginica)
print 'QDA... '
qda(mu1, mu2, mu3, sigma1, sigma2, sigma3, setosa, versicolor, virginica)

# calculate new identity matrix for LDA
id_average_sigma = get_id_matrix(average_sigma)

# calculate new identity matrix for QDA
id_sigma1 = get_id_matrix(sigma1)
id_sigma2 = get_id_matrix(sigma2)
id_sigma3 = get_id_matrix(sigma3)

# do LDA and QDA on test data w/ independent features
print 'Calculating LDA and QDA for test data with independent features... '

print 'LDA... '
lda(mu1, mu2, mu3, id_average_sigma, test_setosa, test_versicolor, test_virginica)
print 'QDA... '
qda(mu1, mu2, mu3, id_sigma1, id_sigma2, id_sigma3, test_setosa, test_versicolor, test_virginica)

# do LDA and QDA on training data w/ independent features
print 'Calculating LDA and QDA for training data with independent features... '

print 'LDA... '
lda(mu1, mu2, mu3, id_average_sigma, setosa, versicolor, virginica)
print 'QDA... '
qda(mu1, mu2, mu3, id_sigma1, id_sigma2, id_sigma3, setosa, versicolor, virginica)
