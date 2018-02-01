from parse import parse_file
from parse import get_train
from parse import get_test

import numpy
import math

from functions import get_mu
from functions import get_sigma
from functions import sigma

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
sigma1 = sigma(setosa)
sigma2 = sigma(versicolor)
sigma3 = sigma(virginica)

# prepare average sigma for Linear Discriminant Analysis
average_sigma = (sigma1 + sigma2 + sigma3)/3

# do LDA and QDA
lda(mu1, mu2, mu3, average_sigma, test_setosa, test_versicolor, test_virginica)
qda(mu1, mu2, mu3, sigma1, sigma2, sigma3, test_setosa, test_versicolor, test_virginica)
