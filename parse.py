import numpy as np

# parse data from given file
def parse_file(file_name):
    with open(file_name) as f:
        return [[float(val) for val in line.split(',')[:4]] for line in f]

def get_train(data):
    return (data[0:40], data[50:90], data[100:140])

def get_test(data):
    return (data[40:50], data[90:100], data[140:150])
