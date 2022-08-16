import numpy as np
from .se2_utils import *

# left multiply function, expects two tuples
def lm(x,y):
    return (x[0]@y[0],x[1]+y[1])


# embeds state and noise into special euclidean group of order 2
# inputs:
# x - 5 dimensional state array, [x, y, theta, trans_vel, ang_vel]
def se2_x_R2_exponential(x):
    
    chi_se2 = special_euclidean_exponential(x[:3])
    return (chi_se2, x[-2:])

# approximation of the SE(2) exponential with taylor expansion using num_terms
def apx_se2_x_R2_exponential(x,num_terms=10):
    G = apx_special_euclidean_exponential(x[:3],num_terms)
    return (G,x[-2:])


# takes an element of special euclidean group x x R^2
# computes its inverse group element (faster than the typical matrix inverse)
def se2_x_R2_inverse(x):
    #return np.linalg.inv(x)
    inv = special_euclidean_inverse(x[0])
    return (inv,-x[1])


def se2_x_R2_logarithm(x):
    xi = special_euclidean_logarithm(x[0])
    return np.concatenate((xi,x[1]))


def apx_se2_x_R2_logarithm(x,num_terms=15):
    xi = apx_special_euclidean_logarithm(x[0],num_terms)
    return np.concatenate((xi,x[1]))