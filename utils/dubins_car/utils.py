# utilities for computing errors, lifts, etc. in dubins car example
import numpy as np
from ..shared.se2_utils import special_euclidean_inverse, build_V_matrix, special_orthogonal_logarithm

def left_invariant_error(x,xbar):
    return special_euclidean_inverse(x) @ xbar

def invert_left_invariant_error(eta,xbar):
    return xbar @ special_euclidean_inverse(eta)

def right_invariant_error(x,xbar):
    return xbar @ special_euclidean_inverse(x)

def invert_right_invariant_error(eta,xbar):
    return special_euclidean_inverse(eta) @ xbar

def lift_error(e):
    xi = np.zeros((4,))

    xi[0] = np.sin(e[2])
    xi[1] = np.cos(e[2])
    xi[2:] = build_V_matrix(e[2],15) @ e[0:2]

    return xi

def project_error(xi):
    e = np.zeros((3,))

    e[2] = np.arctan2(xi[0],xi[1])
    V = build_V_matrix(e[2],15)
    e[0:2] = np.linalg.solve(V, xi[2:])
    #X = (1.0/(1.0-xi[0]-xi[1]))*np.array([[xi[0], -xi[1]],
                                          #[xi[1]-1.0, xi[0]-1.0]])
    #e[0:2] = e[2] * (X @ xi[2:])                                                
    
    return e

def special_euclidean_state(G):
    return np.array([G[0,2],G[1,2],special_orthogonal_logarithm(G[0:2,0:2])])