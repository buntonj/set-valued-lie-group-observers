import numpy as np

def rotation_matrix(theta):
    return(np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]]))


# embeds state and noise into special euclidean group of order 2
# inputs:
# x - 3 dimensional state array, [x, y, theta]
# u - 2 dimensional input, [fwd/bwd, rotational]
def special_euclidean_exponential(x):
    chi = np.zeros((3,3))
    chi[0:2,0:2] = rotation_matrix(x[2])
    chi[0:2,2] = np.array([np.sin(x[2])*x[0]-x[1]+np.cos(x[2])*x[1],x[0]-np.cos(x[2])*x[0]+np.sin(x[2])*x[1]])/x[2]
    chi[2,2] = 1.0
    return chi

# approximation of the SE(2) exponential with taylor expansion using num_terms
def apx_special_euclidean_exponential(x,num_terms=10):
    V = build_V_matrix(x[2],num_terms)
    G = np.zeros((3,3))
    G[0:2,0:2] = rotation_matrix(x[2])
    G[0:2,2] = V @ x[0:2]
    G[2,2] = 1.0
    return G

def build_V_matrix(theta,num_terms):
    I = np.eye(2)
    J = np.array([[0.0, -1.0],[1.0, 0.0]])
    V = np.zeros((2,2))
    for i in range(0,num_terms):
        V += (((-1.0)**(i))/(np.math.factorial(2*i+1)))*(theta**(2*i))*I
        V += (((-1.0)**(i))/(np.math.factorial(2*i+2)))*(theta**(2*i+1))*J
    return V
    
 
def special_euclidean_input(u):
    v = np.array([[0.0, -u[0]*u[1], u[1]],
                  [u[0]*u[1], 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
    return v


# takes an element of special euclidean group x
# computes its inverse group element (faster than the typical matrix inverse)

def special_euclidean_inverse(x):
    #return np.linalg.inv(x)
    inv = np.zeros((3,3))
    inv[0:2,0:2] = x[0:2,0:2].T
    inv[0:2,2] = -x[0:2,0:2].T @ x[0:2,2]
    inv[2,2] = 1.0
    return inv




# Takes element of special euclidean group x
# and identifies it with its lie algebra's equivalent R^3 representation

def special_euclidean_logarithm(x):

    # extract angle
    theta = special_orthogonal_logarithm(x[0:2,0:2])

    # compute translation

    if theta != 0:
        # solving the system of equations:
        # V*t  = x[0:2,2]
        # for t, where V = 1/theta*[[sin(theta), -(1-cos(theta))], [1-cos(theta), sin(theta)]]
        # just using closed form expression for 2x2 matrices

        a = np.sin(theta)/theta
        b = (1.0-np.cos(theta))/theta

        V_inv = (1.0/(a**2.0 + b**2.0))*np.array([[a, b],[-b, a]])
        t = V_inv @ x[0:2,2]

        # assemble variable array: [x, y, theta]
        xi = np.array([t[0],t[1],theta])
    
    else : # if the angle is zero then we shouldn't divide by it
        xi = np.array([x[2,0],x[2,1],0.0])

    return xi

def apx_special_euclidean_logarithm(x,num_terms=15):
    theta = special_orthogonal_logarithm(x[0:2,0:2])
    V = build_V_matrix(theta,num_terms)
    t = np.linalg.solve(V, x[0:2,2])
    return np.array([t[0],t[1],theta])

# Takes element of special orthogonal group in 2D, R (rotation matrix)
# Returns the associated rotation angle theta

def special_orthogonal_logarithm(R):
    return np.arctan2(R[1,0],R[0,0])

def sample_simplex(n_simplex,n_samples,seed=0):
    rng = np.random.default_rng(seed=seed)
    sampled_simplex = rng.exponential(scale=1.0,size=(n_simplex,n_samples))
    norm_constants = np.sum(sampled_simplex,0)
    sampled_simplex = sampled_simplex / norm_constants[None,:] # normalize random points to live on simplex
    return sampled_simplex # array of shape (n_simplex,n_samples)