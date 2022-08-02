import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
from utils.shared import special_euclidean_input
from utils.shared.se2_utils import rotation_matrix, special_euclidean_exponential, special_euclidean_logarithm

class observer:
    def __init__(self,f,xhat,T,LR,D):

        self.xhat = xhat #initialize estimate in R^3, [x,y,theta]
        self.T = T# output frequency

        if LR == 'L': # left or right invariant filter?
            self.discrete_update = self.LIEKF
            self.compute_gains = self.compute_gains_L
        elif LR == 'R':
            self.discrete_update = self.RIEKF
            self.compute_gains = self.compute_gains_R
        else :
            raise ValueError('Must specify left or right invariant outputs.')

        self.f = f # differential equation right-hand-side, function of (x,u,t)
        self.D = D # object on which the lie group state acts on to produce outputs, in R^(3 x k)
        self.num_outputs = D.shape[1] # what is k?
        self.D = np.reshape(D,(3*self.num_outputs,1))

        self.Ln = self.compute_gains(LR) # map from R^(3k) to R^(3)

    def step(self,u,y):
        odefun = lambda t,x: self.f(x,u,t) # define rhs with fixed input
        t,xhat = solve_ivp(self.f,(0.0,self.T),self.xhat)
        
        self.xhat = xhat[-1,:] # we only care about the value of the estimate after evolving for T time

        xhat = special_euclidean_exponential(self.xhat) # reshape the state into its SE(2) group element

        self.compute_gains(u) # compute the observer gains via Ricatti equation with input-dependent matrices

        xhat = self.discrete_update(xhat,y)

        self.xhat = special_euclidean_logarithm(xhat) # change estimate from its lie group element to [x,y,theta]



    # update of state estimate when new output arrives
    def LIEKF(self,xhat,y):
        alg_element = self.Ln @ (np.linalg.inv(xhat)@ y - self.D)
        return xhat @ special_euclidean_exponential(alg_element)

    def RIEKF(self,xhat,y):
        alg_element = self.Ln @ (xhat @ y - self.D)
        return special_euclidean_exponential(alg_element) @ xhat


    # TO DO!
    # Need to either input or require associated A,H, Q, etc matrices for ricatti equation
    # then solve ricatti equation to compute the gains
    def compute_gains_L(self,u):
        A  = -np.array([[0.0, 0.0, 0.0],
                        [0.0, 0.0, -u[0]*u[1]],
                        [-u[0], u[0]*u[1], 0.0]])

        H = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        
        Q = self.process_covariance

        R = rotation_matrix(self.xhat[2])

        N = R @ self.measurement_covariance @ R.T

        
        return


