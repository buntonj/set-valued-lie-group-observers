import numpy as np
from ..shared.se2_utils import special_euclidean_exponential,special_euclidean_input, special_euclidean_inverse
from scipy.integrate import solve_ivp


class dubins_car_se2 :
    def __init__(self,x0,output):
        self.n_state = 3
        self.x0 = x0 # save initial state for resetting purposes
        self.x = x0
        self.y = None
        self.set_output_map(output)


    # step forward T units of time with constant u applied
    # u[0] - translational velocity
    # u[1] - rotational velocity
    # starts at current self.x
    # saves new state to self.x
    # saves new output to self.y
    def step(self,u,T):
        f = lambda t,x: self.vectorize(self.rhs(x,u,t))
        sol = solve_ivp(f,(0,T),self.vectorize(self.x))
        self.x = self.unvectorize(sol.y[:,-1])
        self.y = self.output_map(self.x,u)

    # ODE right-hand-side
    # input:
    # x - state variables, [x, y, theta]
    # u - control input
    # t - time
    def rhs(self,x,u,t):
        return self.unvectorize(x) @ special_euclidean_input(u)
        #return np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[0]*u[1]])

    # range and bearing outputs
    # output is 2 x num_landmarks
    # y[:,i] = vector pointing to landmark i
    # not normalized, so distance to ith landmark is np.linalg.norm(y[i,:])
    def range_bearing_output(self,x,u):
        y = -special_euclidean_inverse(x) @ np.vstack((self.landmarks,np.ones((1,3))))
        return y[0:2,:]
        #return rotation_matrix(-x[2]) @ (x[0:2][:,np.newaxis] - self.landmarks)

    # GPS reading outputs
    # output is just state position
    def gps_output(self,x,u):
        return x[2,0:2]

    def vectorize(self,input):
        return np.reshape(input,9)

    def unvectorize(self,input):
        return np.reshape(input,(3,3))

    # reset initial conditions
    def reset(self):
        self.x = self.x0
        self.y = None

    # swap output map
    def set_output_map(self,output):
        if output == 'landmarks': 
            # three landmarks here
            self.landmarks = np.array([[0, 0],
                                       [0, 1],
                                       [1, 0]]).T  
            self.output_map = self.range_bearing_output
            self.n_output = (2,3)
        elif output == 'gps':
            self.output_map = self.gps_output
            self.n_output = 2
        else:
            self.output_map = lambda x,u : None # otherwise, no output

