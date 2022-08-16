import numpy as np
from ..shared.se2_utils import rotation_matrix,special_euclidean_inverse
from scipy.integrate import solve_ivp


class dubins_car :
    def __init__(self,x0,output,acc_inputs=False):
        if acc_inputs : # option to include angular and linear acceleration inputs
            self.n_state = 5 # state is x,y,theta, dv, dtheta
            self.rhs = self.rhs_acc_inputs
            self.state_group_rep = self.state_group_rep_se2_x_R2
        else :
            self.n_state = 3
            self.rhs = self.rhs_vel_inputs
            self.state_group_rep_se2_x_R2
        
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
        f = lambda t,x: self.rhs(x,u,t)
        sol = solve_ivp(f,(0,T),self.x)
        self.x = sol.y[:,-1] 
        self.x[0:2] = (self.x[0:2] + np.pi) % (2*np.pi) - np.pi
        self.y = self.output_map(self.x,u)

    # ODE right-hand-sides
    # input:
    # x - state variables, [x, y, theta]
    # u - control input (velocity)
    # t - time
    def rhs_vel_inputs(self,x,u,t):
        return np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[0]*u[1]])

    # input:
    # x - state variables [x, y, theta, dv, dtheta]
    # u - control input [u_v, a_theta] (accelerations)
    # t - time
    # returns:
    # (5,) numpy array of the ODE right-hand-side with current applied inputs
    def rhs_acc_inputs(self,x,u,t):
        return np.array([x[3]*np.cos(x[2]), x[3]*np.sin(x[2]), x[3]*x[4], u[0], u[1]])
    

    # range and bearing outputs
    # output is 2 x num_landmarks
    # y[:,i] = vector pointing to landmark i
    # not normalized, so distance to ith landmark is np.linalg.norm(y[i,:])
    def range_bearing_output(self,x,u):
        #y = -special_euclidean_inverse(self.state_group_rep(x)) @ np.vstack((self.landmarks,np.ones((1,3))))
        #return y#[0:2,:]
        return np.vstack((rotation_matrix(-x[2]) @ (x[0:2][:,np.newaxis] - self.landmarks),-np.ones((1,3))))

    # GPS reading outputs
    # output is just state position
    def gps_output(self,x,u):
        return x[0:2]

    # left-inverse of output map for range-bearing outputs
    def range_bearing_left_inverse(self,y):
        x = np.linalg.solve(np.vstack((self.landmarks,np.ones((1,3)))).T,-y.T)
        return x.T

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
            self.left_inverse = self.range_bearing_left_inverse
            self.n_output = (3,3)#(2,3)
        elif output == 'gps':
            self.output_map = self.gps_output
            self.n_output = 2
        else:
            self.output_map = lambda x,u : None # otherwise, no output

    # return the current state as an element of SE(2)
    # assumes input is ordered as [x,y,theta,__]
    def state_group_rep_se2(self,state=None):
        if state is None:
            state = self.x
        G = np.zeros((3,3))
        G[0:2,0:2] = rotation_matrix(state[2])
        G[0:2,2] = state[0:2]
        G[2,2] = 1.0
        return G

    def state_group_rep_se2_x_R2(self,state=None):
        if state is None:
            state = (self.x[:3],self.x[3:])
        G = self.state_group_rep_se2(state[0])
        return (G,state[1])

