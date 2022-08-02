import numpy as np
from scipy.integrate import solve_ivp

# discrete-time system evolving according to:
# x(t+1) = A(t) @x(t) + B(t)@u(t)
# y(t) = C(t)@x(t) + D(t)@u(t)
#
# system matrices are passed in as FUNCTION HANDLES

class dt_ltv_system:
    def __init__(self,A,B,C,D,x0,t0=0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.n_state = np.shape(self.A(0.0))[0]
        self.n_output = np.shape(self.C(0.0,))[0]
        self.x0 = x0
        self.x = x0
        self.t = t0
        self.y = None

    def step(self,u,T):
        self.x = self.rhs(self.t,self.x,u)
        self.y = self.output_map(self.t,self.x,u)
        self.t += 1

    def reset(self):
        self.x = self.x0
        self.y = None
        self.t = 0


    def rhs(self,t,x,u):
        return self.A(t)@ x + self.B(t)@u

    def output_map(self,t,x,u):
        return self.C(t)@x + self.D(t)@u

    def system_matrices(self,t):
        return self.A(t), self.B(t), self.C(t), self.D(t)
