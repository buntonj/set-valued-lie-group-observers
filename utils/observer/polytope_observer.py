import numpy as np
import cvxpy as cp
import cdd

from utils.observer.pyfomo.redundancy_reduction import canonical_polytope
from .pyfomo.main import fourier_motzkin_eliminate_single
from .pyfomo.redundancy_reduction import canonical_polytope

class polytope_observer :
    def __init__(self,Aineq,bineq,state_noise,output_noise):

        # polytope data
        # set is described by:
        # Aineq * x <= bineq
        self.Aineq = Aineq
        self.bineq = bineq
        self.num_ineq = self.Aineq.shape[0] # number of inequalities

        self.n = Aineq.shape[1] # dimension of state space

        self.state_noise = state_noise # dynamics disturbance inf norm bound
        self.output_noise = output_noise # output disturbance inf norm bound

        self.tol = 1e-7 # tolerance on fourier-motzkin elimination
        self.fme_verbose = True # verbosity of fourier-motzkin elimination package

        self.chebyshev_center = None
        self.chebyshev_radius = None
        

    # step forward the polytope and update with the new output
    def step(self,sys_A,sys_C,y):
        Aineq1, bineq1 = self.propagate(sys_A)
        Aineq2, bineq2 = self.update(sys_C,y)

        # update the polytope equalities and inequalities
        self.Aineq = np.vstack((Aineq1,Aineq2))
        self.bineq = np.vstack((bineq1,bineq2))

        self.Aineq, self.bineq = canonical_polytope(self.Aineq,self.bineq,None,self.tol,self.fme_verbose)
        self.num_ineq = self.Aineq.shape[0]
    
    # step the polytope through the linear dynamics
    def propagate(self,sys_A):

        # build polytope over x(k), x(k-1)
        # inequality constraints
        Aineq = np.block([[np.eye(self.n), -sys_A],
                          [-np.eye(self.n), sys_A],
                          [np.zeros_like(self.Aineq), self.Aineq]])
        bineq = np.vstack((self.state_noise*np.ones((self.n,1)),self.state_noise*np.ones((self.n,1)),self.bineq))

        # quantifier elimination to remove constraints on previous state
        A, b = self.eliminate_prev(Aineq,bineq)

        return A, b # return the new polytope inequalities

    
    def update(self,sys_C,y):
        # build polytope over x(k)
        Aineq = np.vstack((sys_C,-sys_C))
        # h-stacking here because output is a row vector in python
        bineq = np.hstack((y + np.ones_like(y)*self.output_noise, -y + np.ones_like(y)*self.output_noise))[:,np.newaxis]
        return Aineq, bineq # return new polytope inequalities


    # function to eliminate the variables associated with the previous state
    def eliminate_prev(self,Aineq,bineq):
        for i in range(2*self.n-1,self.n-1,-1): # 
            Aineq, bineq = fourier_motzkin_eliminate_single(i,Aineq,bineq,self.tol,self.fme_verbose)
        return Aineq, bineq

    
    # returns current H-representation of polytope
    # {x :  Aineq @ x <= bineq}
    def current_polytope(self):
        return (self.Aineq,self.bineq)

    # tests if a given x is in the polytope
    def test_membership(self,x):
        return np.all(self.Aineq @ x <= self.bineq.T)

    
    def compute_chebyshev_center(self,verbose=False):
        xc = cp.Variable((self.n))
        r = cp.Variable()
        objective = cp.Maximize(r)
        constraints = [self.Aineq @ xc + np.linalg.norm(self.Aineq,axis=1)*r <= self.bineq[:,0]]
        prob = cp.Problem(objective,constraints)
        ans = prob.solve(solver=cp.GUROBI)
        if prob.status not in ["infeasible", "unbounded"]:
            self.chebyshev_center = xc.value
            self.chebyshev_radius = r.value
        else :
            raise InfeasibleException('Chebyshev centering failed. CVX status: {}'.format(prob.status))

    def compute_vertices(self):
        M = cdd.Matrix(np.hstack((self.bineq,-self.Aineq)))
        M.rep_type = cdd.RepType.INEQUALITY # declare this as the halfspace representation
        P = cdd.Polyhedron(M)
        X = np.array(P.get_generators()) #pull vertex set
        f = X[:,0] != 0
        self.vertices = X[f,1:]
        self.rays = X[~f,1:]
        return self.vertices




class InfeasibleException(Exception):
    pass