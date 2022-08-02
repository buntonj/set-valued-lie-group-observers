import numpy as np
import matplotlib.pyplot as plt
from utils.generic_ltv_system.dt_ltv_system import dt_ltv_system
from utils.observer.polytope_observer import polytope_observer

T = 1 # timestep
num_steps = 100

n_state = 1
n_output = 1

# defining discrete-time LTV system state matrices
def A(t):
    return np.array([[0.99]])

def C(t):
    return np.array([[1.0]])

def B(t):
    return np.array([[0.0]])

def D(t):
    return np.array([[0.0]])


x0 = np.array([5.0]) # initial condition

# initialize LTV system object
t0 = 0
model = dt_ltv_system(A,B,C,D,x0,t0)

# Building the polytope set-valued observer
Aineq = np.array([[1.0], 
                  [-1.0]])

bineq = np.array([[10.0],
                  [10.0]])

state_noise = 0.5
output_noise = 0.5

observer = polytope_observer(Aineq,bineq,state_noise,output_noise)
observer.fme_verbose = False
sim_verbose = True

u = np.array([0.0]) # set the input to be applied to system
x = np.zeros((model.n_state,num_steps))
x[:,0] = x0
y = np.zeros((model.n_output,num_steps))

# tracking polytopes
polytopes = [observer.current_polytope()]
ub = np.zeros_like(x) # we're in just R, so we can track upper and lower bounds at each step
lb = np.zeros_like(x)
ub[:,0] = observer.bineq[0,0]
lb[:,0] = -observer.bineq[1,0]

for step in range(1,num_steps):
    if sim_verbose:
        print('STEP {}'.format(step))
    # pull relevant system matrices
    (A_prev,_,C_prev,_) = model.system_matrices(model.t) #pull most recent system matrices
   
    # step dynamical system forward
    model.step(u,T)
    x[:,step] = model.x
    y[:,step] = model.y
    
    # step observer forward 
    observer.step(A_prev,C_prev,model.y)
    polytopes.append(observer.current_polytope())
    
    # print out observer progress details if desired
    if sim_verbose:
        print(model.x,observer.test_membership(model.x))
        for i in range(observer.num_ineq):
            a = observer.Aineq[i,:]
            b = observer.bineq[i,:]
            if a < 0: # some cheeky checks for upper and lower bounds, in general need to find box containing polyhedra
                lb[:,step] = -b
            else :
                ub[:,step] = b
            flag = a @ model.x <= b
            print(a,b,flag)

# now we can plot the bounds just for fun
f = plt.figure(figsize=(10,7))
ax = f.add_subplot()
ax.fill_between(range(0,num_steps),ub[0,:],lb[0,:],alpha=0.4)
ax.plot(ub[0,:],'--',lw=1,label='Upper bound')
ax.plot(lb[0,:],'--',lw=1,label='Lower bound')
ax.plot(x[0,:],lw=2,color='black',label='Trajectory')
ax.grid()
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_title('Trajectory bounds')
ax.legend()
plt.show()