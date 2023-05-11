from matplotlib import animation
import numpy as np
from utils.dubins_car.dubins_car import dubins_car # relative import of dubins car object
from utils.observer.polytope_observer import polytope_observer # importing polytope observer
from utils.dubins_car.utils import *
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Wedge
from matplotlib.animation import FuncAnimation,PillowWriter
from utils.shared.se2_utils import apx_special_euclidean_exponential,special_euclidean_exponential,apx_special_euclidean_logarithm,special_euclidean_logarithm
from utils.shared.se2_utils import sample_simplex

seed = 0
rng = np.random.default_rng(seed=seed)

apx_se2 = True # should we approximate the SE(2) log and exp functions with taylor expansions?

# defining shortcut handles for the lie algebra exponential and logarithm maps
if apx_se2:
    exp = apx_special_euclidean_exponential
    log = apx_special_euclidean_logarithm
else:
    exp = special_euclidean_exponential
    log = special_euclidean_logarithm


x0 = 2.0*(rng.random((3,))-0.5)  # initial position and heading, random in [-1,1]^3
output = 'landmarks'  # code is heavily tailored to this choice!

model = dubins_car(x0, output)

landmarks = model.landmarks  # pull the landmark locations
num_landmarks = landmarks.shape[1] # number of landmarks in simulation

T = 0.01  # simulation timestep
num_steps = 500
u = np.array([1.0, 2.0]) # spin in a circle!

x = np.zeros((model.n_state, num_steps))
x[:,0] = x0
y = np.zeros((*model.n_output, num_steps))
y[:,:,0] = model.output_map(x[:,0],u)

# setting up reference system
xhat0 = 2.0*(rng.random((3,))-0.5)
reference = dubins_car(xhat0,output) # initializing the reference model

# tracking reference trajectory
xref = np.zeros_like(x)
xref[:,0] = xhat0
estimate_center = np.zeros_like(x) # lie algebra (x,y,theta) representation of estimate center
estimate_center_group = np.zeros((3,3,num_steps)) # group element representation of estimate center
set_radius = np.zeros((num_steps,))

error_LR = 'R' # left or right invariant error (this code is really built for R)

if error_LR == 'L':
    error = left_invariant_error # function that computes error element in SE(2)
    error_to_state = invert_left_invariant_error
elif error_LR == 'R':
    error = right_invariant_error
    error_to_state = invert_right_invariant_error
else :
    raise ValueError('Invalid choice of left/right invariant error.')

# setting up polytope observer in lie algebra of group
set_n_state = 3 # we lift the error dynamics into R^3 so that the output map is linear
set_n_output = 2*num_landmarks # we have a 2D output for each landmark, we will vectorize it

set_sys_A = np.eye(set_n_state) # error dynamics are stationary

# construct the C matrix given the landmarks
set_sys_C = np.eye(set_n_state)
#for i in range(num_landmarks):
#    set_sys_C[2*i,:] = np.array([-1.0, 0.0, landmarks[1,i]])
#    set_sys_C[2*i+1,:] = np.array([0.0, -1.0, -landmarks[0,i]])


ub = 10.0*np.ones((set_n_state,1))
lb = -10.0*np.ones((set_n_state,1))

Aineq = np.zeros((2*set_n_state,set_n_state))
bineq = np.zeros((2*set_n_state,1))

for i in range(set_n_state):
    Aineq[2*i,i] = 1.0 # upper bound
    bineq[2*i] = ub[i]

    Aineq[2*i+1,i] = -1.0 # lower bound
    bineq[2*i+1] = -lb[i]

state_noise = 0.5
output_noise = 0.5
observer = polytope_observer(Aineq,bineq,state_noise,output_noise)
observer.compute_chebyshev_center(verbose=False)
error_center = observer.chebyshev_center
estimate_center_group[:,:,0] = error_to_state(exp(error_center),reference.state_group_rep())
estimate_center[:,0] = log(estimate_center_group[:,:,0])
set_radius[0] = observer.chebyshev_radius

observer.fme_verbose = False
sim_verbose = True

print('='*75)
print('STEP {}'.format(0))
print('='*75)
group_error = error(model.state_group_rep(),reference.state_group_rep())
init_algebra_error = log(group_error) # error as lie algebra element in R^3
lifted_init_algebra_error = init_algebra_error # lifted error in R^4
#projected_init_algebra_error = project_error(lifted_init_algebra_error)
#print('Lift/project error:',projected_init_algebra_error-init_algebra_error)
#print('Group error:', *group_error)
print('Algebra error:', init_algebra_error)
#print('Lifted error:', lifted_init_algebra_error)


ybar = reference.state_group_rep() @ y[:,:,0]# np.vstack((y[:,:,0],np.ones((1,num_landmarks))))
#print(reference.left_inverse(ybar))
vectorized_output = log(reference.left_inverse(ybar))
#print('xibar:',*xibar)
#lifted_output = reference.output_tangent(xibar)
#vectorized_output = np.reshape(xibar[0:2,:], (2*num_landmarks,), order='F')
print('ybar:',vectorized_output)
print('Output linearity error:', (set_sys_C @ lifted_init_algebra_error  - vectorized_output))
#print('output mismatch:',vectorized_output-set_sys_C @ lifted_init_algebra_error)
print('Beginning in polytope?',observer.test_membership(lifted_init_algebra_error))
#print('Size of polytope:',observer.chebyshev_radius)
#print('Initial estimate:',estimate_center[:,0])

# tracking polytopes
polytopes = [observer.current_polytope()]
vertices = [observer.compute_vertices()]

# simulation loop

noise_scale = 0.0

for step in range(1,num_steps):

    # step both models forward in time
    model.step(u,T)
    x[:,step] = model.x
    y[:,:,step] = model.y + 2.0*(rng.random(model.y.shape)-0.5)*noise_scale
    reference.step(u,T)
    xref[:,step] = reference.x

    ybar = reference.state_group_rep() @ y[:,:,step]
    vectorized_output = log(reference.left_inverse(ybar))#np.reshape(ybar[0:2,:], (2*num_landmarks,), order='F')

    observer.step(set_sys_A, set_sys_C, vectorized_output)
    polytopes.append(observer.current_polytope())
    vertices.append(observer.compute_vertices())

    # computing a state estimate with chebyshev center
    observer.compute_chebyshev_center(verbose=False)
    set_radius[step] = observer.chebyshev_radius
    error_center = observer.chebyshev_center
    estimate_center_group[:,:,step] = error_to_state(exp(error_center),reference.state_group_rep()) # group element representing center of estimate set
    estimate_center[:,step] = special_euclidean_state(estimate_center_group[:,:,step])#log(estimate_center_group[:,:,step]) # lie algebra element in R^3 representing center of estimate set
    
    # print out observer progress details if desired
    if sim_verbose:
        print('='*75)
        print('STEP {}'.format(step))
        print('='*75)
        #print('True state SE(2):',*model.state_group_rep())
        #print('Estimated state SE(2):',*estimate_center_group[:,:,step])
        #print('True state (R^3):', model.x)
        #print('Estimated state (R^3):', estimate_center[:,step])
        #print('Error:', np.linalg.norm(model.x - estimate_center[:,step]))
        group_error = error(model.state_group_rep(),reference.state_group_rep())
        algebra_error = log(group_error)
        lifted_algebra_error = algebra_error # error as lifted element in R^4
        #true_estimate = error_to_state(exp(project_error(lifted_algebra_error)),reference.state_group_rep())
        #print('True est - true state:', true_estimate-model.state_group_rep())
        #print('Group error:', *group_error)
        print('Algebra error:', algebra_error) # should be constnant!
        #print('Lifted algebra error:', lifted_algebra_error)
        print('ybar:',vectorized_output)
        #print('Output error (wrt init):', np.linalg.norm(vectorized_output - set_sys_C @ lifted_init_algebra_error))
        #print('error mismatch:',np.linalg.norm(init_true_error-true_error))
        print('Output linearity error:', (set_sys_C @ lifted_algebra_error  - vectorized_output))
        print('In polytope:',observer.test_membership(lifted_algebra_error))
        #print('Size of polytope:',observer.chebyshev_radius)
        #print('Inequalities:')
        #for i in range(observer.num_ineq):
        #    a = observer.Aineq[i,:]
        #    b = observer.bineq[i]
        #    flag = np.all(a @ lifted_algebra_error <= b)
        #    #flag = np.all(a @ observer.chebyshev_center <= b)
        #    print(a,b,flag)


f = plt.figure()
ax = f.add_subplot()
ax.scatter(x[0,0],x[1,0],marker="*",s=100,color='green',label='Initial position') # initial position
ax.scatter(x[0,-1],x[1,-1],marker="*",s=100,color='red',label='Final position') # terminal position
ax.plot(x[0,:],x[1,:],label='True trajectory',color='blue') # trajectory
ax.scatter(xref[0,0],xref[1,0],marker="*",s=100,color='green',label='Initial position') # initial position
ax.scatter(xref[0,-1],xref[1,-1],marker="*",s=100,color='red',label='Final position') # terminal position
ax.plot(xref[0,:],xref[1,:],label='Ref trajectory',color='red') # trajectory
ax.scatter(estimate_center[0,0],estimate_center[1,0],marker="*",s=100,color='green',label='Initial position') # initial position
ax.scatter(estimate_center[0,-1],estimate_center[1,-1],marker="*",s=100,color='red',label='Final position') # terminal position
ax.plot(estimate_center[0,:],estimate_center[1,:],label='Estimated trajectory',color='green') # trajectory
ax.grid()
ax.set_xlabel('x position')
ax.set_ylabel('y position')
ax.set_title('Trajectory')
ax.legend()

f2 = plt.figure()
ax2 = f2.add_subplot()
ax2.plot(set_radius)
ax2.set_xlabel('Time (steps)')
ax2.set_ylabel('Estimate set radius')
ax2.set_title('Convergence of set')
ax2.grid()

f3 = plt.figure()
ax3 = f3.add_subplot()
ax3.plot(np.linalg.norm(x - estimate_center,axis=0))
ax3.set_xlabel('Time (steps)')
ax3.set_ylabel('Estimate error')
ax3.set_title('Convergence of estimate')
ax3.grid()


# plot error region estimates
# roughly estimate the last plot by randomly sampling inside final polytope
num_vertices = vertices[-1].shape[0]
# if np.size(observer.rays) > 0:
#     print('Polyhedron is unbounded!')
# randomly sample in the polyhedron
num_samples = 4000
sampled_simplex = sample_simplex(num_vertices,num_samples,seed=seed)
sampled_polytope = vertices[-1].T @ sampled_simplex # randomly sampled points in polytope
sampled_states = np.empty((3,num_samples))

for i in range(num_samples):
    sampled_states[:,i] = special_euclidean_state(error_to_state(exp(sampled_polytope[:,i]),reference.state_group_rep(xref[:,-1])))

f4 = plt.figure()
ax4 = f4.add_subplot()
hull = ConvexHull(sampled_states.T[:,0:2])
P = plt.Polygon(sampled_states.T[hull.vertices,0:2],fill=True,alpha=0.5,color='blue')
ax4.add_patch(P)
ax4.scatter(sampled_states[0,:],sampled_states[1,:],marker='.',label='sample')
s = 0.05
lb = 180 * min(sampled_states[2,:])/np.pi
ub = 180* max(sampled_states[2,:])/np.pi
angle = Wedge((x[0,-1],x[1,-1]),1.2*s,lb,ub,alpha=0.5,color='blue')
ax4.add_patch(angle)
ax4.scatter(x[0,-1],x[1,-1],marker='*',s=100,label='true state',color='red')
ax4.arrow(x[0,-1],x[1,-1],s*np.cos(x[2,-1]),s*np.sin(x[2,-1]),width=0.002,color='red')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('{} random samples in error polytope'.format(num_samples))
ax4.grid()
ax4.legend()


s = 0.1
num_samples = 500
f5, ax5 = plt.subplots()
#sample_sc = ax5.scatter(np.zeros((num_samples,)),np.zeros((num_samples,)),label='Samples')
poly = plt.Polygon(rng.random((4,2)),fill=True,ec='blue',alpha=0.5,closed=True,lw=1)
ax5.add_patch(poly)
est_angle = Wedge((0,0),1.2*s,0,1,alpha=0.5,color='blue')
ax5.add_patch(est_angle)
truth_sc = ax5.scatter(0,0,label='Truth',marker='*',color='red',s=100)
truth_angle = ax5.arrow(0,0,1,1,width=0.002,color='red')
ax5.set_xlim(0.0,1.5)
ax5.set_ylim(-1.0,1.0)
ax5.grid()
ax5.legend()

def sample_states(i,num_samples): # randomly generates num_samples feasible states from error polytope at timestep i
    num_vertices = vertices[i].shape[0]
    sampled_polytope = vertices[i].T @ sample_simplex(num_vertices,num_samples)
    sampled_states = np.empty((3,num_samples))
    for j in range(num_samples):
        sampled_states[:,j] = special_euclidean_state(error_to_state(exp(sampled_polytope[:,j]),reference.state_group_rep(xref[:,i])))
    return sampled_states

def update(i):
    title = f'Timestep {i}'
    pts = sample_states(i,num_samples)
    hull = ConvexHull(pts.T[:,0:2])
    #sample_sc.set_offsets(np.c_[pts[0,:],pts[1,:]])
    est_angle.set_center((x[0,i],x[1,i]))
    lb = 180*min(pts[2,:]) / np.pi
    ub = 180*max(pts[2,:]) / np.pi
    est_angle.set_theta1(lb)
    est_angle.set_theta2(ub)
    truth_angle.set_data(x=x[0,i],y=x[1,i],dx=s*np.cos(x[2,i]),dy=s*np.sin(x[2,i]))
    poly.set_xy(pts.T[hull.vertices,0:2])
    truth_sc.set_offsets(np.c_[x[0,i], x[1,i]])
    ax5.set_title(title)
    return poly, truth_sc, truth_angle, est_angle

anim = FuncAnimation(f5,update,frames=np.arange(0,num_steps),repeat=True,interval=200,blit=False)

anim.save('set_evolution.gif', dpi=300, writer=PillowWriter(fps=5))


# TODO: Plotting the 4D Polyhedron by projecting it into two dimenions a few ways

plt.show()