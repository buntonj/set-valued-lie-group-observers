import numpy as np
from utils.dubins_car.dubins_car import dubins_car # relative import of dubins car object
import matplotlib.pyplot as plt

x0 = np.array([0.0,0.0,0.0]) # initial position and heading
x0 = np.zeros((5,)) #initial position/heading, acc inputs
output = 'gps'

model = dubins_car(x0,output,acc_inputs=True)

T = 0.01 #simulation timestep
num_steps = 400
u = np.array([1.0,2.0]) # spin in a circle!
u = np.array([1.0,1.0]) # SPIRAL FASTER

x = np.zeros((model.n_state,num_steps))
x[:,0] = x0
y = np.zeros((model.n_output,num_steps))
y[:,0] = model.output_map(x[:,0],u)

for step in range(1,num_steps):
    model.step(u,T) # applying constant input
    x[:,step] = model.x
    y[:,step] = model.y
    


f = plt.figure()
ax = f.add_subplot()
ax.scatter(x[0,0],x[1,0],marker="*",s=100,color='green',label='Initial position') # initial position
ax.scatter(x[0,-1],x[1,-1],marker="*",s=100,color='red',label='Final position') # terminal position
ax.plot(x[0,:],x[1,:],label='Trajectory') # trajectory
ax.grid()
ax.set_xlabel('x position')
ax.set_ylabel('y position')
ax.set_title('Trajectory')
#plt.show()

f2 = plt.figure()
ax2 = f2.add_subplot()
ax2.plot(y[0,:],y[1,:])
ax2.grid()
ax2.set_title('GPS Output')
#plt.show()

# now testing the range/distance output, cheating to not simulate the system from scratch again
output = 'landmarks'
#model.reset()
model.set_output_map(output)

ax.scatter(model.landmarks[0,0],model.landmarks[1,0],marker="x",s=150,color='purple',label='Landmark 0') # plot landmark 0
ax.scatter(model.landmarks[0,1],model.landmarks[1,1],marker="x",s=150,color='green',label='Landmark 1') # plot landmark 1
ax.scatter(model.landmarks[0,2],model.landmarks[1,2],marker="x",s=150,color='red',label='Landmark 2') # plot landmark 2
ax.legend()

y2 = np.zeros((*model.n_output,num_steps))
y2[:,:,0] = model.output_map(x[:,0],u)
for step in range(1,num_steps):
    y2[:,:,step] = model.output_map(x[:,step],u)

f3 = plt.figure()
ax3 = f3.add_subplot()
ax3.plot(np.linalg.norm(y2[:,0,:],axis=0),color='purple',label='Landmark 0')
ax3.plot(np.linalg.norm(y2[:,1,:],axis=0),color='green',label='Landmark 1')
ax3.plot(np.linalg.norm(y2[:,2,:],axis=0),color='red',label='Landmark 2')
ax3.legend()
ax3.grid()
ax3.set_xlabel('timestep')
ax3.set_ylabel('distance from landmark')
ax3.set_title('Distance to outputs')

plt.show()

