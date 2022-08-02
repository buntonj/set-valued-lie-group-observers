# This repository implements a basic polytope-based set-valued observer for a dubins car model.

The novelty is that the dubins car is nonlinear, however, its error dynamics are linear and thus enable using set-valued observer tools developed for linear systems.

In the "utils" folder there are the dubins-car and linear-time-varying system objects for use in simulation.
Within shared utils, there are appropriate Lie logarithm and exponential maps as needed by the various simulations.  The outermost example python files run simple examples to check the set-valued observer (polytope_test_sim_1d.py and polytope_test_sim_2d.py) and the dubins car (dubins_test_sim.py). The files (dubins_polytope_observer.py and dubins_polytope_observer_th.py) use the polytope observer on the lie logarithm of the error dynamics of the dubins car.
