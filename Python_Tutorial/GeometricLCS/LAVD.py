
import os, sys, copy
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.integrate import odeint
from typing import Dict, Callable, List

# Compute the ftle field on arbitrary dimensions.
def computeIVD(
    states: np.ndarray, times: np.ndarray, gradientFunction: Callable) -> np.ndarray:
    
    # Get the shape of the data
    n_particles, n_times, dim = np.shape(states)
    
    # Number of elements in a vorticity vector
    n_elem = int((dim-1)*dim/2)

    # Get the size of the output data
    assert len(times) == n_times, "The time vector does not match the data!"
    
    # define a function to compute the vorticity
    def get_vorticity_from_grad_v(position, time, gradientFunction):        
        # Compute the velocity gradient at the time and position
        velGrad = gradientFunction(position, time)
        
        # Get vorticity as the off-diagonal difference. 
        vort = np.zeros((n_elem,1))
        c = 0
        for i in range(dim-1, -1, -1):
            for j in range(dim-1, -1, -1):
                if j < i:
                    vort[c] = velGrad[i,j] - velGrad[j,i]
                    c += 1
        
        return vort
    
    # initialize
    ivd_array = np.zeros((n_particles, n_times))    # LAVD
    vort = np.zeros((n_particles,n_elem))        # vorticity vector
    vort_dev = np.zeros((n_particles,n_elem))
    
    # iterate through particles
    for i, t in enumerate(times):
        
        for j in range(n_particles):
            pos = states[j,i,:].squeeze()
            vort[j] = get_vorticity_from_grad_v(pos, t, gradientFunction)
        
        # Compute vorticity deviation
        avg_vort = np.mean(vort, axis=0)
        vort_dev = vort-avg_vort
        ivd = np.linalg.norm(vort_dev, axis=1)
        ivd_array[:,i] = np.copy(ivd)
            
    # Call the function
    return ivd_array


def computeLAVD(
    ivd_array: np.ndarray, times: np.ndarray, interval: float) -> np.ndarray:
    
    # array sizing
    n_particles, n_times = np.shape(ivd_array)
    
    # assertions
    assert n_times == len(times)
    assert interval <= n_times
        
    # Get a list of the time increments.  duplicate last difference.
    dt_vec = np.append(np.diff(times), times[-1]-times[-2])
    
    # Iterate through all of the ivd values to accumulate the lavd.
    lavd_array = np.nan * np.ones((n_particles, n_times))
    for i in range(n_times):
        j = i+interval
        
        if i+j <= n_times:
            lavd_array[:,i] = np.sum(ivd_array[:,i:i+j] * dt_vec[i:i+j], axis=1)
            
    return lavd_array
        
        
    

# For testing...
if __name__ == "__main__":
    sys.path.append(r"D:\OneDrive\Research\PhD\Code\Tutorials\LCS_Primer\Python_Tutorial\Flows")
    from Flows import Flow
    
    # Initialize a flow object
    function_name = "Gyre"
    gyre = Flow()

    ''' Now, we need to define the initial conditions for our integrations. We will define them on a grid for now.  '''
    # Specify the flow domain using dim[0] = x axis and dim[1] = y axis
    domain = np.array([[0, 2],[0, 1]])

    # # Now, make vectors associated with each axis.
    # n_y = 40            # number of rows
    # dx = 1/n_y          # row spacing
    # x_vec = np.arange(domain[0,0],domain[0,1],dx)     # 50 columns
    # y_vec = np.arange(domain[1,0],domain[1,1],dx)     # 25 rows

    # # Then, make the mesh grid and flatten it to get a single vector of positions.  
    # mesh = np.meshgrid(x_vec, y_vec, indexing= 'xy')
    # x = mesh[0].reshape(-1,1)
    # y = mesh[1].reshape(-1,1)
    # initial_conditions = np.append(x, y, axis=1)
    
    # Sample particles randomly.
    n_particles = 250
    initial_conditions = np.random.rand(n_particles, 2)
    initial_conditions[:,0] = initial_conditions[:,0] * 2

    # Next, we need to make a time vector
    t0 = 0      # initial time
    t1 = 30      # final time
    dt = 0.1    # time increment <- # For standard FTLE we will only need the first and last time, but 
                                    # it will be helpful when computing LAVD to have increments.
    interval = 150  # Time for computing the LAVD.
    
    time_vector = np.arange(t0,t1+dt,dt)
    '''We now need to specify the flow that the Flow object will operate on.'''
    parameters = {  # These are specified as defaults as well. 
        "A": 0.1,
        "epsilon":0.1,
        "omega":2*np.pi/10
    }
    gyre.predefined_function(function_name, initial_conditions, time_vector, parameters=parameters, include_gradv=True)

    # Integrate the particles over the defined time vector
    gyre.integrate_trajectories()
    
    # No compute the jacobian:
    ivd_array = computeIVD(gyre.states, gyre.time_vector, gyre.gradv_function)
    lavd_array = computeLAVD(ivd_array, gyre.time_vector, interval)
    
    #%%
    # Plotting a test frame
    frame = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter(gyre.states[:,frame,0], gyre.states[:,frame,1], s=15, c='gray')  #c=lavd_array[:,frame]
    
    # Define a window to look inside.
    win = [1.3, 0.4, 0.6, 0.4]
    def find_points_within_bounds(data, window):
        # Use boolean indexing to find points within bounds
        within_bounds = ((data[:,0] >= window[0]) & (data[:,0] <= window[0]+window[2]) &
                         (data[:,1] >= window[1]) & (data[:,1] <= window[1]+window[3]))
        
        # Return the points within bounds
        return within_bounds
    
    indices = find_points_within_bounds(gyre.states[:,frame,:].squeeze(), win)
    
    rectangle = patches.Rectangle((win[0], win[1]), win[2], win[3], edgecolor ='k', facecolor ='none')
    ax.add_patch(rectangle)
    sc = ax.scatter(gyre.states[indices,frame,0], gyre.states[indices,frame,1], 
               s=15, c=lavd_array[indices,frame]) 
    
    sc.set_clim([0, np.nanmax(lavd_array[:,frame])])
    
    ax.set_xlim([0,2])
    ax.set_ylim([0,1])
    ax.axis('scaled')
    #%%
