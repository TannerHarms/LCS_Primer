
import os, sys, copy
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Dict, Callable, List

# Compute the ftle field on arbitrary dimensions.
def computeLAVD(
    states: np.ndarray, times: np.ndarray, mesh: List[np.ndarray], 
    gradientFunction: Callable) -> np.ndarray:
    
    # Get the shape of the data
    n_particles, n_times, dim = np.shape(states)
    
    # Number of elements in a vorticity vector
    n_elem = int((dim-1)*dim/2)

    # Get the size of the output data
    assert np.size(mesh[0]) == n_particles, "The mesh provided does not match the data!"
    assert mesh[0].ndim == dim, "The mesh provided does not match the data!"
    assert len(times) == n_times, "The time vector does not match the data!"
    mesh_shape = np.shape(mesh)[1:]
    
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
    lavd_vec = np.zeros((n_particles))    # LAVD
    vort = np.zeros((n_particles,n_elem))        # vorticity vector
    vort_dev = np.zeros((n_particles,n_elem))
    dt_vec = np.append(np.diff(times), times[-1]-times[-2])
    # iterate through particles
    for i, t in enumerate(times):
        
        for j in range(n_particles):
            pos = states[j,i,:].squeeze()
            vort[j] = get_vorticity_from_grad_v(pos, t, gradientFunction)
        
        # Compute vorticity deviation
        avg_vort = np.mean(vort, axis=0)
        vort_dev = vort-avg_vort
        vort_dev_mag = np.linalg.norm(vort_dev, axis=1)
        
        # Update the intrinsic rotation angle.  
        lavd_vec += vort_dev_mag * dt_vec[i]
    
    # Reshape the data to the mesh.
    lavd_field = lavd_vec.reshape(tuple(list(mesh_shape)))
            
    # Call the function
    return lavd_field


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

    # Now, make vectors associated with each axis.
    n_y = 40            # number of rows
    dx = 1/n_y          # row spacing
    x_vec = np.arange(domain[0,0],domain[0,1],dx)     # 50 columns
    y_vec = np.arange(domain[1,0],domain[1,1],dx)     # 25 rows

    # Then, make the mesh grid and flatten it to get a single vector of positions.  
    mesh = np.meshgrid(x_vec, y_vec, indexing= 'xy')
    x = mesh[0].reshape(-1,1)
    y = mesh[1].reshape(-1,1)
    initial_conditions = np.append(x, y, axis=1)

    # Next, we need to make a time vector
    t0 = 0      # initial time
    t1 = 12      # final time
    dt = 0.1    # time increment <- # For standard FTLE we will only need the first and last time, but 
                                    # it will be helpful when computing LAVD to have increments.
    time_vector = np.arange(t0,t1+dt,dt)
    '''We now need to specify the flow that the Flow object will operate on.'''
    parameters = {  # These are specified as defaults as well. 
        "A": 0.1,
        "epsilon":1,
        "omega":2*np.pi/10
    }
    gyre.predefined_function(function_name, initial_conditions, time_vector, parameters=parameters, include_gradv=True)

    # Integrate the particles over the defined time vector
    gyre.integrate_trajectories()
    
    # No compute the jacobian:
    lavd_field = computeLAVD(gyre.states, gyre.time_vector, mesh, gyre.gradv_function)
    #%%
    plt.pcolormesh(mesh[0], mesh[1], lavd_field)
    plt.colorbar()
    plt.show()
    #%%
