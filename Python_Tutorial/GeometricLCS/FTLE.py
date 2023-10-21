
import os, sys, copy
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Dict, Callable, List


def computeJacobian(trajectories: np.ndarray, mesh: List[np.ndarray]) -> np.ndarray:
    
    # Get the size of the data
    n_particles, n_times, dim = np.shape(trajectories)
    
    # Ensure that the mesh represents the data
    assert np.size(mesh[0]) == n_particles, "The mesh provided does not match the data!"
    assert mesh[0].ndim == dim, "The mesh provided does not match the data!"
    mesh_shape = np.shape(mesh)[1:]
    
    # We are working with data that is x-y indexed.  for the next steps we will switch that.
    mesh_ij = np.transpose(mesh, axes=(0, 2, 1, *range(3, dim+1)))
    
    # Find the spacing in each dimension on the mesh.
    #   We are assuming that the spacing is the same between each 
    #   initial condition for a given dimension.
    spacing = []
    for i in range(dim):
        diff = np.abs(mesh_ij[i][tuple([0]*i + [1] + [0]*(dim-i-1))] - mesh_ij[i][tuple([0]*dim)])
        spacing.append(diff)
    
    # Get the differences in particle position from the first to the final time
    # and format them to the mesh
    q0 = trajectories[:,0,:].squeeze()
    q1 = trajectories[:,-1,:].squeeze()
    dq = q1-q0
    dq_meshed = dq.reshape(tuple(list(mesh_shape) + [dim]))
    
    # Initialize a Jacobian
    J_array = np.zeros(tuple(list(mesh_shape) + [dim,dim]))
    
    # Use the final positions of the 
    for i in range(dim):
        for j in range(dim):
            J_array[...,i,j] = np.gradient(dq_meshed[...,i],spacing[i], axis=j)
    
    return J_array

# Compute the ftle field on arbitrary dimensions.
def computeFTLE(J_array: np.ndarray, dt: float) -> np.ndarray:
    
    # dimension of the flow
    dim = np.shape(J_array)[-1]
    dim_sizes = tuple(np.shape(J_array)[:-2])
    
    # initialize
    ftle_field = np.zeros(dim_sizes)
    
    # Recursive function to ensure arbitrary dimension support
    def iterate_arbitrary_dimensions(dim, d_now=0, index=[0]*dim):
        if d_now < dim:
            d_now += 1
            for j in range(dim_sizes[d_now-1]):
                index[d_now-1] = j
                iterate_arbitrary_dimensions(dim, d_now, index)
        else:
            _, lam, _ = np.linalg.svd(J_array[tuple(index + [...])])
            ftle_field[tuple(index)] = 1/np.abs(dt)*np.log(np.max(lam))
            
    # Call the function
    iterate_arbitrary_dimensions(dim)
    return ftle_field


# For testing...
if __name__ == "__main__":
    sys.path.append(r"C:\Users\harms\OneDrive\Research\PhD\Code\Tutorials\LCS_Primer\Python_Tutorial\Flows")
    from Flows import Flow
    
    # Initialize a flow object
    function_name = "Gyre"
    gyre = Flow()

    ''' Now, we need to define the initial conditions for our integrations. We will define them on a grid for now.  '''
    # Specify the flow domain using dim[0] = x axis and dim[1] = y axis
    domain = np.array([[0, 2],[0, 1]])

    # Now, make vectors associated with each axis.
    n_y = 25            # number of rows
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
        "epsilon":0.1,
        "omega":2*np.pi/10
    }
    gyre.predefined_function(function_name, initial_conditions, time_vector, parameters=parameters, include_gradv=True)

    # Integrate the particles over the defined time vector
    gyre.integrate_trajectories()
    
    # No compute the jacobian:
    J = computeJacobian(gyre.states, mesh)
    ftle_field = computeFTLE(J,t1-t0)
    #%%
    plt.pcolormesh(mesh[0], mesh[1], ftle_field)
    plt.colorbar()
    plt.show()
    #%%
