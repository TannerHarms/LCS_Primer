'''
Here the Bickley Jet flow of Rypina et al. (2007) is implemented.  
For more information about the flow, see ... TODO
'''

import numpy as np
import sympy as sym
from typing import Dict, Callable

U_default = 62.66
BickleyJet_defaultParameters: Dict[str, float] = {
    "U" : U_default,
    "L" : 1.77 * 10**6,
    "r0" : 6.371 * 10**6,
    "epsilon" : [0.0075, 0.15, 0.3],
    "c" : [0.1446*U_default, 0.205*U_default, 0.461*U_default],
    "bounds" : [[0, 20 * 10**6],[-3 * 10**6, 3 * 10**6]]
}

def BickleyJet(parameters: Dict[str, float] = BickleyJet_defaultParameters, gradV: bool = False) -> Callable:
    
    # Read the parameters
    U_jet = parameters["U"]
    L = parameters["L"]
    r0 = parameters["r0"]
    epsilon = parameters["epsilon"]
    c = parameters["c"]
    bounds = parameters["bounds"]
    
    # computed parameters
    sigma = [val - c[2] for val in c]
    k = [2*(i+1)/r0 for i in range(3)]
    
    # Symbolic Gyre Function
    xs = sym.symbols('x, y')
    x, y, t = sym.symbols('x, y, t')

    psi_0 = c[2]*y - U_jet*L*sym.tanh(y/L)
    psi_1 = U_jet*L*(sym.sech(y/L))**2 * (
        epsilon[0]*sym.cos(k[0]*(x - sigma[0]*t)) + 
        epsilon[1]*sym.cos(k[1]*(x - sigma[1]*t)) + 
        epsilon[2]*sym.cos(k[2]*(x - sigma[2]*t)) 
    )
    psi = psi_0 + psi_1
    
    # Compute the velocities
    u = sym.diff(psi, y)
    v = -sym.diff(psi, x)
    U = [u,v]

    if gradV:
        gradV = [[sym.diff(u,x), sym.diff(u,y)], [sym.diff(v,x), sym.diff(v,y)]]

    # Lambda Functions
    U_fun = sym.lambdify([xs, t], U)
    gradV_fun = sym.lambdify([xs, t], gradV)
    
    def flowFun(q, t):
        # make periodic in x
        if q[0] < bounds[0][0] or q[0] > bounds[0][1]:
            q[0] = bounds[0][0] + ((q[0] - bounds[0][0]) % (bounds[0][1] - bounds[0][0]))
        
        return U_fun(q, t)

    def gradVFun(q, t):
        # make periodic in x
        if q[0] < bounds[0][0] or q[0] > bounds[0][1]:
            q[0] = bounds[0][0] + ((q[0] - bounds[0][0]) % (bounds[0][1] - bounds[0][0]))
        
        return np.array(gradV_fun(q, t))
    
    if not gradV:
        return flowFun
    else:
        return flowFun, gradVFun