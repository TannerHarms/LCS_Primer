
'''
Here the gyre flow common to the LCS literature is implemented.  
For more information about the flow, see ... TODO
'''

import numpy as np
import sympy as sym
from typing import Dict, Callable

Gyre_defaultParameters: Dict[str, float] = {
    "A" : 0.1,
    "epsilon" : 0.1,
    "omega" : 2*np.pi/10
}

def Gyre(parameters: Dict[str, float] = Gyre_defaultParameters, gradV: bool = False) -> Callable:
    
    # Read the parameters
    A = parameters["A"]
    B = parameters["B"]
    C = parameters["C"]
    
    # Symbolic Gyre Function
    xs = sym.symbols('x, y, z')
    x, y, z, t = sym.symbols('x, y, z, t')

    u = A * sym.sin(z) + C * sym.cos(y)
    v = B * sym.sin(x) + A * sym.cos(z)
    w = C * sym.sin(y) + B * sym.cos(x)
    U = [u, v, w]

    if gradV:
        gradV = [
                [sym.diff(u,x), sym.diff(u,y), sym.diff(u,z)],
                [sym.diff(v,x), sym.diff(v,y), sym.diff(v,z)],
                [sym.diff(w,x), sym.diff(w,y), sym.diff(w,z)]
            ]

    # Lambda Functions
    U_fun = sym.lambdify([xs, t], U)
    gradV_fun = sym.lambdify([xs, t], gradV)
    
    def flowFun(q, t):
        return U_fun(q, t)

    def gradVFun(q, t):
        return np.array(gradV_fun(q, t))
    
    if not gradV:
        return flowFun
    else:
        return flowFun, gradVFun

