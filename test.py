import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Lorenz system parameters
sigma = 10
rho = 28
beta = 8/3

def lorenz_system(current_state, t):
    x, y, z = current_state
    dx_dt = sigma*(y - x)
    dy_dt = x*(rho - z) - y
    dz_dt = x*y - beta*z
    return [dx_dt, dy_dt, dz_dt]

# Initial conditions: (x, y, z)
initial_cond1 = [1, 1, 1]   
initial_cond2 = [0, 1, 1.5] 
initial_states = [initial_cond1, initial_cond2]

# Time points
t = np.linspace(0, 50, 5000)

# Solve differential equation for each initial condition
trajectories = [odeint(lorenz_system, init_state, t) for init_state in initial_states]

# Create figure
fig = plt.figure(facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

# Set color and linewidth for trajectories
colors = ['cyan', 'yellow']
linewidths = [1, 1.5]

lines = [ax.plot([], [], [], '-', c=c, lw=lw)[0] for c, lw in zip(colors, linewidths)]
ax.grid(False)

# Set up formatting for the movie files
Writer = animation.writers['pillow']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

def animate(i):
    for line, traj in zip(lines, trajectories):
        line.set_data(traj[:i, 0], traj[:i, 1])
        line.set_3d_properties(traj[:i, 2])
    return lines

ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=1)

# Save the animation
ani.save('lorenz_attractor.mp4', writer=writer)

plt.show()