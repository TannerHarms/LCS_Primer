import numpy as np
import matplotlib.pyplot as plt

# Grid size 
N = 128

# Viscosity - You can change it to your desired values 
viscosity = 0.000625

# Time Step 
dt = 0.02

# Frequency
freq = 2.0

x = np.linspace(0, 1, N, endpoint=False)
y = np.linspace(0, 1, N, endpoint=False)

Y, X = np.meshgrid(y, x)

# Initial Conditions 
omega = np.sin(freq * 2.0 * np.pi * X) * np.cos(freq * 2.0 * np.pi * Y)

# Omega in t + 1 
omega_ = np.empty_like(omega)

f_x = np.empty_like(omega)
f_y = np.empty_like(omega)

for count in range(4000):
    # zero-padded 1st order derivative
    f_x[:, :-1] = np.diff(omega, axis=1) / 1.
    f_x[:, -1] = omega[:, 0] - omega[:, -1]
    
    f_y[:-1, :] = np.diff(omega, axis=0) / 1.
    f_y[-1, :] = omega[0, :] - omega[-1, :]
    
    omega_npo = np.pad(omega, ((1, 1), (1, 1)), mode='wrap')
    laplacian = ((np.roll(omega_npo, 1, axis=0) + np.roll(omega_npo, -1, axis=0) +
                  np.roll(omega_npo, 1, axis=1) + np.roll(omega_npo, -1, axis=1)) -
                 4 * omega_npo[1:-1, 1:-1]) / (1.**2)
    
    omega_ = omega + dt * (- (f_x * f_y) + viscosity * laplacian)
    omega, omega_ = omega_, omega

plt.imshow(omega)
plt.colorbar()
