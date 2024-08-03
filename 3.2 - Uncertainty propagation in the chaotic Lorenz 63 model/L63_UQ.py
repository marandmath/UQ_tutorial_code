# Code corresponding to Section 3.2 - "Uncertainty propagation in the chaotic 
# Lorenz 63 model" in the Supplementary Document of the paper "Taming 
# Uncertainty in a Complex World: The Rise of Uncertainty Quantification — A 
# Tutorial for Beginners". It specifically produces Figure 3 in the main text 
# (in Section "Examples of uncertainty propagation in linear and nonlinear 
# dynamical systems").
#
# Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
#
# Code: Simulate a chaotic system (Lorenz 63 model) with and without initial
# uncertainty (Python code).
#
# Python Package Requirements:
#   * matplotlib==3.5.3
#   * numpy==1.23.2
#   * scipy==1.9.0
#
# Useful documentation for MATLAB users: 
#   https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

################################################################################
################################################################################

# Importing the necessary Python packages to run this code.
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.seed(42) # Setting the seed number for consistent and reproducible results
T = 150 # Total length of the simulation in time units
dt = 0.005 # Numerical integration time step
N = round(T/dt) # Total numerical integration steps

# Model parameters (same as the ones in the original paper of Lorenz).
sigma = 10
rho = 28
beta = 8/3
Ens = 1000 # Number of ensemble members used to calculate the mean ensemble trajectory in the presence of initial uncertainty

################################################################################
##################### Simulation With Initial Uncertainty ######################
################################################################################

# State variables.
x1 = np.zeros((Ens, N))
y1 = np.zeros((Ens, N))
z1 = np.zeros((Ens, N))

# Initial values with an added normal noise of zero-mean and unit variance.
x1[:, 0] = 20 + norm.rvs(0, 1, Ens)
y1[:, 0] = -20 + norm.rvs(0, 1, Ens)
z1[:, 0] = 25 + norm.rvs(0, 1, Ens)

# Model simulation using forward Euler.
for i in range(1, N):
    x1[:, i] = x1[:, i-1] + sigma * (y1[:, i-1] - x1[:, i-1]) * dt
    y1[:, i] = y1[:, i-1] + (x1[:, i-1] * (rho - z1[:, i-1]) - y1[:, i-1]) * dt
    z1[:, i] = z1[:, i-1] + (x1[:, i-1] * y1[:, i-1] - beta * z1[:, i-1]) * dt

################################################################################
#################### Simulation With No Initial Uncertainty ####################
################################################################################

# State variables.
x2 = np.zeros(N)
y2 = np.zeros(N)
z2 = np.zeros(N)

# Initial value which differs from the first one by a small amount 
# (deterministic in this case).
x2[0] = 20
y2[0] = -20
z2[0] = 25

# Model simulation using forward Euler.
for i in range(1, N):
    x2[i] = x2[i-1] + sigma * (y2[i-1] - x2[i-1]) * dt
    y2[i] = y2[i-1] + (x2[i-1] * (rho - z2[i-1]) - y2[i-1]) * dt
    z2[i] = z2[i-1] + (x2[i-1] * y2[i-1] - beta * z2[i-1]) * dt

# Plotting the trajectories for both simulations.
plot_time = 15
fig = plt.figure()
gs = fig.add_gridspec(2, 6)
ax = fig.add_subplot(gs[:, 0:2], projection='3d')
plt.plot(*np.vstack((x2, y2, z2)), 'b', linewidth = 1.5)
ax.view_init(azim = 150, elev = 30)
ax.set_title(f'(a) Phase Plot of Lorenz 63 Attractor\nWith Deterministic IC (T={T})', fontsize = 18)
ax.set_xlabel('x', fontsize = 12)
ax.set_ylabel('y', fontsize = 12)
ax.set_zlabel('z', fontsize = 12)
ax = fig.add_subplot(gs[0, 2:6])
plt.plot(np.linspace(dt, round(plot_time/dt)*dt, round(plot_time/dt)), x2[0:round(plot_time/dt)], 'b', linewidth = 2)
ax.set_title(f'(b) Deterministic Initial Condition (T={plot_time})', fontsize = 18)
ax.set_xlabel('t', fontsize = 12)
ax.set_ylabel('x', fontsize = 12)
plt.autoscale(enable=True, axis='both', tight=True)
ax = fig.add_subplot(gs[1, 2:6])
ensembles_plot = plt.plot(np.tile(np.linspace(dt, round(plot_time/dt)*dt, round(plot_time/dt)), [Ens, 1]).T, x1[:, 0:round(plot_time/dt)].T, 'k', linewidth=0.5)
mean_plot, = plt.plot(np.linspace(dt, round(plot_time/dt)*dt, round(plot_time/dt)), np.mean(x1[:, 0:round(plot_time/dt)], 0), 'r', linewidth=2)
ax.legend([ensembles_plot[0], mean_plot], ['Ensemble Members', 'Mean Time Series'], fontsize = 14, loc = 'upper right')
ax.set_title(f'(c) Initial Condition With Small Uncertainty (T={plot_time})', fontsize = 18)
ax.set_xlabel('t', fontsize = 12)
ax.set_ylabel('x', fontsize = 12)
plt.autoscale(enable=True, axis='both', tight=True)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
fig.tight_layout(pad=0, w_pad=-2, h_pad=-2)
plt.show()