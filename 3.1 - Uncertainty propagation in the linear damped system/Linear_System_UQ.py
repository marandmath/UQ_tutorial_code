# Code corresponding to Section 3.1 - "Uncertainty propagation in the linear 
# damped system" in the Supplementary Document of the paper "Taming Uncertainty 
# in a Complex World: The Rise of Uncertainty Quantification — A Tutorial for 
# Beginners". It specifically produces Figure 2 in the main text (in Section 
# "Examples of uncertainty propagation in linear and nonlinear dynamical 
# systems").
#
# Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
#
# Code: Simulating a linear (stochastic) damped system with initial uncertainty 
# (Python code).
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
N = 2000 # Total numerical integration steps
dt = 0.005 # Numerical integration time step
T = N*dt # Total length of the simulation in time units

a = 1 # Deterministic and constant damping of the process
f = 1 # Deterministic and constant forcing of the process
sigma = 0.5 # Noise amplitude/feedback in the case where a stochastic process is being simulated
Ens = 1000 # Number of ensemble members used to calculate the mean ensemble trajectory in the presence of initial uncertainty

x1 = np.zeros(N) # Stochastic process with deterministic initial condition
x2 = np.zeros((Ens, N)) # Stochastic process with uncertainty in its initial value

# Initializing the processes.
x1[0] = 2
x2[:, 0] = 2 + 0.3 * norm.rvs(0, 1, Ens) # A small zero-mean normal noise is added to all ensemble members initially

# Model simulation using forward Euler.
# Remove the commented-out lines for x1[i] and x2[:, i] to instead add a Wiener
# noise or Brownian motion to the dynamical linear system and make it stochastic 
# thus simulating the system using the Euler-Maruyama method for numerical 
# integration.
for i in range(1, N):
    x1[i] = x1[i-1] + (-a * x1[i-1] + f) * dt
    x2[:, i] = x2[:, i-1] + (-a * x2[:, i-1] + f) * dt
    # x1[i] = x1[i-1] + (-a * x1[i-1] + f) * dt + sigma * np.sqrt(dt) * norm.rvs(0, 1)
    # x2[:, i] = x2[:, i-1] + (-a * x2[:, i-1] + f) * dt + sigma * np.sqrt(dt) * norm.rvs(0, 1, Ens)

# Plotting the simulated time series and ensemble members (in the case of the 
# process with random initial values).
fig = plt.figure()
ax = fig.add_subplot(121)
plt.plot(np.linspace(dt, N*dt, N), x1, 'b', linewidth = 2)
ax.set_title('(a) x (Deterministic Initial Condition)', fontsize = 18)
ax.set_xlabel('t', fontsize = 12)
ax.set_ylabel('x', fontsize = 12)
ax.set_xlim(dt, N*dt)
ax.set_ylim(np.min(x2), np.max(x2))
ax = fig.add_subplot(122)
ensembles_plot = plt.plot(np.tile(np.linspace(dt, N*dt, N), [Ens, 1]).T, x2.T, 'k', linewidth = 0.5)
mean_plot, = plt.plot(np.linspace(dt, N*dt, N), np.mean(x2, 0), 'r', linewidth = 2)
ax.legend([ensembles_plot[0], mean_plot], ['Ensemble Members', 'Mean Time Series'], fontsize = 14)
ax.set_title('(b) x (Initial Condition With Small Uncertainty)', fontsize = 18)
ax.set_xlabel('t', fontsize = 12)
ax.set_ylabel('x', fontsize = 12)
ax.set_xlim(dt, N*dt)
ax.set_ylim(np.min(x2), np.max(x2))

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
fig.tight_layout(pad=0, w_pad=-2, h_pad=-2)
plt.show()