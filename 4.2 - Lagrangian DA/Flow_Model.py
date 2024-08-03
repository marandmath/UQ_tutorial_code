# Code corresponding to Section 4.2 - "Lagrangian DA" in the Supplementary 
# Document of the paper "Taming Uncertainty in a Complex World: The Rise of 
# Uncertainty Quantification — A Tutorial for Beginners".
#
# Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
#
# Code: Generating a 2D incompressible flow with random amplitude based on a 
# linear stochastic model (OU process) for the Fourier modes, mimicking the 
# quasi-geostrophic (QG) or potential vortical flow (Python code).
#
# Info: Only the geostrophically balanced (GB) modes are considered in this
# simulation, for simplicity, thus making the underlying velocity field
# incompressible due to the omission of the Poincare/inertio-gravity 
# compressible wave modes. The Fourier wavenumber grid is also plotted along 
# with the order numbers that are being used for each mode in the global system.
# This script file is imported by LDA_Function_of_L.py as to use its variables.
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

K_max = 2 # The range of Fourier modes is [-K_max, K_max]^2∩Z
k = np.zeros((2, (2 * K_max + 1) * (2 * K_max + 1))) # Collection of Fourier wavenumbers over the symmetric discrete lattice

# Arranging the Fourier wavenumbers in such a way that the complex conjugate 
# modes are next to each other, namely (-k1, -k2) will be next to (k1, k2). This 
# will facilitate data assimilation by allowing the structure of the matrix 
# coefficients in the conditional Gaussian nonlinear system framework (which in
# this case consists of the OU process driving the Fourier modes of the flow for
# the GB incompressible waves (Forecast flow model) and the acceleration 
# relation for the tracers' trajectories (Observational process)) to be block 
# diagonal matrices, with each block being a 2 by 2 matrix, while also 
# simplifying the form of the deterministic forcing in the forecast flow model. 
# This then simplifies the enforcement of the reality/conjugate conditions onto 
# the Fourier conjugate wavenumbers, which are required as to establish that the 
# flow velocity is a real-valued vector field.
m = 0
for i in range(-K_max, K_max+1):
    if i < 0:
        for j in range(-K_max, i+1):
            k[0, m] = i
            k[1, m] = j
            m = m + 2
    else:
        for j in range(-K_max, i):
            k[0, m] = i
            k[1, m] = j
            m = m + 2
k[:, 1:-1:2] = - k[:, 0:-2:2]

kk = k[:, :-1] # Remove the (0, 0) mode by assuming no background mean flow/sweep 

# Defining the eigenvectors that contain the relationships between different 
# components of the flow field. E.g. in the case of an incompressible flow field
# the divergence-free condition is reflected in the eigenvector of each Fourier 
# mode. In the case where we consider the inclusion of the compressible waves, 
# we order the eigenvectors for the gravity modes in the following manner (not
# applicable in this case):
# omegak = [
#               omegak_+(1:(2*K_max+1)^2-1), 
#               omegak_-(1:(2*K_max+1)^2-1), 
#               omegak_+(end), 
#               omegak_-(end)
# ];
# In the last column of the compressible eigenvectors rk2 and rk3, is the (0,0) 
# inertio-gravity or Poincare mode, which needs to be dealt with in a different 
# way. This structure simplifies the enforcement of the conjugate/reality 
# condition on the eigenvectors and model parameters, which extends to keeping 
# the flow field real-valued. The correspondence of the eigenvectors to 
# atmospheric modes is the following:
#       rk1: Geostrophically Balanced (GB); rk2: Gravity +; rk3: Gravity -.
# We also mention the degrees of freedom of our system when assuming complex 
# coefficients = ((2*K_max+1)^2+1)/2 (due to the conjugate/reality condition 
# enforced on the conjugate Fourier wavenumbers and the inclusion of only the 
# GB modes).

# Assuming no Poincare/inertio-gravity compressible waves, just the GB modes 
# corresponding to the incompressible flow.
rk1 = np.array([
    1 / np.sqrt(np.power(k[0, :], 2) + np.power(k[1, :], 2) + 1) * (-1j * k[1, :]),
    1 / np.sqrt(np.power(k[0, :], 2) + np.power(k[1, :], 2) + 1) * (1j * k[0, :]) 
])
rk1 = rk1[:, :-1]
rk = rk1

T = 15 # Total length of the simulation in time units
dt = 0.001 # Numerical integration time step
N = round(T/dt) # Total numerical integration steps
Dim_U = kk.shape[1]; # Dimension of system = Number of Fourier modes - Origin = (2*K_max+1)**2-1
u_hat = np.zeros((Dim_U, N), dtype = 'complex'); # Defining all the Fourier modes of the velocity field

# Damping feedback of the GB modes: To preserve the reality of the flow field, 
# it must be that d_k=d_(-k), i.e. the conjugate Fourier mode has the same
# damping coefficient.
d_B = 0.5
# The GB flows are non-divergent, which is embodied in the eigenvector, and so
# the associated phase/rotational speeds or eigenvalues are necessarily equal to 
# zero.
omega_B = 0
# Noise of the GB modes: To preserve the reality of the flow field, it must be 
# that σ_k=σ_(-k), i.e. the conjugate Fourier modes have the same noise 
# coefficient, which is necessarily nonnegative and real.
sigma_B = 0.5

Gamma = (d_B - 1j * omega_B) * np.eye(Dim_U, Dim_U, dtype = 'complex') # Γ (Gamma): Damping and phase coefficients of the Fourier modes' OU processes
f = np.zeros((Dim_U, N-1)) # F: Deterministic and periodic forcing in the Fourier modes' OU processes; Being deterministic (or at least an affine function of the Fourier mode) ensures a Gaussian attractor/equilibrium distribution
f_strength = 0.25 # Strength of the background deterministic force (force is assumed zero for the purposes of the paper)
f_period = 2 # Period of the background deterministic force (force is assumed zero for the purposes of the paper)
Sigma_u_hat = np.zeros((Dim_U, Dim_U), dtype = 'complex') # Σ_u_hat (Sigma_u_hat): Noise matrix of the flow field's Fourier modes
for i in range(0, Dim_U, 2):
    Sigma_u_hat[i, i] = 1 / np.sqrt(2) * sigma_B
    Sigma_u_hat[i+1, i+1] = -1j / np.sqrt(2) * sigma_B
    Sigma_u_hat[i, i+1] = 1j / np.sqrt(2) * sigma_B
    Sigma_u_hat[i+1, i] = 1 / np.sqrt(2) * sigma_B

# Model simulation using Euler-Maruyama for the Fourier modes.
# A stochastic system is utilized for each Fourier mode, which is written in a
# vector form that fits the conditional Gaussian nonlinear system framework in
# the following way:
#      (Forecast Flow Model) du_hat = (-Γ * u_hat + F) * dt + Σ_u_hat * dW_u_hat
dw_u_hat = norm.rvs(0, 1, (Dim_U, N-1)) # Wiener noise values for the simulation
for i in range(1, N):

    t = i * dt # Current time in the simulation

    # Forcing can be either zero (homogeneous linear SDE) or a periodic complex 
    # exponential/plane sinusoidal wave (in general a complex function of some
    # period): To preserve the reality of the flow field, it must be that 
    # f_k=f_(-k)*, where * denotes the complex conjugate, i.e. the conjugate 
    # Fourier mode has complex conjugate forcing. For this simulation we assume
    # zero forcing, but the following can be uncommented in case we want to
    # include a background periodic and deterministic force.
    # f[:, i-1] = f_strength * np.reshape( ...
    #     [
    #         np.exp(1j * 2*np.pi*t/f_period) * np.ones(round(Dim_U/2), dtype = 'complex'),
    #         np.exp(-1j * 2*np.pi*t/f_period) * np.ones(round(Dim_U/2), dtype = 'complex') ...
    #     ], ...
    # (Dim_U, 1), order = 'F').copy(); # Force feedback = 0.25; Force period = 5 time units

    u_hat[:, i] = u_hat[:, i-1] + (-Gamma @ u_hat[:, i-1] + f[:, i-1]) * dt + Sigma_u_hat * np.sqrt(dt) @ dw_u_hat[:, i-1]

def main():

    # Showing the grid (discrete and symmetric around the origin lattice) of 
    # Fourier wavenumbers.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(k[0, 0:-2:2], k[1, 0:-2:2], 'ro', linewidth = 4)
    plt.plot(k[0, 1:-1:2], k[1, 1:-1:2], 'go', linewidth = 4)
    plt.plot(k[0, -1], k[1, -1], 'ko', linewidth = 4)
    for i in range((2 * K_max + 1) * (2 * K_max + 1)):
        plt.text(k[0, i]+0.05, k[1, i]-0.05, str(i), fontsize = 12, fontweight = 'bold')
    ax.set_title(f'Fourier Wavenumbers Over the Discrete Symmetric Lattice in [-{K_max},{K_max}]$^2$ and Their Order in the Global Vector', fontsize = 18)
    ax.set_xlabel('k$_1$', fontsize = 12)
    ax.set_ylabel('k$_2$', fontsize = 12)
    ax.grid()

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
    plt.show()

    # Spatiotemporal reconstruction of the incompressible velocity field.
    Dim_Grid = 25
    xx, yy = np.meshgrid(np.linspace(-np.pi, np.pi, Dim_Grid), np.linspace(-np.pi, np.pi, Dim_Grid))
    x_vec = np.hstack([np.reshape(xx, (-1, 1), order = 'F'), np.reshape(yy, (-1, 1), order = 'F')])

    fig = plt.figure()
    for i in range(1, 7):
        ax = fig.add_subplot(2, 3, i)
        u = np.real(np.exp(1j * x_vec @ kk) @ (u_hat[:, 1000*i] * rk[0, :]))
        v = np.real(np.exp(1j * x_vec @ kk) @ (u_hat[:, 1000*i] * rk[1, :]))
        u = np.reshape(u, (Dim_Grid, Dim_Grid), order='F').copy()
        v = np.reshape(v, (Dim_Grid, Dim_Grid), order='F').copy()
        ax.quiver(xx, yy, u, v, linewidth = 1, color = 'b')
        ax.set_title(f't = {(1000*i)*dt:0.2f}', fontsize = 16)
        ax.set_xlabel('x', fontsize = 12)
        ax.set_ylabel('y', fontsize = 12)
    fig.suptitle('Simulation of the Incompressible Flow Field at 6 Different Time Instants', fontsize = 18)


    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()