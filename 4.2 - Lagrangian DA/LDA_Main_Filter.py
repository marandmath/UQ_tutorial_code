# Code corresponding to Section 4.2 - "Lagrangian DA" in the Supplementary 
# Document of the paper "Taming Uncertainty in a Complex World: The Rise of 
# Uncertainty Quantification — A Tutorial for Beginners".
#
# Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
#
# Code: Calculating the posterior filter mean and covariance for the Fourier 
# modes of the velocity field. Only the diagonal part of the filter covariance 
# matrix is being kept for the calculation, which is made possible by the 
# incompressibility of the fluid, with this approximation being more accurate as 
# L grows larger (Python code).
#
# Info: This script is called via LDA_Function_of_L.py. It calculates the tracer
# locations through the recovered velocity field via the inverse Fourier 
# transform linear operator, using a noisy version of the acceleration relation 
# while incorporating periodic boundary conditions in [-π,π]. It then uses the 
# tracer locations to calculate the posterior mean and variance in an optimal 
# manner through the filtering solution (this code uses filtering, though a 
# better way is to use smoothing), and then calculates the relative entropy 
# between the Gaussian filter solution and the Gaussian statistical attractor or
# equilibrium distribution of the Fourier modes, known as the information gain. 
# We mention also that we only keep the diagonal of the recovered covariance 
# matrix of the optimal filter posterior distribution since it is known that 
# for L (i.e. number of observations) large then this matrix will converge to a 
# diagonal one for incompressible fluids due to the small Rossby numbers, which 
# is the case here for this experiment since we only consider the 
# geostrophically balanced Fourier modes in the spectral decomposition of the 
# velocity field via the local Fourier basis of planar exponential waves. A code
# snippet is included (but commented out), which can be used to calculate the 
# posterior smoother solution as well for better recovery of the signal.
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
# "The pickle module implements binary protocols for serializing and 
# de-serializing a Python object structure" - Used to store and load variables 
# (to share between Python scripts).
import pickle
import numpy as np
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt

with open('temp_objs_PYTHON_caller.pkl', 'rb') as pkl_file:
        L, dt, N, Dim_U, kk, rk, Sigma_u_hat, Gamma, f, f_strength, f_period, u_hat = pickle.load(pkl_file)

np.random.seed(37) # Setting the seed number for consistent and reproducible results
sigma_xy = 0.01 # Noise in the Lagrangian tracer equations

# For convenience we order the components of the tracers in the observation
# vector in the following manner: First we put the x-coordinates of the L
# tracers, and then the y-coordinates by vertical concatenation, as such
# totalling a 2L-dimensional vector.
x = np.zeros((L, N)); # Zonal or latitudinal coordinate of the tracers (x)
y = np.zeros((L, N)); # Meridional or longitudinal coordinate of the tracers (y)

# Using a uniform distribution as an initial condition which is the
# equilibrium (or climatological prior) distribution of the tracers' locations 
# in the case of an incompressible fluid.
x[:, 0] = uniform.rvs(-np.pi, 2*np.pi, L) 
y[:, 0] = uniform.rvs(-np.pi, 2*np.pi, L)

P = np.zeros((2*L, Dim_U, N), dtype = 'complex') # The linear and discrete inverse Fourier operator used in the observational process to recover the underlying velocity field from the Fourier modes
P[:L, :, 0] = np.exp(1j * x[:, 0].reshape(L, 1) @ kk[0, :].reshape(1, Dim_U) + 1j * y[:, 0].reshape(L, 1) @ kk[1, :].reshape(1, Dim_U)) * (np.ones((L, 1), dtype = 'complex') @ rk[0, :].reshape(1, Dim_U))
P[L:, :, 0] = np.exp(1j * x[:, 0].reshape(L, 1) @ kk[0, :].reshape(1, Dim_U) + 1j * y[:, 0].reshape(L, 1) @ kk[1, :].reshape(1, Dim_U)) * (np.ones((L, 1), dtype = 'complex') @ rk[1, :].reshape(1, Dim_U))

# Model simulation using Euler-Maruyama for the tracers' locations.
# A stochastic system is utilized for each tracer's location, which is written 
# in a vector form that fits the conditional Gaussian nonlinear system 
# framework in the following way:
#      (Observational Processes) dx = P(x) * u_hat * dt + Σ_x * dW_x
dw_xy = norm.rvs(0, 1, (2*L, N-1)); # Wiener noise values for the simulation
for i in range(1, N):

    # Generating the tracer locations.
    # x[:, i] = np.real(x[:, i-1] + np.exp(1j * x[:, i-1].reshape(L, 1) @ kk[0, :].reshape(1, Dim_U) + 1j * y[:, i-1].reshape(L, 1) @ kk[1, :].reshape(1, Dim_U)) @ (u_hat[:, i-1] * rk[0,:]) * dt + sigma_xy * np.sqrt(dt) * dw_xy[:L, i-1])
    # y[:, i] = np.real(y[:, i-1] + np.exp(1j * x[:, i-1].reshape(L, 1) @ kk[0, :].reshape(1, Dim_U) + 1j * y[:, i-1].reshape(L, 1) @ kk[1, :].reshape(1, Dim_U)) @ (u_hat[:, i-1] * rk[1,:]) * dt + sigma_xy * np.sqrt(dt) * dw_xy[L:, i-1])
    x[:, i] = np.real(x[:, i-1] + P[:L, :, i-1] @ u_hat[:, i-1] * dt + sigma_xy * np.sqrt(dt) * dw_xy[:L, i-1])
    y[:, i] = np.real(y[:, i-1] + P[L:, :, i-1] @ u_hat[:, i-1] * dt + sigma_xy * np.sqrt(dt) * dw_xy[L:, i-1])

    # Accounting for the periodic boundary conditions in [-π,π] x [-π,π].
    x[:, i] = np.mod(x[:, i] + np.pi, 2*np.pi) - np.pi
    y[:, i] = np.mod(y[:, i] + np.pi, 2*np.pi) - np.pi

    # P: Observational coefficient matrix (inverse Fourier operator).
    P[:L, :, i] = np.exp(1j * x[:, i].reshape(L, 1) @ kk[0, :].reshape(1, Dim_U) + 1j * y[:, i].reshape(L, 1) @ kk[1, :].reshape(1, Dim_U)) * (np.ones((L, 1), dtype = 'complex') @ rk[0, :].reshape(1, Dim_U))
    P[L:, :, i] = np.exp(1j * x[:, i].reshape(L, 1) @ kk[0, :].reshape(1, Dim_U) + 1j * y[:, i].reshape(L, 1) @ kk[1, :].reshape(1, Dim_U)) * (np.ones((L, 1), dtype = 'complex') @ rk[1, :].reshape(1, Dim_U))

# Auxiliary matrices used in the filtering formulae.
S_xyoS_xy_inv = np.eye(2*L, 2*L)/sigma_xy**2 # Grammian of the observational or tracer noise
S_u_hatoS_u_hat = Sigma_u_hat @ Sigma_u_hat.conj().T # Grammian of the flow field's noise feedback

# Provided there is no forcing, the equilibrium/attractor mean is no longer 
# periodic with same period just like the forcing, and instead collapses down to 
# the zero vector, while the covariance tensor has no dependence on the 
# deterministic force regardless of its form.
eq_mean = np.hstack((np.zeros((Dim_U, 1), dtype = 'complex'), f)) / (np.diag(Gamma).reshape(Dim_U, 1) + np.reshape(
            [
                1j * 2*np.pi/f_period * np.ones(round(Dim_U/2), dtype = 'complex'),
                - 1j * 2*np.pi/f_period * np.ones(round(Dim_U/2), dtype = 'complex')
            ],
            (Dim_U, 1), order = 'F').copy()
        ) # This is for the case where there is a periodic forcing in the flow model (if f = 0 then the equilibrium mean is zero)
eq_cov_full = np.linalg.inv(2 * Gamma) @ S_u_hatoS_u_hat
eq_cov_diag = np.diag(np.diag(eq_cov_full))

# Quantify the uncertainty reduction using relative entropy. This measures the 
# information gain of the filter Gaussian statistics beyond the equilibrium 
# statistics of the Gaussian attractor.
Relative_Entropy_Signal = np.zeros(N)
Relative_Entropy_Dispersion = np.zeros(N)

# Posterior filter solution (smoothing can also be used after the posterior
# filter solution has been recovered, where the code for the smoothing part can 
# be found below but is left commented out).
u_post_mean = np.zeros((Dim_U, N), dtype = 'complex') # Posterior filter mean of the Fourier modes vector
u_post_cov_full = np.zeros((Dim_U, Dim_U, N), dtype = 'complex') # Posterior covariance matrix of the Fourier modes vector; Here we keep the whole covariance matrix, not just the diagonal
u_post_cov_diag = np.zeros((Dim_U, N)) # Posterior covariance matrix of the Fourier modes vector; Keeping only the diagonal due to the incompressibility (becomes better as an approximation as L grows)

mu0 = u_hat[:, 0].reshape(Dim_U, 1) # Initial value of the posterior filter mean
R0 = 0.0001 * np.eye(Dim_U, Dim_U, dtype = 'complex') # Initial value of the posterior filter covariance matrix (choosing a positive definite matrix as to preserve positive-definiteness of the posterior filter and smoother covariance matrices)
diag_R0 = np.diag(R0) # Initial value of the posterior filter covariance matrix's diagonal
u_post_mean[:, 0] = mu0.ravel()
u_post_cov_full[:, :, 0] = R0
u_post_cov_diag[:, 0] = np.real(diag_R0)
for i in range(1, N):

    # dx term which is used to define the innovation or measurement pre-fit 
    # residual which is then multiplied by the optimal gain matrix as to obtain 
    # the optimal and refined a-posteriori state estimation.
    x_diff = x[:, i] - x[:, i-1]
    y_diff = y[:, i] - y[:, i-1]
    # Accounting for the periodic boundary conditions in [-π,π] x [-π,π].
    x_diff[x_diff > np.pi]  = x_diff[x_diff > np.pi]  - 2 * np.pi
    x_diff[x_diff < -np.pi] = x_diff[x_diff < -np.pi] + 2 * np.pi
    y_diff[y_diff > np.pi]  = y_diff[y_diff > np.pi]  - 2 * np.pi
    y_diff[y_diff < -np.pi] = y_diff[y_diff < -np.pi] + 2 * np.pi
   
    # Update the posterior filter mean and posterior filter covariance using the 
    # discrete update formulas.
    mu = mu0 + (f[:, i-1].reshape(Dim_U, 1) - Gamma @ mu0) * dt + (R0 @ P[:, :, i-1].conj().T) @ S_xyoS_xy_inv @ (np.vstack((x_diff.reshape(L, 1), y_diff.reshape(L, 1))) - P[:, :, i-1] @ mu0 * dt)
    R = R0 + (-Gamma @ R0 - R0 @ Gamma.conj().T + S_u_hatoS_u_hat - (R0 @ P[:, :, i-1].conj().T) @ S_xyoS_xy_inv @ (P[:, :, i-1] @ R0)) * dt
    u_post_mean[:, i] = mu.ravel()
    u_post_cov_full[:, :, i] = R
    u_post_cov_diag[:, i] = np.real(np.diag(R))
    mu0 = mu
    R0 = R

    # Computing the information gain via relative entropy between the posterior 
    # filter solution and the equilibrium or climatological prior distribution
    # of the Fourier modes (both Gaussian for this system, so the 
    # signal-dispersion decomposition formula is utilized).
    Relative_Entropy_Signal[i] = np.real(1/2 * (mu - eq_mean[:, i-1].reshape(Dim_U, 1)).conj().T @ np.linalg.inv(eq_cov_full) @ (mu - eq_mean[:, i-1].reshape(Dim_U, 1)))
    Relative_Entropy_Dispersion[i] = np.real(1/2 * (np.trace( np.real(R) @ np.linalg.inv(eq_cov_full) ) - Dim_U - np.log( np.linalg.det( np.real(R) @ np.linalg.inv(eq_cov_full) ) )))

################################################################################
####### Use Ctrl+/ (for VS Code) on the following selection to uncomment #######
# # Posterior smoother solution, also Gaussian (see also LDA_Main_Smoother.py,
# # the Python script which is used for Section 5.2 - "Eddy identification" in 
# # the Supplementary Document).
# u_smoother_mean = np.zeros((Dim_U, N), dtype = 'complex') # Posterior smoother mean of the Fourier modes vector
# u_smoother_cov = np.zeros((Dim_U, Dim_U, N), dtype = 'complex') # Posterior smoother covariance matrix of the Fourier modes vector; Here we keep the whole filter covariance matrix, not just the diagonal
#
# # Quantify the uncertainty reduction using relative entropy. This measures the 
# # information gain of the smoother Gaussian statistics beyond the equilibrium 
# # statistics of the Gaussian attractor.
# Relative_Entropy_Signal_Smoother = np.zeros(N)
# Relative_Entropy_Dispersion_Smoother = np.zeros(N)
#
# # Smoothing runs backwards; "Intial" values for smoothing (at the last time 
# # instant) are exactly the filter estimates.
# muT = u_post_mean[:, -1] # "Initial" value of the posterior smoother mean
# RT = u_post_cov_full[:, :, -1] # "Initial" value of the posterior smoother covariance matrix
# u_smoother_mean[:, -1] = muT
# u_smoother_cov[:, :, -1] = RT
# C_jj_matrices = np.zeros((Dim_U, Dim_U, N), dtype = 'complex') # Auxiliary matrix used in the smoother formula
# # C_jj_matrices[:, :, -1] = np.eye(Dim_U, Dim_U, dtype = 'complex') - (-Gamma + S_u_hatoS_u_hat @ np.linalg.inv(RT)) * dt # Up to O(Δt) 
# C_jj_matrices[:, :, -1] = RT @ (np.eye(Dim_U, Dim_U, dtype = 'complex') - Gamma * dt).conj().T @ np.linalg.inv(S_u_hatoS_u_hat * dt + (np.eye(Dim_U, Dim_U, dtype = 'complex') - Gamma * dt) @ RT @ (np.eye(Dim_U, Dim_U, dtype = 'complex') - Gamma * dt).conj().T)
#
# for i in range(N-2, -1, -1):
#
#     # Update the posterior smoother mean vector and posterior smoother 
#     # covariance tensor using the discrete update formulas after keeping only 
#     # the O(Δt) order terms for efficiency.
#     # C_jj = np.eye(Dim_U, Dim_U, dtype = 'complex') - (-Gamma + S_u_hatoS_u_hat @ np.linalg.inv(u_post_cov_full[:, :, i])) * dt # Up to O(Δt) 
#     C_jj = u_post_cov_full[:, :, i] @ (np.eye(Dim_U, Dim_U, dtype = 'complex') - Gamma * dt).conj().T @ np.linalg.inv(S_u_hatoS_u_hat * dt + (np.eye(Dim_U, Dim_U, dtype = 'complex') - Gamma * dt) @ u_post_cov_full[:, :, i] @ (np.eye(Dim_U, Dim_U, dtype = 'complex') - Gamma * dt).conj().T)    
#     C_jj_matrices[:, :, i] = C_jj
#     mu = u_post_mean[:, i] + C_jj @ (muT - f[:, i] * dt - (np.eye(Dim_U, Dim_U, dtype = 'complex') - Gamma * dt) @ u_post_mean[:, i]) # Up to O(Δt) 
#     R = u_post_cov_full[:, :, i] + C_jj @ (RT @ C_jj.conj().T - (np.eye(Dim_U, Dim_U, dtype = 'complex') - Gamma * dt) @ u_post_cov_full[:, :, i]) # Up to O(Δt) 
#     u_smoother_mean[:, i] = mu
#     u_smoother_cov[:, :, i] = R
#     muT = mu
#     RT = R
#
#     # Computing the information gain via relative entropy between the posterior 
#     # filter solution and the equilibrium or climatological prior distribution
#     # of the Fourier modes (both Gaussian for this system, so the 
#     # signal-dispersion decomposition formula is utilized).
#     Relative_Entropy_Signal_Smoother[i] = np.real(1/2 * (mu.reshape(Dim_U, 1) - eq_mean[:, i-1].reshape(Dim_U, 1)).conj().T @ np.linalg.inv(eq_cov_full) @ (mu.reshape(Dim_U, 1) - eq_mean[:, i-1].reshape(Dim_U, 1)))
#     Relative_Entropy_Dispersion_Smoother[i] = np.real(1/2 * (np.trace( np.real(R) @ np.linalg.inv(eq_cov_full) ) - Dim_U - np.log( np.linalg.det( np.real(R) @ np.linalg.inv(eq_cov_full) ) )))
################################################################################

# Calculating the mean of the signal and dispersion parts of the information 
# gain with respect to time for this value of L, while accounting for some 
# burn-in period initially.
Relative_Entropy_Signal_All = np.mean(Relative_Entropy_Signal[1000:])
Relative_Entropy_Dispersion_All = np.mean(Relative_Entropy_Dispersion[1000:])
# Relative_Entropy_Signal_All = np.mean(Relative_Entropy_Signal_Smoother[1000:])
# Relative_Entropy_Dispersion_All = np.mean(Relative_Entropy_Dispersion_Smoother[1000:])

with open('temp_objs_PYTHON_callee.pkl', 'wb') as pkl_file:
    pickle.dump([x, y, u_post_mean, u_post_cov_diag, Relative_Entropy_Signal_All, Relative_Entropy_Dispersion_All], pkl_file)