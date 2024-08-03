# Code corresponding to Section 5.2 - "Eddy identification" in the Supplementary 
# Document of the paper "Taming Uncertainty in a Complex World: The Rise of 
# Uncertainty Quantification — A Tutorial for Beginners".
#
# Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
#
# Code: Uses the OW parameter as the criterion for identifying eddies in a 2D 
# random amplitude incompressible velocity field constructed by a linear 
# stochastic model (OU process) for the Fourier modes, where the flow field is 
# recovered by passive tracers using Lagrangian data assimilation via the 
# posterior smoother Gaussian distribution (Python code).
#
# Info: This code should be run together with LDA_Main_Filter.py and 
# Flow_Model.py. Both will be automatically called through this script, so no 
# further input from the user is required. Flow_Model.py will be utilised by
# first importing it and then using the necessary variables for this script,
# while LDA_Main_Filter.py will be called using the built-in subprocess module. 
# For the latter script, variables are shared between it (the "calee") and this 
# script (the "caller"), using the built-in pickle module. This requires the 
# creation of temporary pickle files (.pkl), which are all deleted after the 
# termination of this  script, and are constrained within the current script's 
# directory. Pickle is preferred over Numpy's saving variable capabilities due 
# to speed (while at the cost of larger usage of storage, especially for large 
# L). Flow_Model.py is being used to generate an underlying 2D incompressible 
# velocity field via a spectral decomposition which uses local Fourier basis 
# functions for the geostrophically balanced modes by further assigning to them 
# an OU process. On the other hand, LDA_Main_Smoother.py is being used to 
# calculate the optimal posterior distribution (smoother solution), as to 
# showcase the eddy diagnostic at a specific diagnostic time. The adoption of 
# the OW parameter is used as the criterion for identifying eddies; When the OW 
# parameter is negative, the relative vorticity is larger than the strain 
# components, indicating vortical flow. Backward sampled realizations or 
# smoother-based samples of the unobservable Fourier modes are also generated, 
# which while not the same as the truth, are still crucial as to be collected 
# and construct a PDF describing the statistical behavior of the OW parameter. 
# In the presence of uncertainty, such uncertainty quantification in the 
# diagnostics is important since the deterministic solution arising from 
# averaging may lose a crucial amount of information.
#
# Python Script File Requirements (No need to run prior to this script; 
# Variables will be imported automatically):
#   * Flow_Model.py
#   * LDA_Main_Smoother.py
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
import Flow_Model # Generating the background 2D incompressible flow field
import subprocess # Used to call other Python scripts (LDA_Main_Smoother.py in this case)
# "The pickle module implements binary protocols for serializing and 
# de-serializing a Python object structure" - Used to store and load variables 
# in a quick and efficient manner in temporary files (to share between Python 
# scripts).
import pickle 
import os # To store and remove temporary files needed by the scripts
import time
import numpy as np
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Extracting the necessary variables from Flow_Model.py.
dt = Flow_Model.dt
N = Flow_Model.N
kk = Flow_Model.kk
rk = Flow_Model.rk
Dim_U = Flow_Model.Dim_U
Sigma_u_hat = Flow_Model.Sigma_u_hat
Gamma = Flow_Model.Gamma
f = Flow_Model.f
f_strength = Flow_Model.f_strength
f_period = Flow_Model.f_period
u_hat = Flow_Model.u_hat

L = 5 # Number of Lagrangian tracers being deployed in the ocean flow

# Storing the variables from Flow_Model.py and L (number of observations),
# as to be used by LDA_Main_Filter.py in a temporary pickle (.pkl) file; 
# This temporary file is deleted as soon as it is no longer needed
with open('temp_objs_PYTHON_caller.pkl', 'wb') as pkl_file:
    pickle.dump([L, dt, N, Dim_U, kk, rk, Sigma_u_hat, Gamma, f, f_strength, f_period, u_hat], pkl_file)

# Calling LDA_Main_Smoother.py to run, which runs Lagrangian data assimilation
# with a given L and calculates the posterior filter and smoother Gaussian 
# statistics as well as the backward (smoother-based) samples of the 
# unobservable.
subprocess.run(['python', 'LDA_Main_Smoother.py'])

# Loading the variables needed in this script that were generated by
# LDA_Main_Smoother.py.
with open('temp_objs_PYTHON_callee.pkl', 'rb') as pkl_file:
    x, y, u_smoother_mean, u_smoother_cov, s_n, Y_Sampling_Save = pickle.load(pkl_file)

u_post_mean = u_smoother_mean # Posterior smoother mean
u_post_cov = u_smoother_cov # Posterior smoother covariance (keeping only the diagonal of the filter covariance during its calculation)

# Removing the unnecessary temporary pickle files from the user's current 
# directory.
time.sleep(1)
os.remove("temp_objs_PYTHON_caller.pkl")
os.remove("temp_objs_PYTHON_callee.pkl")

t_test = 2 # Diagnostic time used to check the flow field recovery and eddy identification skill
t_index = round(t_test/dt)

# Spatiotemporal reconstruction of the true incompressible velocity field.
Dim_Grid = 25
xx, yy = np.meshgrid(np.linspace(-np.pi, np.pi, Dim_Grid), np.linspace(-np.pi, np.pi, Dim_Grid))
x_vec = np.hstack([np.reshape(xx, (-1, 1), order = 'F'), np.reshape(yy, (-1, 1), order = 'F')])
u = np.real(np.exp(1j * x_vec @ kk) @ (u_hat[:, t_index] * rk[0, :]))
v = np.real(np.exp(1j * x_vec @ kk) @ (u_hat[:, t_index] * rk[1, :]))
u = np.reshape(u, (Dim_Grid, Dim_Grid), order='F').copy()
v = np.reshape(v, (Dim_Grid, Dim_Grid), order='F').copy()
u_truth = u
v_truth = v

fig = plt.figure()
# OW parameter calculation using the truth.
gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, 0])
# Note that in Python, when approximating derivatives via finite differences 
# with the diff() function the rows and columns correspond to the y and x 
# directions, respectively. As such, the derivative of y (x) should be along the 
# row (column) direction!
uy = np.diff(u, 1, 0); uy = (np.vstack((uy[-1, :], uy)) + np.vstack((uy, uy[0, :]))) / 2 # y-derivative of the zonal or latitudinal (x-coordinate) velocity
vy = np.diff(v, 1, 0); vy = (np.vstack((vy[-1, :], vy)) + np.vstack((vy, vy[0, :]))) / 2 # y-derivative of the meridional or longitudinal (y-coordinate) velocity
ux = np.diff(u.conj().T, 1, 0); ux = ux.conj().T; ux = (np.hstack((ux[:, -1].reshape(ux.shape[0], 1), ux)) + np.hstack((ux, ux[:, 0].reshape(ux.shape[0], 1)))) / 2 # x-derivative of the zonal or latitudinal (x-coordinate) velocity
vx = np.diff(v.conj().T, 1, 0); vx = vx.conj().T; vx = (np.hstack((vx[:, -1].reshape(vx.shape[0], 1), vx)) + np.hstack((vx, vx[:, 0].reshape(vx.shape[0], 1)))) / 2 # x-derivative of the meridional or longitudinal (y-coordinate) velocity
# Computing the OW parameter.
sn = ux - vy # Normal strain
ss = vx + uy # Shear strain
w = vx - uy # Relative vorticity (vorticity's z-coordinate/signed magnitude)
OW_truth = np.power(sn, 2) + np.power(ss, 2) - np.power(w, 2)
OW_std = np.std(OW_truth)
OW_truth = OW_truth / OW_std # Normalizing the OW parameter as to render the parameter scale invariant
# Plotting the true OW parameter over the spatial domain.
max_OW_truth = np.max(OW_truth)
min_OW_truth = np.min(OW_truth)
im = ax.imshow(OW_truth, cmap = 'jet', vmin = min_OW_truth, vmax = max_OW_truth, extent = [-np.pi, np.pi, -np.pi, np.pi], interpolation = 'lanczos')
plt.colorbar(im, shrink=0.6)
ax.quiver(xx, yy, u_truth, v_truth, linewidth = 1, color = 'b')
ax.set_title(r'(a) $\mathrm{OW}[\mathbf{u}^{\mathrm{truth}}]$', fontsize = 18)
ax.set_xlabel('x', fontsize = 12)
ax.set_ylabel('y', fontsize = 12)
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)

# OW parameter calculation using the posterior smoother mean.
ax = fig.add_subplot(gs[0, 1])
# Spatiotemporal reconstruction of the incompressible velocity field using the 
# posterior smoother mean.
u = np.real(np.exp(1j * x_vec @ kk) @ (u_post_mean[:, t_index] * rk[0, :]))
v = np.real(np.exp(1j * x_vec @ kk) @ (u_post_mean[:, t_index] * rk[1, :]))
u = np.reshape(u, (Dim_Grid, Dim_Grid), order='F').copy()
v = np.reshape(v, (Dim_Grid, Dim_Grid), order='F').copy()
# Note that in Python, when approximating derivatives via finite differences 
# with the diff() function the rows and columns correspond to the y and x 
# directions, respectively. As such, the derivative of y (x) should be along the 
# row (column) direction!
uy = np.diff(u, 1, 0); uy = (np.vstack((uy[-1, :], uy)) + np.vstack((uy, uy[0, :]))) / 2 # y-derivative of the zonal or latitudinal (x-coordinate) velocity
vy = np.diff(v, 1, 0); vy = (np.vstack((vy[-1, :], vy)) + np.vstack((vy, vy[0, :]))) / 2 # y-derivative of the meridional or longitudinal (y-coordinate) velocity
ux = np.diff(u.conj().T, 1, 0); ux = ux.conj().T; ux = (np.hstack((ux[:, -1].reshape(ux.shape[0], 1), ux)) + np.hstack((ux, ux[:, 0].reshape(ux.shape[0], 1)))) / 2 # x-derivative of the zonal or latitudinal (x-coordinate) velocity
vx = np.diff(v.conj().T, 1, 0); vx = vx.conj().T; vx = (np.hstack((vx[:, -1].reshape(vx.shape[0], 1), vx)) + np.hstack((vx, vx[:, 0].reshape(vx.shape[0], 1)))) / 2 # x-derivative of the meridional or longitudinal (y-coordinate) velocity
# Computing the OW parameter.
sn = ux - vy # Normal strain
ss = vx + uy # Shear strain
w = vx - uy # Relative vorticity (vorticity's z-coordinate/signed magnitude)
OW_post_mean = (np.power(sn, 2) + np.power(ss, 2) - np.power(w, 2)) / OW_std # Normalising with respect to the true standard deviation for uniformity
# Plotting the recovered OW parameter (via the posterior smoother mean) over the 
# spatial domain.
im = ax.imshow(OW_post_mean, cmap = 'jet', vmin = min_OW_truth, vmax = max_OW_truth, extent = [-np.pi, np.pi, -np.pi, np.pi], interpolation = 'lanczos')
plt.colorbar(im, shrink=0.6)
ax.quiver(xx, yy, u_truth, v_truth, linewidth = 1, color = 'b') # Plotting the true velocity field on top of the estimated OW parameter for reference
ax.set_title(r'(b) $\mathrm{OW}(\bar{\mathbf{u}})$', fontsize = 18)
ax.set_xlabel('x', fontsize = 12)
ax.set_ylabel('y', fontsize = 12)
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)

# OW parameter estimation using the expectation from the backward 
# (smoother-based) generated samples of the Fourier modes.
ax = fig.add_subplot(gs[0, 2])
OW_sampling = np.zeros(OW_post_mean.shape)
for s in range(s_n):
    # Spatiotemporal reconstruction of the incompressible velocity field using
    # this specific backward sample.
    u_sampling = Y_Sampling_Save[:, :, s]
    u = np.real(np.exp(1j * x_vec @ kk) @ (u_sampling[:, t_index] * rk[0, :]))
    v = np.real(np.exp(1j * x_vec @ kk) @ (u_sampling[:, t_index] * rk[1, :]))
    u = np.reshape(u, (Dim_Grid, Dim_Grid), order='F').copy()
    v = np.reshape(v, (Dim_Grid, Dim_Grid), order='F').copy()
    # Note that in Python, when approximating derivatives via finite differences 
    # with the diff() function the rows and columns correspond to the y and x 
    # directions, respectively. As such, the derivative of y (x) should be along the 
    # row (column) direction!
    uy = np.diff(u, 1, 0); uy = (np.vstack((uy[-1, :], uy)) + np.vstack((uy, uy[0, :]))) / 2 # y-derivative of the zonal or latitudinal (x-coordinate) velocity
    vy = np.diff(v, 1, 0); vy = (np.vstack((vy[-1, :], vy)) + np.vstack((vy, vy[0, :]))) / 2 # y-derivative of the meridional or longitudinal (y-coordinate) velocity
    ux = np.diff(u.conj().T, 1, 0); ux = ux.conj().T; ux = (np.hstack((ux[:, -1].reshape(ux.shape[0], 1), ux)) + np.hstack((ux, ux[:, 0].reshape(ux.shape[0], 1)))) / 2 # x-derivative of the zonal or latitudinal (x-coordinate) velocity
    vx = np.diff(v.conj().T, 1, 0); vx = vx.conj().T; vx = (np.hstack((vx[:, -1].reshape(vx.shape[0], 1), vx)) + np.hstack((vx, vx[:, 0].reshape(vx.shape[0], 1)))) / 2 # x-derivative of the meridional or longitudinal (y-coordinate) velocity
    # Computing the OW parameter.
    sn = ux - vy # Normal strain
    ss = vx + uy # Shear strain
    w = vx - uy # Relative vorticity (vorticity's z-coordinate/signed magnitude)
    OW_sampling = OW_sampling + (np.power(sn, 2) + np.power(ss, 2) - np.power(w, 2)) # Addidng to the previous result as to take the average at the end
OW_sampling = OW_sampling / s_n # Sample mean which approximates the expectation
OW_sampling = OW_sampling / OW_std # Normalising with respect to the true standard deviation for uniformity
# Plotting the recovered OW parameter (via averaging over the backward 
# (smoother-based) samples of the Fourier modes) over the spatial domain.
im = ax.imshow(OW_sampling, cmap = 'jet', vmin = min_OW_truth, vmax = max_OW_truth, extent = [-np.pi, np.pi, -np.pi, np.pi], interpolation = 'lanczos')
plt.colorbar(im, shrink=0.6)
ax.quiver(xx, yy, u_truth, v_truth, linewidth = 1, color = 'b') # Plotting the true velocity field on top of the estimated OW parameter for reference
ax.set_title(r'(c) $E[\mathrm{OW}(\mathbf{u})]$', fontsize = 18)
ax.set_xlabel('x', fontsize = 12)
ax.set_ylabel('y', fontsize = 12)
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)

# Showing the time series of the desired GB Fourier mode's desired part (both 
# the truth and recovered posterior smoother solution).
ax = fig.add_subplot(gs[1, :])
mode_focus = 2 # Any number between 1 and Dim_U = Dimension of system = Number of Fourier modes - Origin = (2*K_max+1)^2-1 (which mode we want to plot)
real_flag = True # Whether to showcase the real or imaginary part of the desired Fourier mode
part_str = 'Real' if real_flag else 'Imaginary'
true_signal = np.real(u_hat[mode_focus-1, :]) if real_flag else np.imag(u_hat[mode_focus-1, :]) # Time series of the truth for the desired part and Fourier mode
smoother_signal = np.real(u_post_mean[mode_focus-1, :]) if real_flag else np.imag(u_post_mean[mode_focus-1, :]) # Time series of the posterior smoother solution for the desired part and Fourier mode
smoother_std = np.sqrt(np.real(u_post_cov[mode_focus-1, mode_focus-1, :])) # Standard deviation of the posterior smoother solution for the desired part and Fourier mode
plt.plot(np.linspace(dt, N*dt, N), true_signal, 'b', linewidth = 2, label = 'True Signal')
plt.plot(np.linspace(dt, N*dt, N), smoother_signal, 'r', linewidth = 2, label = 'Posterior Mean (Smoother)')
ax.fill_between(np.linspace(dt, N*dt, N), smoother_signal - 2 * smoother_std, smoother_signal + 2 * smoother_std, alpha = 0.2, facecolor = 'r', label = '2 Std Posterior (Smoother)')
plt.plot(t_test, true_signal[round(t_test/dt)], 'go', linewidth = 6, label = 'Diagnostic Time (for OW)')
ax.legend(fontsize = 14)
ax.set_title(f'L = {L} Observations\n{part_str} Part of ({int(kk[0, mode_focus-1]):d}, {int(kk[1, mode_focus-1]):d}) Fourier Mode', fontsize = 16)
ax.set_xlabel('t', fontsize = 12)
ax.set_ylabel('Amplitude', fontsize = 12)
ax.set_xlim(dt, N*dt)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
fig.tight_layout()
plt.show()

# Plotting the recovered OW parameter by using 15 of the backward 
# (smoother-based) samples generated for the Fourier modes.
fig = plt.figure()
for s in range(1, 16):
    ax = fig.add_subplot(3, 5, s)
    # Spatiotemporal reconstruction of the incompressible velocity field using
    # this specific backward sample.
    u_sampling = Y_Sampling_Save[:, :, s]
    u = np.real(np.exp(1j * x_vec @ kk) @ (u_sampling[:, t_index] * rk[0, :]))
    v = np.real(np.exp(1j * x_vec @ kk) @ (u_sampling[:, t_index] * rk[1, :]))
    u = np.reshape(u, (Dim_Grid, Dim_Grid), order='F').copy()
    v = np.reshape(v, (Dim_Grid, Dim_Grid), order='F').copy()
    # Note that in Python, when approximating derivatives via finite differences 
    # with the diff() function the rows and columns correspond to the y and x 
    # directions, respectively. As such, the derivative of y (x) should be along the 
    # row (column) direction!
    uy = np.diff(u, 1, 0); uy = (np.vstack((uy[-1, :], uy)) + np.vstack((uy, uy[0, :]))) / 2 # y-derivative of the zonal or latitudinal (x-coordinate) velocity
    vy = np.diff(v, 1, 0); vy = (np.vstack((vy[-1, :], vy)) + np.vstack((vy, vy[0, :]))) / 2 # y-derivative of the meridional or longitudinal (y-coordinate) velocity
    ux = np.diff(u.conj().T, 1, 0); ux = ux.conj().T; ux = (np.hstack((ux[:, -1].reshape(ux.shape[0], 1), ux)) + np.hstack((ux, ux[:, 0].reshape(ux.shape[0], 1)))) / 2 # x-derivative of the zonal or latitudinal (x-coordinate) velocity
    vx = np.diff(v.conj().T, 1, 0); vx = vx.conj().T; vx = (np.hstack((vx[:, -1].reshape(vx.shape[0], 1), vx)) + np.hstack((vx, vx[:, 0].reshape(vx.shape[0], 1)))) / 2 # x-derivative of the meridional or longitudinal (y-coordinate) velocity
    # Computing the OW parameter.
    sn = ux - vy # Normal strain
    ss = vx + uy # Shear strain
    w = vx - uy # Relative vorticity (vorticity's z-coordinate/signed magnitude)
    OW_sampling = (np.power(sn, 2) + np.power(ss, 2) - np.power(w, 2)) / OW_std # Normalising with respect to the true standard deviation for uniformity
    # Plotting the recovered OW parameter for this specific sample over the 
    # spatial domain.
    im = ax.imshow(OW_sampling, cmap = 'jet', vmin = min_OW_truth, vmax = max_OW_truth, extent = [-np.pi, np.pi, -np.pi, np.pi], interpolation = 'lanczos')
    plt.colorbar(im, shrink=0.6)
    ax.quiver(xx, yy, u_truth, v_truth, linewidth = 1, color = 'b') # Plotting the true velocity field on top of the estimated OW parameter for reference
    ax.set_title(f'Sampled realization: {s}', fontsize = 14)
    ax.set_xlabel('x', fontsize = 12)
    ax.set_ylabel('y', fontsize = 12)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
plt.show()
