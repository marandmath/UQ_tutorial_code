# Code corresponding to Section 4.2 - "Lagrangian DA" in the Supplementary 
# Document of the paper "Taming Uncertainty in a Complex World: The Rise of 
# Uncertainty Quantification — A Tutorial for Beginners".
#
# Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
#
# Code: Lagrangian data assimilation for calculating and displaying the 
# uncertainty reduction (relative entropy) as a function of the number of 
# tracers, L, between the Gaussian posterior filter solution and the Gaussian 
# equilibrium or statistical attractor, also known as the climatological 
# prior distribution (Python code).
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
# L). LDA_Main_Filter.py is being used to calculate the optimal posterior 
# distribution (filter solution), as to calculate the relative entropy between 
# it and the equilibrium distribution or statistical attractor of the system 
# (also known as the climatological prior distribution). This showcases the 
# logarithmic growth of the dispersion part of the relative entropy as a 
# function of the number of tracers, L. The time series of a desired GB Fourier 
# mode's part (real or imaginary), with its two standard deviations, along with 
# the true signal, as well as the true and recovered velocity field's magnitude 
# (at some desired time point), are also plotted.
#
# Python Script File Requirements (No need to be run prior to this script; 
# Variables will be imported automatically):
#   * Flow_Model.py
#   * LDA_Main_Filter.py
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
import subprocess # Used to call other Python scripts (LDA_Main_Filter.py in this case)
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

RE_s = np.zeros(8) # Storing the signal part of the relative entropy for each number of tracers (L)
RE_d = np.zeros(8) # Storing the dispersion part of the relative entropy for each number of tracers (L)
LL = [2, 5, 10, 20, 30, 50, 100, 200] # Number of tracers (L) in each test

# Spatiotemporal reconstruction of the true incompressible velocity field.
time_focus = 10000*dt # Time for which to showcase the recovered and true velocity fields
Dim_Grid = 25
xx, yy = np.meshgrid(np.linspace(-np.pi, np.pi, Dim_Grid), np.linspace(-np.pi, np.pi, Dim_Grid))
x_vec = np.hstack([np.reshape(xx, (-1, 1), order = 'F'), np.reshape(yy, (-1, 1), order = 'F')])
u_truth = np.real(np.exp(1j * x_vec @ kk) @ (u_hat[:, round(time_focus/dt)] * rk[0, :]))
v_truth = np.real(np.exp(1j * x_vec @ kk) @ (u_hat[:, round(time_focus/dt)] * rk[1, :]))
u_truth = np.reshape(u_truth, (Dim_Grid, Dim_Grid), order = 'F').copy()
v_truth = np.reshape(v_truth, (Dim_Grid, Dim_Grid), order = 'F').copy()
magnitude_truth = np.sqrt(np.power(u_truth, 2) + np.power(v_truth, 2))
max_magnitude_truth = np.max(magnitude_truth) 
min_magnitude_truth = np.min(magnitude_truth)

fig = plt.figure()
mode_focus = 6 # Any number between 1 and Dim_U = Dimension of system = Number of Fourier modes - Origin = (2*K_max+1)^2-1 (which mode we want to plot)
real_flag = True # Whether to showcase the real or imaginary part of the desired Fourier mode
part_str = 'Real' if real_flag else 'Imaginary'

for p in range(1, len(LL)+1):

    L = LL[p-1] # Number of Lagrangian tracers being deployed in the ocean flow

    # Storing the variables from Flow_Model.py and L (number of observations),
    # as to be used by LDA_Main_Filter.py in a temporary pickle (.pkl) file; 
    # This temporary file is deleted as soon as it is no longer needed.
    with open('temp_objs_PYTHON_caller.pkl', 'wb') as pkl_file:
        pickle.dump([L, dt, N, Dim_U, kk, rk, Sigma_u_hat, Gamma, f, f_strength, f_period, u_hat], pkl_file)

    # Calling LDA_Main_Filter.py to run, which runs Lagrangian data assimilation 
    # with a given L and calculates the posterior filter mean, covariance, and 
    # information gain beyond the equilibrium distribution.
    subprocess.run(['python', 'LDA_Main_Filter.py'])

    # Loading the variables needed in this script that were generated by
    # LDA_Main_Filter.py.
    with open('temp_objs_PYTHON_callee.pkl', 'rb') as pkl_file:
        x, y, u_post_mean, u_post_cov_diag, Relative_Entropy_Signal_All, Relative_Entropy_Dispersion_All = pickle.load(pkl_file)

    # Removing the unnecessary temporary pickle files from the user's current 
    # directory.
    time.sleep(1)
    os.remove("temp_objs_PYTHON_caller.pkl")
    os.remove("temp_objs_PYTHON_callee.pkl")

    RE_s[p-1] = Relative_Entropy_Signal_All # Relative entropy in signal part
    RE_d[p-1] = Relative_Entropy_Dispersion_All # Relative entropy in dispersion part
    
    if p % 2 == 1: # Only showing the results from every other test

        true_signal = np.real(u_hat[mode_focus-1, :]) if real_flag else np.imag(u_hat[mode_focus-1, :]) # Time series of the truth for the desired part and Fourier mode
        filter_signal = np.real(u_post_mean[mode_focus-1, :]) if real_flag else np.imag(u_post_mean[mode_focus-1, :]) # Time series of the posterior filter solution for the desired part and Fourier mode
        filter_std = np.sqrt(u_post_cov_diag[mode_focus-1, :]) # Standard deviation of the posterior filter solution for the desired part and Fourier mode

        ax = fig.add_subplot(2, 4, round((p+1)/2)) # Showing the time series of the desired GB Fourier mode's desired part (both the truth and recovered posterior filter solution)
        plt.plot(np.linspace(dt, N*dt, N), true_signal, 'b', linewidth = 2, label = 'True Signal')
        plt.plot(np.linspace(dt, N*dt, N), filter_signal, 'r', linewidth = 2, label = 'Posterior Mean (Filter)')
        ax.fill_between(np.linspace(dt, N*dt, N), filter_signal - 2 * filter_std, filter_signal + 2 * filter_std, alpha = 0.2, facecolor = 'r', label = '2 Std Posterior (Filter)')
        if p == 7:
            plt.legend(fontsize = 14)
        ax.set_title(f'L = {L} Observations\n{part_str} Part of ({int(kk[0, mode_focus-1]):d}, {int(kk[1, mode_focus-1]):d}) Fourier Mode', fontsize = 14)
        ax.set_xlabel('t', fontsize = 12)
        ax.set_ylabel('Amplitude', fontsize = 12)

        if p <= 5:
            ax = fig.add_subplot(2, 4, round(4+(p+1)/2)) # Plotting the spatiotemporal reconstruction of the incompressible velocity field at a fixed time instant for the recovered posterior filter solution for various L values
            u = np.real(np.exp(1j * x_vec @ kk) @ (u_post_mean[:, round(time_focus/dt)] * rk[0, :]))
            v = np.real(np.exp(1j * x_vec @ kk) @ (u_post_mean[:, round(time_focus/dt)] * rk[1, :]))
            u = np.reshape(u, (Dim_Grid, Dim_Grid), order = 'F').copy()
            v = np.reshape(v, (Dim_Grid, Dim_Grid), order = 'F').copy()
            magnitude_data = np.sqrt(np.power(u, 2) + np.power(v, 2))
            im = ax.imshow(magnitude_data, cmap = 'jet', vmin = min_magnitude_truth, vmax = max_magnitude_truth, extent = [-np.pi, np.pi, -np.pi, np.pi], interpolation = 'lanczos')
            plt.colorbar(im, shrink=0.6)
            ax.contour(xx, yy, magnitude_data, 20, colors = 'k', extent = [-np.pi, np.pi, -np.pi, np.pi], linewidths = 0.5)
            ax.quiver(xx, yy, u, v, linewidth = 1, color = 'r')
            ax.plot(x[:, round(time_focus/dt)], y[:, round(time_focus/dt)], 'ko', linewidth = 6)
            ax.set_title(f'Recovered (Velocity Magnitude)\nL = {L} Observations: t = {time_focus:0.2f}', fontsize = 14)
            ax.set_xlabel('x', fontsize = 12)
            ax.set_ylabel('y', fontsize = 12)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)

        if p == 7:
            ax = fig.add_subplot(2, 4, round(4+(p+1)/2)) # Plotting the spatiotemporal reconstruction of the incompressible velocity field at a fixed time instant for the true flow field
            im = ax.imshow(magnitude_truth, cmap = 'jet', vmin = min_magnitude_truth, vmax = max_magnitude_truth, extent = [-np.pi, np.pi, -np.pi, np.pi], interpolation = 'lanczos')
            plt.colorbar(im, shrink=0.6)
            ax.contour(xx, yy, magnitude_truth, 20, colors = 'k', extent = [-np.pi, np.pi, -np.pi, np.pi], linewidths = 0.5)
            ax.quiver(xx, yy, u_truth, v_truth, linewidth = 1, color = 'r')
            ax.set_title(f'Truth (Velocity Magnitude): t = {time_focus:0.2f}', fontsize = 14)
            ax.set_xlabel('x', fontsize = 12)
            ax.set_ylabel('y', fontsize = 12)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
plt.show()

# Plotting the information gain (uncertainty reduction) through its signal and
# dispersion parts when expressed as functions of L.
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(LL, RE_s, '-ob', linewidth = 2, label = 'Signal')
ax1.set_ylabel('Bits', fontsize = 12, color = 'b')
ax1.set_xlim(LL[0], LL[-1])
ax1.tick_params(axis = 'y', labelcolor = 'b')
ax2 = ax1.twinx()
ax2.plot(LL, RE_d, '-om', linewidth = 2, label = 'Dispersion')
ax2.plot(LL, Dim_U*np.log(LL)/3, '--k', linewidth = 2, label = 'O(ln(L))')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, fontsize = 14, loc = 'lower right')
ax2.set_title(f'Uncertainty Reduction (Information Gain) as a Function of L (Number of Observations) Beyond the Equilibrium Distribution', fontsize = 18)
ax1.set_xlabel('L (Number of Tracers)', fontsize = 12)
ax2.set_ylabel('Bits', fontsize = 12, color = 'm')
ax2.tick_params(axis = 'y', labelcolor = 'm')

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
plt.show()