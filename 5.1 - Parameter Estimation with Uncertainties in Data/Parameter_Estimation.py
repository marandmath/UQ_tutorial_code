# Code corresponding to Section 5.1 - "Parameter Estimation with Uncertainties 
# in Data" in the Supplementary Document of the paper "Taming Uncertainty in a 
# Complex World: The Rise of Uncertainty Quantification — A Tutorial for 
# Beginners".
#
# Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
#
# Code: Estimating the slope parameter 'a' in the following linear system:
#           xdot = dx/dt = ay,
#           ydot = dy/dt = bx.
# We can regard xdot, ydot, y and x as the four variables which we can use to 
# estimate said parameter (Python code).  
# 
# Info: Only two data points will be considered, so the least-squares solution
# through standard linear regression will be needed; If there was only one data 
# point, then we can simply set a = xdot/y if y is non-zero. We will consider 
# two cases; One case where the data for both xdot and y are available, and one 
# where y is not directly observed and instead it is assigned a Gaussian 
# distribution at each realization which leads to nonlinear uncertainties in the 
# least-squares solution for the parameter.
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
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt

################################################################################
############################## No Uncertainty in y #############################
################################################################################

xdot = np.array([1, 2]) # Data points of dx/dt
y = np.array([1, 3]) # Data points of y
a_est_deterministic = (1 / np.sum(np.power(y, 2))) * np.sum(y * xdot) # Least-squares solution through standard linear regression 

# Plotting the data points against the least-squares solution from linear 
# regression, where the slope of the fitted line is exactly the estimated 
# parameter.
fig = plt.figure()
gs = fig.add_gridspec(2, 4)
ax = fig.add_subplot(gs[0, 0:2])
plt.plot(y, xdot, 'bo', linewidth = 2, label = 'Data Points')
plt.plot(np.array([np.min(y)-1, np.max(y)+1]), a_est_deterministic * np.array([np.min(y)-1, np.max(y)+1]), 'b-', linewidth = 2, label = 'Least-Squares Solution')
ax.legend(fontsize = 14, loc = 'upper left')
ax.set_title('(a) Least-Squares Solution With No Uncertainty in y', fontsize = 18)
ax.set_xlabel('y', fontsize = 12)
ax.set_ylabel('dx/dt', fontsize = 12)
ax.set_xlim(np.min(y)-1, np.max(y)+1)

print(f'Estimated value of the slope parameter a by using least-squares, when y has no uncertainty: {a_est_deterministic:0.4f}')

################################################################################
################### Gaussian Uncertainty in Unobservable y #####################
################################################################################

np.random.seed(42) # Setting the seed number for consistent and reproducible results

r = 10 # Uncertainty level (variance) of the unobservable y (assuming the same variance for each observation), which is <y'^2>
a_est_UQ = (1 / np.sum(np.power(y, 2) + r)) * np.sum(y * xdot) # Least-squares solution through standard linear regression in the presence of uncertainty
print(f'Estimated value of the slope parameter a by using least-squares, when y contains uncertainty (Gaussian with variance r = {r:0.2f}): {a_est_UQ:0.4f}')

# Generating samples from the distribution of the unobservable y.
sample_num = 5000 # Number of samples to be obtained from the distribution of y
y_samples = y.reshape(2, 1) + np.sqrt(r) * norm.rvs(0, 1, (2, sample_num)) # Assuming a Gaussian distribution centered around the true value but with large variance

# Plotting the data points against the least-squares solution from linear 
# regression for both cases (with and without uncertainty), along with the
# sampled least-squares solutions. These samples are also plotted and are
# connected via a cyan line denoting their respective pair.
ax = fig.add_subplot(gs[0, 2:])
h1, = plt.plot(y, xdot, 'bo', linewidth = 2)
ax.set_title('(b) Least-Squares Solution With Uncertainty in y', fontsize = 18)
ax.set_xlabel('y', fontsize = 12)
ax.set_ylabel('dx/dt', fontsize = 12)
a_est_each_sample = np.zeros(sample_num)
xdot_plot_point = 3 # Up to which point in the y-axis (dx/dt = xdot) the fitted lines should be plotted
for i in range(sample_num):
    a_est_each_sample[i] = (1 / np.sum(np.power(y_samples[:, i], 2))) * np.sum(y_samples[:, i] * xdot) # Least-squares solution through standard linear regression in the presence of uncertainty for each sample of y values
    if i <= 100: # Plotting 100 of the sampled least-squares solutions (via the respective fitted line)
        plt.plot([0, xdot_plot_point/a_est_each_sample[i]], [0, xdot_plot_point], color = (0.7, 0.7, 0.7, 1), linestyle = '-')
        plt.plot(y_samples[:, i], xdot, 'k.')
        plt.plot(y_samples[:, i], xdot, 'c-')
h2, = plt.plot([0, xdot_plot_point/a_est_deterministic], [0, xdot_plot_point], 'b-', linewidth = 2)
h3, = plt.plot([0, xdot_plot_point/a_est_UQ], [0, xdot_plot_point], 'g-', linewidth = 2)
h4, = plt.plot([0, xdot_plot_point/a_est_each_sample[i]], [0, xdot_plot_point], color = (0.7, 0.7, 0.7, 1), linestyle = '-')
ax.legend([h1, h2, h3, h4], ['Data Points', 'Least-Squares Solution (Deterministic)', 'Least-Squares Solution (With Uncertainty)', 'Solution of Individual Sampled Points'], fontsize = 14, loc = 'upper right')
ax.set_xlim([np.min(y_samples), np.max(y_samples)])
ax.set_ylim([0, 2*xdot_plot_point])

# Plotting the distribution of the slope parameter, a, using the different 
# samples from y when assuming uncertainty in its observation. Its average 
# value, which is exactly the least-squares solution under uncertainty, is also 
# plotted, along with the least-squares solution of the deterministic case.
ax = fig.add_subplot(gs[1, 1:3])
slope_pdf = gaussian_kde(a_est_each_sample)(np.linspace(-3, 3, 200))
plt.plot(np.linspace(-3, 3, 200), slope_pdf, color = (0.7, 0.7, 0.7, 1), linewidth = 2, label = 'Distribution of a in the Case With Uncertainty')
plt.vlines(a_est_deterministic, 0, 1.1*np.max(slope_pdf), 'b', linewidth = 2, label = 'Least-Squares Solution in Deterministic Case')
plt.vlines(a_est_UQ, 0, 1.1*np.max(slope_pdf), 'g', linewidth = 2, label = 'Mean Least-Squares Solution Under Uncertainty')
ax.legend(fontsize = 14, loc = 'upper right')
ax.set_title('(c) Estimated Slope Parameter a', fontsize = 18)
ax.set_xlabel('a', fontsize = 12)
ax.set_ylabel('Probability Density', fontsize = 12)
ax.set_ylim(0, 1.1*np.max(slope_pdf))

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
plt.show()