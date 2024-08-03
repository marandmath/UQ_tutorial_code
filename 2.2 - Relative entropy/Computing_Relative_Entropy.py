# Code corresponding to Section 2.2 - "Relative entropy" in the Supplementary 
# Document of the paper "Taming Uncertainty in a Complex World: The Rise of 
# Uncertainty Quantification — A Tutorial for Beginners".
#
# Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
# 
# Code: Computing the relative entropy for Gaussian distributions (Python code).
#
# Info: Results from the analytic formula and numerical integration will be 
# compared for the relative entropy between two Gaussian distributions; One will
# be considered the ground truth and the other one will be the model 
# distribution. Two cases will be considered; One where both distributions are 
# known, and one where they are both unknown and instead constructed via 
# samples. Special caution is required when dealing with the tail values of the 
# distribution appearing in the denominator of the relative entropy's definition 
# (as well as the reference/true distribution for consistency).
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
from scipy.stats import norm, multivariate_normal, gaussian_kde
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from copy import copy

################################################################################
############### Gaussian Distribution Given by Analytic Formulae ###############
################################################################################

# Defining the reference/ground truth and model Gaussian distributions.
m = 1 # Dimension of the random variable
x = np.linspace(-10, 10, 500) # Domain of the random variable
mu = 0; R = 1 # Mean and variance of reference/ground truth distribution (R is the covariance matrix for m > 1)
# Leading two moments also obtained via norm.stats(mu, np.sqrt(R), moments='mv').
print(f'Mean of p = {mu:.4f} | Variance of p = {R:.4f}') # Displaying the leading two moments
p = 1/np.sqrt( 2 * np.pi * R ) * np.exp( -(x - mu)**2 / 2 / R) # Computing the PDF of p over the domain (p = norm.pdf(x, mu, np.sqrt(R)), or p = multivariate_normal.pdf(x, mu, R) for m > 1 with R being the covariance matrix)
mu_M = 3; R_M = 1 # Mean and variance of the model distribution (R_M is the covariance matrix for m > 1)
# Leading two moments also obtained via norm.stats(mu_M, np.sqrt(R_M), moments='mv').
print(f'Mean of p^M = {mu_M:.4f} | Variance of p^M = {R_M:.4f}') # Displaying the leading two moments
p_M = 1/np.sqrt( 2 * np.pi * R_M ) * np.exp( -(x - mu_M)**2 / 2 / R_M) # Computing the PDF of pᴹ over the domain (p_M = norm.pdf(x, mu_M, np.sqrt(R_M)), or p = multivariate_normal.pdf(x, mu_M, R_M) for m > 1 with R_M being the covariance matrix)

# Plotting the PDFs defined by the analytic formulae.
fig = plt.figure()
ax = fig.add_subplot(221)
plt.plot(x, p, 'b', linewidth = 2, label = 'p(x)')
plt.plot(x, p_M, 'r', linewidth = 2, label = 'p$^M$(x)')
ax.legend(fontsize = 14, loc = 'upper left')
ax.set_title('(a) p(x) Computed Analytically', fontsize = 18)
ax.set_xlabel('x', fontsize = 12)
ax.set_ylabel('Probability Density', fontsize = 12)
ax.set_ylim(0, 1.1*max([np.max(p), np.max(p_M)]))
ax = fig.add_subplot(223) # Plotting the PDFs on a logarithmic scale to better see the tail behavior 
plt.plot(x, p, 'b', linewidth = 2)
plt.plot(x, p_M, 'r', linewidth = 2)
plt.yscale('log')
ax.set_title('Y-axis in Logarithmic Scale', fontsize = 18)
ax.set_xlabel('x', fontsize = 12)
ax.set_ylabel('Probability Density', fontsize = 12)
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(1e-20, 1)

# Computing the relative entropy using the signal-dispersion decomposition
# formula since both distributions are Gaussian.
if m == 1:
    R_M_inv = 1 / R_M
    Signal = 1/2 * (mu - mu_M) * R_M_inv * (mu - mu_M)
    Dispersion = 1/2 * (R * R_M_inv - m - np.log( R * R_M_inv ))
else:
    R_M_inv = np.linalg.inv(R_M)
    Signal = 1/2 * (mu - mu_M).conj().T @ R_M_inv @ (mu - mu_M)
    Dispersion = 1/2 * (np.trace( R @ R_M_inv ) - m - np.log( np.linalg.det( R @ R_M_inv ) ))
RE_theoretic = Signal + Dispersion

# Computing the relative entropy using the definition and numerical integration,
# where no consideration is made about the tails of the model distribution in
# the denominator (or the one corresponding to the truth).
RE_numerical_no_normalization = trapezoid(p * np.log( p / p_M ), x)

# Computing the relative entropy using the definition and numerical integration,
# where the tail probability is set to be a small but nonzero value to account
# for the PDF ratio, and then a corrective normalization is applied to the PDFs.
p_no_remedy = copy(p) # This notation is needed due to Python's reference caveats (we need to use copy.copy() to create a shallow copy of the variable in question)
p_M_no_remedy = copy(p_M) # This notation is needed due to Python's reference caveats (we need to use copy.copy() to create a shallow copy of the variable in question)
p[ p <= 1e-5 ] = 1e-5
p_M[ p_M <= 1e-5 ] = 1e-5
p = p / trapezoid(p, x)
p_M = p_M / trapezoid(p_M, x)
RE_numerical_with_normalization = trapezoid(p * np.log( p / p_M ), x)

print(f'\nGaussian distributions given by analytic formulae')
print(f'Relative entropy:\nTheoretical value: {RE_theoretic:.4f}\nNumerical value without normalization: {RE_numerical_no_normalization:.4f}\nNumerical value with normalization: {RE_numerical_with_normalization:.4f}\n')

################################################################################
################ Gaussian Distribution Constructed From Samples ################
################################################################################

np.random.seed(42) # Setting the seed number for consistent and reproducible results
sample_number = 10000 # Number of samples to be used to recover the PDFs

# Generating samples to reconstruct the PDFs numerically.
Gaussian_rd = norm.rvs(mu, np.sqrt(R), sample_number) # Use multivariate_normal.rvs(mu, R, sample_number) when m > 1 for multivariate normal random samples (R is the covariance matrix)
Gaussian_rd_M = norm.rvs(mu_M, np.sqrt(R_M), sample_number) # Use multivariate_normal.rvs(mu_M, R_M, sample_number) when m > 1 for multivariate normal random samples (R_M is the covariance matrix)

# Reconstructed PDFs from the generated samples using KDE with normal kernels.
p_sampled = gaussian_kde(Gaussian_rd)(x)
p_M_sampled = gaussian_kde(Gaussian_rd_M)(x)

# Numerically computing the mean and variance.
mu_sampled = np.mean(Gaussian_rd)
R_sampled = np.cov(Gaussian_rd, ddof = 1)
mu_M_sampled = np.mean(Gaussian_rd_M)
R_M_sampled = np.cov(Gaussian_rd_M, ddof = 1)

# Plotting the PDFs obtained from the samples.
ax = fig.add_subplot(222)
plt.plot(x, p_sampled, 'b', linewidth = 2)
plt.plot(x, p_M_sampled, 'r', linewidth = 2)
ax.set_title('(b) p(x) Computed Based on Samples', fontsize = 18)
ax.set_xlabel('x', fontsize = 12)
ax.set_ylabel('Probability Density', fontsize = 12)
ax.set_ylim(0, 1.1*max([np.max(p_sampled), np.max(p_M_sampled)]))
ax = fig.add_subplot(224) # Plotting the PDFs on a logarithmic scale to better see the tail behavior 
plt.plot(x, p_sampled, 'b', linewidth = 2)
plt.plot(x, p_M_sampled, 'r', linewidth = 2)
plt.plot(x, p_no_remedy, ':b', linewidth = 2, alpha = 0.2)
plt.plot(x, p_M_no_remedy, ':r', linewidth = 2, alpha = 0.2)
plt.yscale('log')
ax.set_title('Y-axis in Logarithmic Scale', fontsize = 18)
ax.set_xlabel('x', fontsize = 12)
ax.set_ylabel('Probability Density', fontsize = 12)
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(1e-20, 1)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
fig.tight_layout(pad=0, w_pad=-2, h_pad=-2)
plt.show()

# Computing the relative entropy using the signal-dispersion decomposition
# formula since both distributions are Gaussian.
if m == 1:
    R_M_sampled_inv = 1 / R_M
    Signal_sampled = 1/2 * (mu_sampled - mu_M_sampled) * R_M_sampled_inv * (mu_sampled - mu_M_sampled)
    Dispersion_sampled = 1/2 * (R_sampled * R_M_sampled_inv - m - np.log( R_sampled * R_M_sampled_inv ))
else:
    R_M_sampled_inv = np.linalg.inv(R_M_sampled)
    Signal_sampled = 1/2 * (mu_sampled - mu_M_sampled).conj().T @ R_M_sampled_inv @ (mu_sampled - mu_M_sampled)
    Dispersion_sampled = 1/2 * (np.trace( R_sampled @ R_M_sampled_inv ) - m - np.log( np.linalg.det( R_sampled @ R_M_sampled_inv ) ))
RE_theoretic_sampled = Signal_sampled + Dispersion_sampled

# Directly computing the relative entropy based on the PDFs using the definition
# and numerical integration, which suffers from the undersampling of the tail 
# events, with no consideration about the tails of the model distribution
# in the denominator (or the one corresponding to the truth).
RE_numerical_no_normalization_sampled = trapezoid(p_sampled * np.log( p_sampled / p_M_sampled ), x)

# Computing the relative entropy using the definition and numerical integration,
# where the tail probability is set to be a small but nonzero value to account
# for the PDF ratio, and then a corrective normalization is applied to the PDFs.
p_sampled[ p_sampled <= 1e-5 ] = 1e-5
p_M_sampled[ p_M_sampled <= 1e-5 ] = 1e-5
p_sampled = p_sampled / trapezoid(p_sampled, x)
p_M_sampled = p_M_sampled / trapezoid(p_M_sampled, x)
RE_numerical_with_normalization_sampled = trapezoid(p_sampled * np.log( p_sampled / p_M_sampled ), x)

print(f'\nGaussian distributions constructed from samples')
print(f'Relative entropy:\nTheoretical value: {RE_theoretic_sampled:.4f}\nNumerical value without normalization: {RE_numerical_no_normalization_sampled:.4f}\nNumerical value with normalization: {RE_numerical_with_normalization_sampled:.4f}\n')