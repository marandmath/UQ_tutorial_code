# Code corresponding to Section 2.1 - "Shannon’s entropy" in the Supplementary 
# Document of the paper "Taming Uncertainty in a Complex World: The Rise of 
# Uncertainty Quantification — A Tutorial for Beginners".
#
# Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
# 
# Code: Computing Shannon's entropy (Python code).
#
# Info: Shannon's differential entropy is calculated numerically and via
# analytic formulae for a Gaussian and a Gamma random variable.
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
from scipy.stats import gamma as gamma_distr
from scipy.integrate import trapezoid
from scipy.special import gamma, digamma
import matplotlib.pyplot as plt

################################################################################
############################ Gaussian Distribution #############################
################################################################################

# Calculating Shannon's entropy using numerical integration and the analytic 
# formula.
x1 = np.linspace(-10, 10, 500) # Domain of the Gaussian random variable
x = x1
mu = 0; R = 1 # Mean and variance of the Gaussian distribution
# Leading two moments also obtained via norm.stats(mu, R, moments='mv').
print(f'Mean of Gaussian = {mu:.4f} | Variance of Gaussian = {R:.4f}') # Displaying the leading two moments
p_Gauss = 1/np.sqrt( 2 * np.pi * R ) * np.exp( -(x - mu)**2 / 2 / R ) # Computing the Gaussian PDF over the domain (p_Gauss = norm.pdf(x, mu, R))
p_Gauss = p_Gauss / trapezoid(p_Gauss, x) # Normalization to eliminate small numerical errors due to the finite measure of the domain
# Can also calculate the Shannon differential entropy for a Gaussian distribution using norm.entropy(mu, R).
entropy_numerical = - trapezoid(p_Gauss * np.log( p_Gauss ), x) # Numerically computing Shannon's entropy using the definition
entropy_theoretic = 1/2 * np.log( 2 * np.pi ) + 1/2 * np.log( R ) + 1/2 # Analytic formula of computing Shannon's entropy for a Gaussian distribution
print(f'Shannon entropy for the Gaussian distribution N({mu:.3f}, {R:.3f}):\nNumerical value: {entropy_numerical:.4f}\nTheoretical value: {entropy_theoretic:.4f}\n') 

# Plotting the PDF and the theoretical Shannon entropy for the Gaussian 
# random variable.
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x, p_Gauss, 'b', linewidth = 2)
ax.set_title(fr'PDF of Gaussian Distribution With Mean $\mu$ = {mu:.4f} and Variance R = {R:.4f}', fontsize = 18)
ax.set_xlabel('x', fontsize = 12)
ax.set_ylabel('p(x) (Probability Density)', fontsize = 12)
ax.autoscale(enable = True, axis = 'both', tight = True)
plt.text(0.95*np.min(x), 0.95*np.max(p_Gauss), f'Entropy = {entropy_theoretic:.4f}', fontsize = 16)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
plt.show()

################################################################################
############################## Gamma Distribution ##############################
################################################################################

# Calculating Shannon's entropy using numerical integration and the analytic 
# formula.
x2 = np.linspace(0.01, 20.01, 500) # Domain of the Gamma random variable (x must be positive since Gamma distributions are supported on the positive reals)
x = x2
k = 2; theta = np.sqrt(2) # Shape and scale parameters of the Gamma distribution
# Leading four moments also obtained via gamma_distr.stats(a = k, scale = theta, moments='mvsk').
print(f'Mean of Gamma({k:.3f}, {theta:.3f}) = {k * theta:.4f} | Variance of Gamma({k:.3f}, {theta:.3f}) = {k * theta**2:.4f} | Skewness of Gamma({k:.3f}, {theta:.3f}) = {2 / np.sqrt( k ):.4f} | Kurtosis of Gamma({k:.3f}, {theta:.3f}) = {6 / k:.4f}') # Displaying the leading four moments
p_Gamma = gamma_distr.pdf(x, a = k, scale = theta) # Computing the Gamma PDF over the domain
peak_value = np.max(p_Gamma) # Computing the peak value
p_Gamma = p_Gamma / trapezoid(p_Gamma, x) # Normalization to eliminate small numerical errors due to the finite measure of the domain
entropy_numerical = - trapezoid(p_Gamma * np.log( p_Gamma ), x) # Numerically computing Shannon's entropy using the definition
entropy_theoretic = k + np.log( theta ) + np.log( gamma( k ) ) + (1 - k) * digamma( k )  # Analytic formula of computing Shannon's entropy for a Gamma distribution
# Can also calculate the Shannon differential entropy for a Gamma distribution 
# using gamma_distr.entropy(a = k, scale = theta).
print(f'Shannon entropy for the Gamma distribution Gamma({k:.3f}, {theta:.3f}):\nNumerical value: {entropy_numerical:.4f}\nTheoretical value: {entropy_theoretic:.4f}\n') 

# Plotting the PDF and the theoretical Shannon entropy for the Gamma random
# variable.
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x, p_Gamma, 'b', linewidth = 2)
ax.set_title(fr'PDF of Gamma Distribution With Shape k = {k:.4f} and Scale $\theta$ = {theta:.4f}', fontsize = 18)
ax.set_xlabel('x', fontsize = 12)
ax.set_ylabel('p(x) (Probability Density)', fontsize = 12)
ax.autoscale(enable = True, axis = 'both', tight = True)
plt.text(0.85*np.max(x), 0.95*peak_value, f'Entropy = {entropy_theoretic:.4f}', fontsize = 16)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
plt.show()