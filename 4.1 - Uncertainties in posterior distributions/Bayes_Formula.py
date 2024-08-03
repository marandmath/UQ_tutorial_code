# Code corresponding to Section 4.1 - "Uncertainties in posterior distributions" 
# in the Supplementary Document of the paper "Taming Uncertainty in a Complex 
# World: The Rise of Uncertainty Quantification — A Tutorial for Beginners".
#
# Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
#
# Code: Illustration of Bayes' formula, as a basis for data assimilation, in the
# case of a one dimensional state variable and a single or multiple 
# observation(s)/measurement(s) (Python code).
#
# Info: The Bayesian update as a basis for data assimilation is showcased, where 
# observations are utilized to update our prior beliefs and obtain the optimal 
# posterior distribution. Here we assume the simplified case of both a Gaussian 
# prior as well as likelihood. We also showcase the asymptotic behavior of the 
# posterior statistics, as well as the relative entropy between the prior and 
# posterior Gaussian distributions (posterior distribution is also Gaussian due 
# to the conjugacy), when they are treated as functions of the number of 
# observations. Specifically, the logarithmic growth of the dispersion part of 
# the relative entropy is displayed, which signifies the diminishing returns 
# that stem from unboundedly increasing the number of observations.
#
# Python Package Requirements:
#   * matplotlib==3.5.3
#   * numpy==1.23.2
#   * scipy==1.9.0
#
# Useful documentation for MATLAB users: 
#   https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

################################################################################
################################################################################\

# Importing the necessary Python packages to run this code.
import numpy as np
from scipy.stats import norm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

################################################################################
############################ A Single Observation ##############################
################################################################################

u = np.linspace(-4, 4, 500) # Domain of the random state variable
np.random.seed(42) # Setting the seed number for consistent and reproducible results

truth = 1 # Truth (true state/value of u)

# Observational noise and noisy observation/measurement.
ro = 1 # Observational noise, namely the variance
g = 1 # Observational operator
v = g * truth + np.sqrt(ro) * norm.rvs(0, 1) # Observation/measurement

# Prior.
mu_f = -1 # Prior mean
R_f = 1 # Prior variance
p_prior = norm.pdf(u, mu_f, np.sqrt(R_f)) # Prior normal distribution

# Likelihood.
p_likelihood = norm.pdf(u, v, np.sqrt(ro)) # Normal likelihood

# Kalman gain.
K = R_f * g * (g * R_f * g + ro)**(-1)

# Posterior.
mu_a = mu_f + K * (v - g * mu_f) # Posterior mean
R_a = (1 - K * g) * R_f # Posterior variance
p_posterior = norm.pdf(u, mu_a, np.sqrt(R_a)) # Posterior normal distribution

# Plotting the distributions for a single observation.
fig = plt.figure()
ax = fig.add_subplot(221)
plt.plot(u, p_prior, 'g', linewidth = 2, label = 'Prior Distribution')
plt.plot(u, p_likelihood, 'c', linewidth = 2, label = 'Likelihood')
plt.plot(u, p_posterior, 'r', linewidth = 2, label = 'Posterior Distribution')
plt.vlines(truth, 0, 2*max([np.max(p_prior), np.max(p_likelihood), np.max(p_posterior)]), 'k', '--', label = 'True Value')
ax.legend(fontsize = 14, loc = 'upper left')
ax.set_title('(a) Distributions; Single Observation', fontsize = 18)
ax.set_xlabel('u (State Variable)', fontsize = 12)
ax.set_ylabel('Probability Density', fontsize = 12)
ax.set_xlim(np.min(u), np.max(u))
ax.set_ylim(0, 2*max([np.max(p_prior), np.max(p_likelihood), np.max(p_posterior)]))

################################################################################
############################ Multiple Observations #############################
################################################################################

# Observational noise and noisy observations/measurements.
L = 10 # Number of observations
ro = 1 * np.eye(L, L) # Observational noise, namely the covariance matrix
g = 1 * np.ones((L, 1)) # Observational operator
v = g * truth + sqrtm(ro) @ norm.rvs(0, 1, (L, 1)) # Observations/measurements

# Prior.
mu_f = -1 # Prior mean
R_f = 1 # Prior variance
p_prior = norm.pdf(u, mu_f, np.sqrt(R_f)) # Prior normal distribution

# Kalman gain.
K = R_f * g.T @ np.linalg.inv(g * R_f @ g.T + ro)

# Posterior.
mu_a = mu_f + K @ (v - g * mu_f) # Posterior mean
R_a = (1 - K @ g) * R_f # Posterior variance
p_posterior = norm.pdf(u, mu_a, np.sqrt(R_a)).T # Posterior normal distribution

# Plotting the distributions for multiple observations.
ax = fig.add_subplot(222)
plt.plot(u, p_prior, 'g', linewidth = 2, label = 'Prior Distribution')
plt.plot(u, p_posterior, 'r', linewidth = 2, label = 'Posterior Distribution')
plt.vlines(truth, 0, 2*max([np.max(p_prior), np.max(p_posterior)]), 'k', '--', label = 'True Value')
ax.legend(fontsize = 14, loc = 'upper left')
ax.set_title(f'(b) Distributions; L = {L} Observations', fontsize = 18)
ax.set_xlabel('u (State Variable)', fontsize = 12)
ax.set_ylabel('Probability Density', fontsize = 12)
ax.set_xlim(np.min(u), np.max(u))
ax.set_ylim(0, 2*max([np.max(p_prior), np.max(p_posterior)]))

################################################################################
############################# Asymptotic Behavior ##############################
################################################################################

# For each L (i.e. number of observations/measurements), we repeat the 
# experiment Test_num times to obtain the mean behavior and the confidence 
# interval, since the observational noise differs in each test.
Test_num = 100 # Number of times that the experiment will be repeated
L_all = np.array([1, 2, 5, 10, 20, 30, 100, 200, 500]) # Testing different L (number of observations)
mu_all = np.zeros((len(L_all), Test_num)) # Storing the posterior mean for each number of observations (L) and experiment
R_all = np.zeros((len(L_all), Test_num)) # Storing the posterior variance for each number of observations (L) and experiment

for i, L in enumerate(L_all):
    for j in range(Test_num):
        
        # Observational noise and noisy observations/measurements.
        ro = 1 * np.eye(L, L) # Observational noise, namely the covariance matrix
        g = 1 * np.ones((L, 1)) # Observational operator
        v = g * truth + sqrtm(ro) @ norm.rvs(0, 1, (L, 1)) # Observations/measurements

        # Prior.
        mu_f = -1 # Prior mean
        R_f = 1 # Prior variance

        # Kalman gain.
        K = R_f * g.T @ np.linalg.inv(g * R_f @ g.T + ro)

        # Posterior.
        mu_a = mu_f + K @ (v - g * mu_f) # Posterior mean
        R_a = (1 - K @ g) * R_f # Posterior variance

        mu_all[i, j] = mu_a
        # In fact, R does not change for different values of the random noise, 
        # as the noise only affects the mean while the posterior variance 
        # depends on the observational operator only (in this case the number of 
        # observations, L), but for completeness we store this matrix here.
        R_all[i, j] = R_a
        
# Plotting the results for the asymptotic behavior of the posterior mean and
# variance as functions of the number of observations, L, when averaging over 
# the repeated experiments, as well as the information gain (uncertainty 
# reduction) through its signal and dispersion parts when expressed as a 
# function of L.
ax1 = fig.add_subplot(223)
ax1.plot(L_all, np.mean(mu_all, 1), 'b', linewidth = 2, label = 'Posterior Mean')
mean_std = np.std(mu_all, 1, ddof = 1)
ax1.fill_between(L_all, np.mean(mu_all, 1) - 2 * mean_std, np.mean(mu_all, 1) + 2 * mean_std, alpha = 0.2, facecolor = 'b', label = '2 Std Posterior Mean')
# Posterior mean asymptotes at the limit of (Σ_{l=1,...,L} v_l)/L as L tends to 
# infinity, which in this case ends up being 1 (the true value), since g_l = 1, 
# u = 1 (true value), and because the sum of the Gaussian error terms satisfy 
# E(Σ_{l=1,...,L} ε_l) = 0 and Var(Σ_{l=1,...,L} ε_l) = L/(L+1)^2 -> 0, 
# as L grows to infinity.
ax1.hlines(1, L_all[0], L_all[-1], 'k', '--', label = 'True Value')
ax1.set_xlabel('L (Number of Measurements)', fontsize = 12)
ax1.set_ylabel('$\mu_a$ (Posterior Mean)', fontsize = 12, color = 'b')
ax1.set_xlim(L_all[0], L_all[-1])
ax1.tick_params(axis = 'y', labelcolor = 'b')
ax2 = ax1.twinx()
ax2.plot(L_all, np.mean(R_all, 1), 'm', linewidth = 2, label = 'Posterior Variance')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, fontsize = 14, loc = 'center right')
ax2.set_title(f'(c) Asymptotic Behavior of Posterior Statistics', fontsize = 18)
ax2.set_ylabel('$R_a$ (Posterior Variance)', fontsize = 12, color = 'm')
ax2.tick_params(axis = 'y', labelcolor = 'm')
ax1 = fig.add_subplot(224)
signal = 1/2 * (np.mean(mu_all, 1) - mu_f)**2 / R_f
dispersion = 1/2 * (np.mean(R_all, 1) / R_f - 1 - np.log( np.mean(R_all, 1) / R_f )) 
ax1.plot(L_all, signal, 'b', linewidth = 2, label = 'Signal')
ax1.set_xlabel('L (Number of Measurements)', fontsize = 12)
ax1.set_ylabel('Bits', fontsize = 12, color = 'b')
ax1.set_xlim(L_all[0], L_all[-1])
ax1.tick_params(axis = 'y', labelcolor = 'b')
ax2 = ax1.twinx()
ax2.plot(L_all, dispersion, 'm', linewidth = 2, label = 'Dispersion')
ax2.plot(L_all, np.log( L_all + 1 ) / 2 - L_all / (L_all+1) / 2, '--k', linewidth = 2, label = '(ln(L+1)-L/(L+1))/2')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, fontsize = 14, loc = 'lower right')
ax2.set_title(f'(d) Uncertainty Reduction (Relative Entropy)', fontsize = 18)
ax2.set_ylabel('Bits', fontsize = 12, color = 'm')
ax2.tick_params(axis = 'y', labelcolor = 'm')

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
fig.tight_layout(pad=0, w_pad=-2, h_pad=-2)
plt.show()