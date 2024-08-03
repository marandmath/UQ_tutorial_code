# Code corresponding to Section 6 - "Calibrating Stochastic Models Based on UQ" 
# in the Supplementary Document of the paper "Taming Uncertainty in a Complex 
# World: The Rise of Uncertainty Quantification — A Tutorial for Beginners".
#
# Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
#
# Code: Calibrating a linear stochastic model with uncertainty quantification 
# through the second-order spatiotemporal statistics from a stochastic model 
# with cubic damping and multiplicative noise (Python code).
# 
# Info: The equilibrium mean, equilibrium variance, and decorrelation time
# (temporal integral of the autocorrelation function) from a dynamic stochastic 
# model with cubic nonlinearity and correlated additive and multiplicative 
# noise are used to calibrate a linear stochastic model such that it performs 
# similarly under the criteria of model memory and model fidelity. Such 
# nonlinear models are usually used as reduced order climate models for 
# low-frequency atmospheric variability. Four different sets of parameter values 
# are used to exhibit different complex and highly non-Gaussian behaviors,
# where the aforementioned second-order spatiotemporal statistics of the given 
# time series are retrieved and used to uniquely determine the calibrated
# simplified linear system. This code aims to see the performance of the linear
# stochastic model in these different regimes. For the nearly Gaussian case, the 
# linear stochastic model will give a nearly perfect representation of the 
# results. For the highly non-Gaussian cases, only the Gaussian statistics can 
# be captured, which is the best the linear stochastic model can do.
#
# Python Package Requirements:
#   * matplotlib==3.5.3
#   * numpy==1.23.2
#   * scipy==1.9.0
#   * statsmodels==0.13.5
#
# Useful documentation for MATLAB users: 
#   https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

################################################################################
################################################################################

# Importing the necessary Python packages to run this code.
import numpy as np
from scipy.stats import norm, gaussian_kde
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

fig = plt.figure()

N = 1000000 # Total numerical integration steps 
dt = 0.005 # Numerical integration time step
T = N*dt # Total length of the simulation in time units

for j in range(1, 5):

    np.random.seed(42) # Setting the seed number for consistent and reproducible results

    # We inspect four different dynamical regimes which are controlled by the 
    # parameter values of the nonlinear stochastic model with multiplicative 
    # noise.
    match j:
        case 1: # Nearly Gaussian regime
            a = -2.2; b = 0; c = 0; f = 2; A = 0.1; B = 0.1; sigma = 1
        case 2: # Highly skewed regime
            a = -4; b = 2; c = 1; f = 0.1; A = 1; B = -1; sigma = 1
        case 3: # Fat tailed regime
            a = -3; b = -1.5; c = 0.5; f = 0.0; A = 0.5; B = -1; sigma = 1
        case 4: # Bimodal regime
            a = 4; b = 2; c = 1; f = 0.1; A = 1; B = -1; sigma = 1

    x = np.zeros(N) # State variable of the true signal from the cubic nonlinear model with CAM

    # Model simulation using the Euler-Maruyama method which in general has weak 
    # convergence order of p = 1 and strong convergence order of p = 1/2. Since 
    # the true model has multiplicative noise, the Euler-Maruyama and Milstein 
    # methods differ. The latter can be used for a higher order of strong 
    # convergence (p = 1) by replacing 
    # (A - B * x[i-1]) * np.sqrt(dt) * norm.rvs(0, 1) below with:
    # (A - B * x[i-1]) * np.sqrt(dt) * W - 1/2 * B * (A - B * x[i-1]) * (W**2 - dt), 
    # where W = norm.rvs(0, 1) (independent zero-mean and dt variance normal 
    # variable).
    for i in range(1, N):
        x[i] = x[i-1] + (a * x[i-1] + b * x[i-1]**2 - c * x[i-1]**3 + f) * dt \
               + (A - B * x[i-1]) * np.sqrt(dt) * norm.rvs(0, 1) + sigma * np.sqrt(dt) * norm.rvs(0, 1)
    
    mu_truth = np.mean(x[1000::10]) # Mean of the true time series (considering a burn-in period and constant-interval sampling)
    R_truth = np.var(x[1000::10], ddof = 1) # Variance of the true time series (considering a burn-in period and constant-interval sampling)
    if j <= 3:
        lag = 1000
    else:
        lag = 10000 # A larger lag is required for the bimodal regime to capture the overall model memory due to the state/mode switching
    ACF_truth = acf(x, nlags = lag) # Autocorrelation function of the true signal
    PDF_range = np.linspace(np.min(x[10000:]), np.max(x[10000:]), 100)
    PDF_truth = gaussian_kde(x[10000:]) # Time-averaged PDF of the true dynamics (considering a burn-in period)

    # Calculating the damping, forcing, and noise feedback parameters of the 
    # calibrated linear OU stochastic process via the true second-order 
    # statistics which are the equilibrium mean, equilibrium variance, and 
    # decorrelation time (temporal integral of the ACF).
    # Through the 'p0' argument we provide the initial values for the 
    # coefficients used to fit the exponential model which we use to calculate
    # the decorrelation time. This is because the ACF of a linear OU process
    # with additive noise is an exponential decay function with decay rate equal 
    # to the reciprocal of the decorrelation time.
    exp_model = lambda t, a, b: a*np.exp(b*t)
    f = curve_fit(exp_model, np.linspace(0, lag*dt, lag+1), ACF_truth, p0 = [-20, 0])
    a_calibrate = -f[0][1] # Using the exponential model's rate of change parameter to calculate the damping parameter, which is equal to the reciprocal of the decorrelation time
    f_calibrate = a_calibrate * mu_truth;  # Deterministic forcing is equal to the product of the equilibrium mean and damping parameter
    sigma_calibrate = np.sqrt(2 * a_calibrate * R_truth); # Noise feedback/amplitude is equal to the square root of twice the product of the equilibrium mean and damping parameter

    y = np.zeros(N) # State variable of the signal from the linear approximation model
    np.random.seed(42) # Resetting the seed number as to use the same sampled values for the Wiener noise
    # Model simulation using Euler-Maruyama for the approximating linear process
    # (no need for Milstein's method since the noise is additive).
    for i in range(1, N): 
        y[i] = y[i-1] + (- a_calibrate * y[i-1] + f_calibrate) * dt + sigma_calibrate * np.sqrt(dt) * norm.rvs(0, 1)
    
    ACF_calibrate = acf(y, nlags = lag) # Autocorrelation function of the signal from the calibrated linear stochastic model
    PDF_calibrate = gaussian_kde(y[10000:]) # Time-averaged PDF of the signal from the calibrated linear stochastic model (considering a burn-in period)

    ax = fig.add_subplot(3, 4, j)
    # For visual clarity we plot the time series every 10*dt time units.
    tt = np.linspace(10001*dt, 50001*dt, 4000)
    plt.plot(tt, x[10000:50000:10], 'b', linewidth = 1)
    plt.plot(tt, y[10000:50000:10], 'r', linewidth = 1)
    match j:
        case 1: # Nearly Gaussian regime
            ax.set_title('(a) Nearly Gaussian Regime', fontsize = 18)
        case 2: # Highly skewed regime
            ax.set_title('(b) Highly Skewed Regime', fontsize = 18)
        case 3: # Fat tailed regime
            ax.set_title('(c) Fat-tailed Regime', fontsize = 18)
        case 4: # Bimodal regime
            ax.set_title('(d) Bimodal Regime', fontsize = 18)
    ax.set_xlabel('t', fontsize = 12)
    ax.set_ylabel('x', fontsize = 12)
    ax.set_xlim(10000*dt, 50000*dt)

    ax = fig.add_subplot(3, 4, 4+j)
    plt.plot(np.linspace(0, lag*dt, lag+1),  ACF_truth, 'b', linewidth = 2)
    plt.plot(np.linspace(0, lag*dt, lag+1),  ACF_calibrate, 'r', linewidth = 2)
    ax.set_title('ACF', fontsize = 18)
    ax.set_xlabel('Lag', fontsize = 12)
    ax.set_ylabel('Autocorrelation', fontsize = 12)
    ax.set_xlim(0, lag*dt)

    ax = fig.add_subplot(3, 4, 8+j)
    plt.plot(PDF_range, PDF_truth(PDF_range), 'b', linewidth = 2, label = 'Truth (Cubic Model With Multiplicative Noise)')
    plt.plot(PDF_range, PDF_calibrate(PDF_range), 'r', linewidth = 2, label = 'Linear Stochastic Model')
    ax.set_title('PDF', fontsize = 18)
    if j == 4:
        ax.legend(fontsize = 14, loc = 'upper right')
    ax.set_xlabel('x', fontsize = 12)
    ax.set_ylabel('p(x) (Probability Density)', fontsize = 12)
    ax.set_xlim(np.min(PDF_range), np.max(PDF_range))

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized() # Maximize the figure window using the 'Qt4Agg' backend; Use plt.switch_backend('QT4Agg') prior to this line if needed
plt.subplots_adjust(wspace = 0.2, hspace = 0.4)
plt.show()