% Code corresponding to Section 6 - "Calibrating Stochastic Models Based on UQ" 
% in the Supplementary Document of the paper "Taming Uncertainty in a Complex 
% World: The Rise of Uncertainty Quantification — A Tutorial for Beginners".
%
% Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
%
% Code: Calibrating a linear stochastic model with uncertainty quantification 
% through the second-order spatiotemporal statistics from a stochastic model 
% with cubic damping and multiplicative noise (MATLAB code).
% 
% Info: The equilibrium mean, equilibrium variance, and decorrelation time
% (temporal integral of the autocorrelation function) from a dynamic stochastic 
% model with cubic nonlinearity and correlated additive and multiplicative 
% noise are used to calibrate a linear stochastic model such that it performs 
% similarly under the criteria of model memory and model fidelity. Such 
% nonlinear models are usually used as reduced order climate models for 
% low-frequency atmospheric variability. Four different sets of parameter values 
% are used to exhibit different complex and highly non-Gaussian behaviors,
% where the aforementioned second-order spatiotemporal statistics of the given 
% time series are retrieved and used to uniquely determine the calibrated
% simplified linear system. This code aims to see the performance of the linear
% stochastic model in these different regimes. For the nearly Gaussian case, the 
% linear stochastic model will give a nearly perfect representation of the 
% results. For the highly non-Gaussian cases, only the Gaussian statistics can 
% be captured, which is the best the linear stochastic model can do.
%
% MATLAB Toolbox and M-file Requirements:
%
% [fList, pList] = matlab.codetools.requiredFilesAndProducts('Calibrating_Stochastic_Model_with_UQ.m');
%
% M-files/Scripts:
%   * Calibrating_Stochastic_Model_with_UQ.m
%
% Toolboxes:
%   * Statistics and Machine Learning Toolbox
%   * Curve Fitting Toolbox
%   * Econometrics Toolbox

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig = figure();

N = 1000000; % Total numerical integration steps 
dt = 0.005; % Numerical integration time step
T = N*dt; % Total length of the simulation in time units

for j = 1:4

    rng(42) % Setting the seed number for consistent and reproducible results

    % We inspect four different dynamical regimes which are controlled by the 
    % parameter values of the nonlinear stochastic model with multiplicative 
    % noise.
    switch j
        case 1 % Nearly Gaussian regime
            a = -2.2; b = 0; c = 0; f = 2; A = 0.1; B = 0.1; sigma = 1;
        case 2 % Highly skewed regime
            a = -4; b = 2; c = 1; f = 0.1; A = 1; B = -1; sigma = 1;
        case 3 % Fat tailed regime
            a = -3; b = -1.5; c = 0.5; f = 0.0; A = 0.5; B = -1; sigma = 1;
        case 4 % Bimodal regime
            a = 4; b = 2; c = 1; f = 0.1; A = 1; B = -1; sigma = 1;
    end

    x = zeros(1, N); % State variable of the true signal from the cubic nonlinear model with CAM
    
    % Model simulation using the Euler-Maruyama method which in general has weak 
    % convergence order of p = 1 and strong convergence order of p = 1/2. Since 
    % the true model has multiplicative noise, the Euler-Maruyama and Milstein 
    % methods differ. The latter can be used for a higher order of strong 
    % convergence (p = 1) by replacing (A - B * x(i-1)) * sqrt(dt) * randn below
    % with:
    % (A - B * x(i-1)) * sqrt(dt) * W - 1/2 * B * (A - B * x(i-1)) * (W^2 - dt), 
    % where W = randn (independent zero-mean and dt variance normal variable).
    for i = 2:N
        x(i) = x(i-1) + (a * x(i-1) + b * x(i-1)^2 - c * x(i-1)^3 + f) * dt + ...
               (A - B * x(i-1)) * sqrt(dt) * randn + sigma * sqrt(dt) * randn;
    end
    
    mu_truth = mean(x(1000:10:end)); % Mean of the true time series (considering a burn-in period and constant-interval sampling)
    R_truth = var(x(1000:10:end)); % Variance of the true time series (considering a burn-in period and constant-interval sampling)
    if j <=3
        lag = 1000;
    else
        lag = 10000; % A larger lag is required for the bimodal regime to capture the overall model memory due to the state/mode switching
    end
    ACF_truth = autocorr(x, NumLags = lag); % Autocorrelation function of the true signal
    [PDF_truth, PDF_range] = ksdensity(x(10000:end)); % Time-averaged PDF of the true dynamics (considering a burn-in period)

    % Calculating the damping, forcing, and noise feedback parameters of the 
    % calibrated linear OU stochastic process via the true second-order 
    % statistics which are the equilibrium mean, equilibrium variance, and 
    % decorrelation time (temporal integral of the ACF).
    % Through the 'StartPoint' argument we provide the initial values for the 
    % coefficients used to fit the exponential model which we use to calculate
    % the decorrelation time. This is because the ACF of a linear OU process 
    % with additive noise is an exponential decay function with decay rate equal 
    % to the reciprocal of the decorrelation time.
    f = fit((0:dt:lag*dt)', ACF_truth', 'exp1', 'StartPoint', [-20, 0]);
    a_calibrate = -f.b; % Using the exponential model's rate of change parameter to calculate the damping parameter, which is equal to the reciprocal of the decorrelation time
    f_calibrate = a_calibrate * mu_truth;  % Deterministic forcing is equal to the product of the equilibrium mean and damping parameter
    sigma_calibrate = sqrt(2 * a_calibrate * R_truth); % Noise feedback/amplitude is equal to the square root of twice the product of the equilibrium mean and damping parameter
    
    y = zeros(1, N); % State variable of the signal from the linear approximation model
    rng(42) % Resetting the seed number as to use the same sampled values for the Wiener noise
    % Model simulation using Euler-Maruyama for the approximating linear process
    % (no need for Milstein's method since the noise is additive).
    for i = 2:N 
        y(i) = y(i-1) + (- a_calibrate * y(i-1) + f_calibrate) * dt + sigma_calibrate * sqrt(dt) * randn;
    end
    
    ACF_calibrate = autocorr(y, NumLags = lag); % Autocorrelation function of the signal from the calibrated linear stochastic model
    [PDF_calibrate, ~] = ksdensity(y(10000:end), PDF_range); % Time-averaged PDF of the signal from the calibrated linear stochastic model (considering a burn-in period)
    
    subplot(3, 4, j)
    hold on
    % For visual clarity we plot the time series every 10*dt time units.
    plot(10000*dt:10*dt:50000*dt, x(10000:10:50000), 'b', 'LineWidth', 1)
    plot(10000*dt:10*dt:50000*dt, y(10000:10:50000), 'r', 'LineWidth', 1)
    box on 
    set(gca, 'fontsize', 12)
    switch j
        case 1 % Nearly Gaussian regime
            title('(a) Nearly Gaussian Regime', 'FontSize', 18)
        case 2 % Highly skewed regime
            title('(b) Highly Skewed Regime', 'FontSize', 18)
        case 3 % Fat tailed regime
            title('(c) Fat-tailed Regime', 'FontSize', 18)
        case 4 % Bimodal regime
            title('(d) Bimodal Regime', 'FontSize', 18)
    end
    xlabel('t')
    ylabel('x')
    xlim([10000*dt, 50000*dt])

    subplot(3, 4, 4+j)
    hold on
    plot(0:dt:lag*dt, ACF_truth, 'b', 'LineWidth', 2)
    plot(0:dt:lag*dt, ACF_calibrate, 'r', 'LineWidth', 2)
    box on 
    set(gca, 'FontSize', 12)
    title('ACF', 'FontSize', 18)     
    xlabel('Lag')
    ylabel('Autocorrelation')
    xlim([0, lag*dt])

    subplot(3, 4, 8+j)
    hold on
    plot(PDF_range, PDF_truth, 'b', 'LineWidth', 2)
    plot(PDF_range, PDF_calibrate, 'r', 'LineWidth', 2)
    box on 
    set(gca, 'FontSize', 12)
    title('PDF', 'FontSize', 18)
    if j == 4
        legend('Truth (Cubic Model With Multiplicative Noise)', 'Linear Stochastic Model', 'FontSize', 14, 'Location', 'Best')
    end
    xlabel('x')
    ylabel('p(x) (Probability Density)')
    xlim([min(PDF_range), max(PDF_range)])

end

fig.WindowState = 'Maximized'; % Maximize the figure window