% Code corresponding to Section 3.1 - "Uncertainty propagation in the linear 
% damped system" in the Supplementary Document of the paper "Taming Uncertainty 
% in a Complex World: The Rise of Uncertainty Quantification — A Tutorial for 
% Beginners". It specifically produces Figure 2 in the main text (in Section 
% "Examples of uncertainty propagation in linear and nonlinear dynamical 
% systems").
%
% Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
%
% Code: Simulating a linear (stochastic) damped system with initial uncertainty
% (MATLAB code).
%
% MATLAB Toolbox and M-file Requirements:
% 
% Code used to obtain the required m-file scripts and MATLAB toolboxes:
% [fList, pList] = matlab.codetools.requiredFilesAndProducts('Linear_System_UQ.m');
%
% M-file Scripts:
%   * Linear_System_UQ.m
%
% Toolboxes:
%   * None

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(42) % Setting the seed number for consistent and reproducible results
N = 2000; % Total numerical integration steps
dt = 0.005; % Numerical integration time step
T = N*dt; % Total length of the simulation in time units

a = 1; % Deterministic and constant damping of the process 
f = 1; % Deterministic and constant forcing of the process
sigma = 0.5; % Noise amplitude/feedback in the case where a stochastic process is being simulated
Ens = 1000; % Number of ensemble members used to calculate the mean ensemble trajectory in the presence of initial uncertainty

x1 = zeros(1, N); % Stochastic process with deterministic initial condition
x2 = zeros(Ens, N); % Stochastic process with uncertainty in its initial value

% Initializing the processes.
x1(1) = 2;
x2(:, 1) = 2 + 0.3 * randn(Ens, 1); % A small zero-mean normal noise is added to all ensemble members initially

% Model simulation using forward Euler.
% Remove the commented-out lines for x1(i) and x2(:, i) to instead add a Wiener
% noise or Brownian motion to the dynamical linear system and make it stochastic 
% thus simulating the system using the Euler-Maruyama method for numerical 
% integration.
for i = 2:N
    x1(i) = x1(i-1) + (-a * x1(i-1) + f) * dt;
    x2(:, i) = x2(:, i-1) + (-a * x2(:, i-1) + f) * dt;
    % x1(i) = x1(i-1) + (-a * x1(i-1) + f) * dt + sigma * sqrt(dt) * randn;
    % x2(:, i) = x2(:, i-1) + (-a * x2(:, i-1) + f) * dt + sigma * sqrt(dt) * randn(Ens, 1);
end

% Plotting the simulated time series and ensemble members (in the case of the 
% process with random initial values).
fig = figure();
subplot(1, 2, 1)
plot(dt:dt:N*dt, x1, 'b', 'LineWidth', 2);
box on
set(gca, 'FontSize', 12)
title('(a) x (Deterministic Initial Condition)', 'FontSize', 18)
xlabel('t')
ylabel('x')
ylim([min(x2, [], 'all'), max(x2, [], 'all')])
subplot(1, 2, 2)
hold on
h = plot(dt:dt:N*dt, x2, 'k', 'LineWidth', 0.5);
h(Ens+1) = plot(dt:dt:N*dt, mean(x2), 'r', 'LineWidth', 2);
box on
set(gca, 'FontSize', 12)
legend(h([1, Ens+1]), {'Ensemble Members', 'Mean Time Series'}, 'FontSize', 14)
title('(b) x (Initial Condition With Small Uncertainty)', 'FontSize', 18)
xlabel('t')
ylabel('x')
ylim([min(x2, [], 'all'), max(x2, [], 'all')])

fig.WindowState = 'Maximized'; % Maximize the figure window