% Code corresponding to Section 3.2 - "Uncertainty propagation in the chaotic 
% Lorenz 63 model" in the Supplementary Document of the paper "Taming 
% Uncertainty in a Complex World: The Rise of Uncertainty Quantification — A 
% Tutorial for Beginners". It specifically produces Figure 3 in the main text 
% (in Section "Examples of uncertainty propagation in linear and nonlinear 
% dynamical systems").
%
% Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
%
% Code: Simulate a chaotic system (Lorenz 63 model) with and without initial
% uncertainty (MATLAB code).
%
% MATLAB Toolbox and M-file Requirements:
% 
% Code used to obtain the required m-file scripts and MATLAB toolboxes:
% [fList, pList] = matlab.codetools.requiredFilesAndProducts('L63_UQ.m');
%
% M-file Scripts:
%   * L63_UQ.m
%
% Toolboxes:
%   * None

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(42) % Setting the seed number for consistent and reproducible results
T = 150; % Total length of the simulation in time units
dt = 0.005; % Numerical integration time step 
N = round(T/dt); % Total numerical integration steps

% Model parameters (same as the ones in the original paper of Lorenz).
sigma = 10;
rho = 28;
beta = 8/3;
Ens = 1000; % Number of ensemble members used to calculate the mean ensemble trajectory in the presence of initial uncertainty

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Simulation With Initial Uncertainty %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% State variables.
x1 = zeros(Ens, N);
y1 = zeros(Ens, N);
z1 = zeros(Ens, N);

% Initial values with an added normal noise of zero-mean and unit variance.
x1(:, 1) =  20 + randn(Ens, 1);
y1(:, 1) = -20 + randn(Ens, 1);
z1(:, 1) =  25 + randn(Ens, 1);

% Model simulation using forward Euler.
for i = 2:N
    x1(:, i) = x1(:, i-1) + sigma * (y1(:, i-1) - x1(:, i-1)) * dt;
    y1(:, i) = y1(:, i-1) + (x1(:, i-1) .* (rho - z1(:, i-1)) - y1(:, i-1)) * dt;
    z1(:, i) = z1(:, i-1) + (x1(:, i-1) .* y1(:, i-1) - beta * z1(:, i-1)) * dt;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Simulation With No Initial Uncertainty %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% State variables.
x2 = zeros(1, N);
y2 = zeros(1, N);
z2 = zeros(1, N);

% Initial value which differs from the first one by a small amount
% (deterministic in this case).
x2(1) =  20;
y2(1) = -20;
z2(1) =  25;

% Model simulation using forward Euler.
for i = 2:N
    x2(i) = x2(i-1) + sigma * (y2(i-1) - x2(i-1)) * dt;
    y2(i) = y2(i-1) + (x2(i-1) * (rho - z2(i-1)) - y2(i-1)) * dt;
    z2(i) = z2(i-1) + (x2(i-1) * y2(i-1) - beta * z2(i-1)) * dt;
end

% Plotting the trajectories for both simulations.
plot_time = 15;
fig = figure();
subplot(2, 6, [1, 2, 7, 8])
plot3(x2, y2, z2, 'b', 'LineWidth', 1.5);
view(210, 30)
box on
grid on
set(gca, 'FontSize', 12)
title(sprintf('(a) Phase Plot of Lorenz 63 Attractor\nWith Deterministic IC (T=%d)', T), 'FontSize', 18)
xlabel('x')
ylabel('y')
zlabel('z')
subplot(2, 6, 3:6)
plot(dt:dt:round(plot_time/dt)*dt, x2(1:round(plot_time/dt)), 'b', 'LineWidth', 2);
box on
set(gca, 'FontSize', 12)
title(sprintf('(b) Deterministic Initial Condition (T=%d)', plot_time), 'FontSize', 18)
xlabel('t')
ylabel('x')
subplot(2, 6, 9:12)
hold on
h = plot(dt:dt:round(plot_time/dt)*dt, x1(:, 1:round(plot_time/dt)), 'k', 'LineWidth', 0.5);
h(N+1) = plot(dt:dt:round(plot_time/dt)*dt, mean(x1(:, 1:round(plot_time/dt))), 'r', 'LineWidth', 2);
box on
set(gca, 'FontSize', 12)
legend(h([1, N+1]), {'Ensemble Members', 'Mean Time Series'}, 'FontSize', 14, 'Location', 'Northeast')
title(sprintf('(c) Initial Condition With Small Uncertainty (T=%d)', plot_time), 'FontSize', 18)
xlabel('t')
ylabel('x')

fig.WindowState = 'Maximized'; % Maximize the figure window