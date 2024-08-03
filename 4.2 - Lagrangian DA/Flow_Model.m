% Code corresponding to Section 4.2 - "Lagrangian DA" in the Supplementary 
% Document of the paper "Taming Uncertainty in a Complex World: The Rise of 
% Uncertainty Quantification — A Tutorial for Beginners".
%
% Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
%
% Code: Generates a 2D incompressible flow with random amplitude based on a 
% linear stochastic model (OU process) for the Fourier modes, mimicking the 
% quasi-geostrophic (QG) or potential vortical flow (MATLAB code).
%
% Info: Only the geostrophically balanced (GB) modes are considered in this
% simulation, for simplicity, thus making the underlying velocity field
% incompressible due to the omission of the Poincare/inertio-gravity 
% compressible wave modes. The Fourier wavenumber grid is also plotted along 
% with the order numbers that are being used for each mode in the global system.
% This m-file is used by LDA_Function_of_L.m as to use its variables.
%
% MATLAB Toolbox and M-file Requirements:
% 
% Code used to obtain the required m-file scripts and MATLAB toolboxes:
% [fList, pList] = matlab.codetools.requiredFilesAndProducts('Flow_Model.m');
%
% M-file Scripts:
%   * Flow_Model.m
%
% Toolboxes:
%   * None

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(42) % Setting the seed number for consistent and reproducible results

K_max = 2; % The range of the Fourier modes is [-K_max, K_max]^2∩Z
k = zeros(2, (2 * K_max + 1) * (2 * K_max + 1)); % Collection of Fourier wavenumbers over the symmetric discrete lattice

% Arranging the Fourier wavenumbers in such a way that the complex conjugate 
% modes are next to each other, namely (-k1, -k2) will be next to (k1, k2). This 
% will facilitate data assimilation by allowing the structure of the matrix 
% coefficients in the conditional Gaussian nonlinear system framework (which in
% this case consists of the OU process driving the Fourier modes of the flow for
% the GB incompressible waves (Forecast flow model) and the acceleration 
% relation for the tracers' trajectories (Observational process)) to be block 
% diagonal matrices, with each block being a 2 by 2 matrix, while also 
% simplifying the form of the deterministic forcing in the forecast flow model. 
% This then simplifies the enforcement of the reality/conjugate conditions onto 
% the Fourier conjugate wavenumbers, which are required as to establish that the 
% flow velocity is a real-valued vector field.
m = 1;
for i = -K_max:K_max
    if i < 0
        for j = -K_max:i
            k(1, m) = i;
            k(2, m) = j;
            m = m + 2;
        end
    else
        for j = -K_max:i-1
            k(1, m) = i;
            k(2, m) = j;
            m = m + 2;
        end
    end
end
k(:, 2:2:end-1) = - k(:, 1:2:end-2);

% Showing the grid (discrete and symmetric around the origin lattice) of Fourier 
% wavenumbers.
fig = figure();
hold on
plot(k(1, 1:2:end-2), k(2, 1:2:end-2), 'ro', 'LineWidth', 4);
plot(k(1, 2:2:end-1), k(2, 2:2:end-1), 'go', 'LineWidth', 4);
plot(k(1, end), k(2, end), 'ko', 'LineWidth', 4);
text(k(1, :)+0.05, k(2, :)-0.05, string(1:size(k, 2)), 'FontSize', 12, 'FontWeight', 'Bold')
box on
grid on
set(gca, 'FontSize', 12)
title(sprintf('Fourier Wavenumbers Over the Discrete Symmetric Lattice in [-%d,%d]^2 and Their Order in the Global Vector', K_max, K_max), 'FontSize', 18)
xlabel('k_1')
ylabel('k_2')

fig.WindowState = 'Maximized'; % Maximize the figure window

kk = k(:, 1:end-1); % Remove the (0, 0) mode by assuming no background mean flow/sweep 

% Defining the eigenvectors that contain the relationships between different 
% components of the flow field. E.g. in the case of an incompressible flow field
% the divergence-free condition is reflected in the eigenvector of each Fourier 
% mode. In the case where we consider the inclusion of the compressible waves, 
% we order the eigenvectors for the gravity modes in the following manner (not
% applicable in this case):
% omegak = [
%               omegak_+(1:(2*K_max+1)^2-1), 
%               omegak_-(1:(2*K_max+1)^2-1), 
%               omegak_+(end), 
%               omegak_-(end)
% ];
% In the last column of the compressible eigenvectors rk2 and rk3, is the (0,0) 
% inertio-gravity or Poincare mode, which needs to be dealt with in a different 
% way. This structure simplifies the enforcement of the conjugate/reality 
% condition on the eigenvectors and model parameters, which extends to keeping 
% the flow field real-valued. The correspondence of the eigenvectors to 
% atmospheric modes is the following:
%       rk1: Geostrophically Balanced (GB); rk2: Gravity +; rk3: Gravity -.
% We also mention the degrees of freedom of our system when assuming complex 
% coefficients = ((2*K_max+1)^2+1)/2 (due to the conjugate/reality condition 
% enforced on the conjugate Fourier wavenumbers and the inclusion of only the 
% GB modes).

% Assuming no Poincare/inertio-gravity compressible waves, just the GB modes 
% corresponding to the incompressible flow.
rk1 = [
    1./sqrt(k(1, :).^2 + k(2, :).^2 + 1) .* (-1i * k(2, :));
    1./sqrt(k(1, :).^2 + k(2, :).^2 + 1) .* (1i * k(1, :))
];
rk1 = rk1(:, 1:end-1);
rk = rk1;

T = 15; % Total length of the simulation in time units
dt = 0.001; % Numerical integration time step
N = round(T/dt); % Total numerical integration steps
Dim_U = size(kk, 2); % Dimension of system = Number of Fourier modes - Origin = (2*K_max+1)^2-1
u_hat = zeros(Dim_U, N); % Defining all the Fourier modes of the velocity field

% Damping feedback of the GB modes: To preserve the reality of the flow field, 
% it must be that d_k=d_(-k), i.e. the conjugate Fourier mode has the same
% damping coefficient.
d_B = 0.5;
% The GB flows are non-divergent, which is embodied in the eigenvector, and so
% the associated phase/rotational speeds or eigenvalues are necessarily equal to 
% zero.
omega_B = 0;
% Noise of the GB modes: To preserve the reality of the flow field, it must be 
% that σ_k=σ_(-k), i.e. the conjugate Fourier modes have the same noise 
% coefficient, which is necessarily nonnegative and real.
sigma_B = 0.5;

Gamma = (d_B - 1i * omega_B) * eye(Dim_U); % Γ (Gamma): Damping and phase coefficients of the Fourier modes' OU processes
f = zeros(Dim_U, N-1); % F: Deterministic and periodic forcing in the Fourier modes' OU processes; Being deterministic (or at least an affine function of the Fourier mode) ensures a Gaussian attractor/equilibrium distribution
f_strength = 0.25; % Strength of the background deterministic force (force is assumed zero for the purposes of the paper)
f_period = 2; % Period of the background deterministic force (force is assumed zero for the purposes of the paper)
Sigma_u_hat = zeros(Dim_U, Dim_U); % Σ_u_hat (Sigma_u_hat): Noise matrix of the flow field's Fourier modes
for j = 1:2:Dim_U
    Sigma_u_hat(j, j) = 1 / sqrt(2) * sigma_B;
    Sigma_u_hat(j+1, j+1) = -1i / sqrt(2) * sigma_B;
    Sigma_u_hat(j, j+1) = 1i / sqrt(2) * sigma_B;
    Sigma_u_hat(j+1, j) = 1 / sqrt(2) * sigma_B;
end

% Model simulation using Euler-Maruyama for the Fourier modes.
% A stochastic system is utilized for each Fourier mode, which is written in a
% vector form that fits the conditional Gaussian nonlinear system framework in
% the following way:
%      (Forecast Flow Model) du_hat = (-Γ * u_hat + F) * dt + Σ_u_hat * dW_u_hat
dw_u_hat = randn(Dim_U, N-1); % Wiener noise values for the simulation
for j = 2:N
    
    t = j * dt; % Current time in the simulation

    % Forcing can be either zero (homogeneous linear SDE) or a periodic complex 
    % exponential/plane sinusoidal wave (in general a complex function of some
    % period): To preserve the reality of the flow field, it must be that 
    % f_k=f_(-k)*, where * denotes the complex conjugate, i.e. the conjugate 
    % Fourier mode has complex conjugate forcing. For this simulation we assume
    % zero forcing, but the following can be uncommented in case we want to
    % include a non-zero background periodic and deterministic force.
    % f(:, j-1) = f_strength * reshape( ...
    %     [
    %         exp(1i * 2*pi*t/f_period) * ones(1, round(Dim_U/2));
    %         exp(-1i * 2*pi*t/f_period) * ones(1, round(Dim_U/2)) ...
    %     ], ...
    % [], 1); % Force feedback = 0.25; Force period = 5 time units

    u_hat(:, j) = u_hat(:, j-1) + (-Gamma * u_hat(:, j-1) + f(:, j-1)) * dt + Sigma_u_hat * sqrt(dt) * dw_u_hat(:, j-1);

end

% Spatiotemporal reconstruction of the incompressible velocity field.
Dim_Grid = 25;
[xx, yy] = meshgrid(linspace(-pi, pi, Dim_Grid), linspace(-pi, pi, Dim_Grid));
x_vec = [reshape(xx, [], 1), reshape(yy, [], 1)]; 
 
fig = figure();
for j = 1:6
    subplot(2, 3, j)
    u = exp(1i * x_vec * kk) * (u_hat(:, 1000*j) .* transpose(rk(1, :)));
    v = exp(1i * x_vec * kk) * (u_hat(:, 1000*j) .* transpose(rk(2, :)));
    u = reshape(u, Dim_Grid, Dim_Grid);
    v = reshape(v, Dim_Grid, Dim_Grid);
    quiver(xx, yy, u, v, 'LineWidth', 1, 'Color', 'b')
    box on        
    set(gca, 'FontSize', 12)
    title(sprintf('t = %0.2f', (1000*j)*dt), 'FontSize', 16)
    xlabel('x')
    ylabel('y');
    axis square
    xlim([-pi, pi])
    ylim([-pi, pi])
end
sgtitle('Simulation of the Incompressible Flow Field at 6 Different Time Instants', 'FontSize', 18)

fig.WindowState = 'Maximized'; % Maximize the figure window