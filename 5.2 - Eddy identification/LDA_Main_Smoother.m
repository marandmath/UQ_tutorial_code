% Code corresponding to Section 5.2 - "Eddy identification" in the Supplementary 
% Document of the paper "Taming Uncertainty in a Complex World: The Rise of 
% Uncertainty Quantification — A Tutorial for Beginners".
%
% Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
%
% Code: Calculating the posterior smoother mean and covariance for the 
% Fourier modes of the velocity field. Only the diagonal part of the filter
% covariance matrix is being kept for the calculation, which is made possible 
% by the incompressibility of the fluid, with this approximation being more 
% accurate as L grows larger. Backward (smoother-based) samples of the
% unobservable process (Fourier modes) are also generated (MATLAB code).
%
% Info: This m-file is called via Eddy_Identification.m. It calculates the 
% tracer locations through the recovered velocity field via the inverse 
% Fourier transform linear operator, using a noisy version of the acceleration 
% relation while incorporating periodic boundary conditions in [-π,π] x [-π,π]. 
% It then uses the tracer locations to calculate the posterior mean and variance 
% in an optimal manner through the smoother solution, as well as informed 
% samples of the Fourier modes which are dynamically consistent and incorporate
% uncertainty that might be lost by the deterministic solution which averages 
% out the uncertainty in one way or another. We mention also that we only keep 
% the diagonal of the recovered covariance matrix of the optimal filter 
% posterior distribution for the calculations of the smoother Gaussian
% statistics since it is known that for L (i.e. number of observations) large 
% then this matrix will converge to a diagonal one for incompressible fluids due 
% to the small Rossby numbers, which is the case here for this experiment since
% we only consider the geostrophically balanced Fourier modes in the spectral 
% decomposition of the velocity field via the local Fourier basis of planar 
% exponential waves.
%
% MATLAB Toolbox and M-file Requirements:
% 
% Code used to obtain the required m-file scripts and MATLAB toolboxes:
% [fList, pList] = matlab.codetools.requiredFilesAndProducts('LDA_Main_Smoother.m');
%
% M-file Scripts:
%   * LDA_Main_Smoother.m
%
% Toolboxes:
%   * Statistics and Machine Learning Toolbox

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(33); % Setting the seed number for consistent and reproducible results
sigma_xy = 0.01; % Noise in the Lagrangian tracer equations

% For convenience we order the components of the tracers in the observation
% vector in the following manner: First we put the x-coordinates of the L
% tracers, and then the y-coordinates by vertical concatenation, as such
% totalling a 2L-dimensional vector.
x = zeros(L, N); % Zonal or latitudinal coordinate of the tracers (x)
y = zeros(L, N); % Meridional or longitudinal coordinate of the tracers (y)

% Using a uniform distribution as an initial condition which is the
% equilibrium (or climatological prior) distribution of the tracers' locations 
% in the case of an incompressible fluid.
x(:, 1) = unifrnd(-pi, pi, L, 1); 
y(:, 1) = unifrnd(-pi, pi, L, 1);

P = zeros(2*L, Dim_U, N); % The linear and discrete inverse Fourier operator used in the observational process to recover the underlying velocity field from the Fourier modes
P(1:L, :, 1)     = exp(1i * x(:, 1) * kk(1, :) + 1i * y(:, 1) * kk(2, :)) .* (ones(L, 1) * rk(1, :));
P(L+1:2*L, :, 1) = exp(1i * x(:, 1) * kk(1, :) + 1i * y(:, 1) * kk(2, :)) .* (ones(L, 1) * rk(2, :));

% Model simulation using Euler-Maruyama for the tracers' locations.
% A stochastic system is utilized for each tracer's location, which is written 
% in a vector form that fits the conditional Gaussian nonlinear system 
% framework in the following way:
%      (Observational Processes) dx = P(x) * u_hat * dt + Σ_x * dW_x
dw_xy = randn(2*L, N-1); % Wiener noise values for the simulation
for j = 2:N

    % Generating the tracer locations.
    % x(:, j) = x(:, j-1) + exp(1i * x(:, j-1) * kk(1, :) + 1i * y(:, j-1) * kk(2, :)) * (u_hat(:, j-1) .* transpose(rk(1, :))) * dt + sigma_xy * sqrt(dt) * dw_xy(1:L, j-1);
    % y(:, j) = y(:, j-1) + exp(1i * x(:, j-1) * kk(1, :) + 1i * y(:, j-1) * kk(2, :)) * (u_hat(:, j-1) .* transpose(rk(2, :))) * dt + sigma_xy * sqrt(dt) * dw_xy(L+1:2*L, j-1);
    x(:, j) = x(:, j-1) + P(1:L, :, j-1) * u_hat(:, j-1) * dt + sigma_xy * sqrt(dt) * dw_xy(1:L, j-1);
    y(:, j) = y(:, j-1) + P(L+1:2*L, :, j-1) * u_hat(:, j-1) * dt + sigma_xy * sqrt(dt) * dw_xy(L+1:2*L, j-1);

    % Accounting for the periodic boundary conditions in [-π,π] x [-π,π].
    x(:, j) = mod(real(x(:, j)) + pi, 2*pi) - pi;
    y(:, j) = mod(real(y(:, j)) + pi, 2*pi) - pi;

    % P: Observational coefficient matrix (inverse Fourier operator).
    P(1:L, :, j)     = exp(1i * x(:, j) * kk(1, :) + 1i * y(:, j) * kk(2, :)) .* (ones(L, 1) * rk(1, :));
    P(L+1:2*L, :, j) = exp(1i * x(:, j) * kk(1, :) + 1i * y(:, j) * kk(2, :)) .* (ones(L, 1) * rk(2, :));
    
end

% Auxiliary matrices used in the filtering formulae.
S_xyoS_xy_inv = eye(2*L)/sigma_xy^2; % Grammian of the observational or tracer noise
S_u_hatoS_u_hat = Sigma_u_hat * Sigma_u_hat'; % Grammian of the flow field's noise feedback

% Posterior filter solution (Gaussian)
u_filter_mean = zeros(Dim_U, N); % Posterior filter mean of the Fourier modes vector
u_filter_cov_full = zeros(Dim_U, Dim_U, N); % Posterior covariance matrix of the Fourier modes vector; Here we keep the whole covariance matrix, not just the diagonal
u_filter_cov_diag = zeros(Dim_U, N); % Posterior covariance matrix of the Fourier modes vector; Keeping only the diagonal due to the incompressibility (becomes better as an approximation as L grows)

mu0 = u_hat(:, 1); % Initial value of the posterior filter mean
R0 = 0.0001 * eye(Dim_U); % Initial value of the posterior filter covariance matrix (choosing a positive definite matrix as to preserve positive-definiteness of the posterior filter and smoother covariance matrices)
diag_R0 = diag(R0); % Initial value of the posterior filter covariance matrix's diagonal
u_filter_mean(:, 1) = mu0;
u_filter_cov_full(:, :, 1) = R0;
u_filter_cov_diag(:, 1) = diag_R0;
for j = 2:N

    % dx term which is used to define the innovation or measurement pre-fit 
    % residual which is then multiplied by the optimal gain matrix as to obtain 
    % the optimal and refined a-posteriori state estimation.
    x_diff = x(:, j) - x(:, j-1);
    y_diff = y(:, j) - y(:, j-1);
    % Accounting for the periodic boundary conditions in [-π,π] x [-π,π].
    x_diff(x_diff > pi)  = x_diff(x_diff > pi)  - 2 * pi;
    x_diff(x_diff < -pi) = x_diff(x_diff < -pi) + 2 * pi;
    y_diff(y_diff > pi)  = y_diff(y_diff > pi)  - 2 * pi;
    y_diff(y_diff < -pi) = y_diff(y_diff < -pi) + 2 * pi;

    % Update the posterior filter mean and posterior filter covariance using the 
    % discrete update formulas.
    mu = mu0 + (f(:, j-1) - Gamma * mu0) * dt + (R0 * P(:, :, j-1)') * S_xyoS_xy_inv * ([x_diff; y_diff] - P(:, :, j-1) * mu0 * dt);
    R = R0 + (-Gamma * R0 - R0 * Gamma' + S_u_hatoS_u_hat - (R0 * P(:, :, j-1)') * S_xyoS_xy_inv * (R0 * P(:, :, j-1)')') * dt;
    u_filter_mean(:, j) = mu;
    u_filter_cov_full(:, :, j) = R;
    u_filter_cov_diag(:, j) = real(diag(R));
    mu0 = mu;
    R0 = R;
            
end

% Posterior smoother solution, also Gaussian, and backward sampling formula.
% The sampled trajectory can be used to cook up examples to show that the
% posterior covariance is crucial for eddy detection, where uncertainty will 
% break the structure if computations are based only on the mean.
u_smoother_mean = zeros(Dim_U, N); % Posterior smoother mean of the Fourier modes vector
% Posterior smoother covariance matrix of the Fourier modes vector. Here we 
% only use the diagonal of the filter covariance (made possible by the 
% incompressibility and true for L large) to calculate the Gaussian smoother
% statistics, but code is also provided (but left commented out) where the full
% filter covariance is being used instead.
u_smoother_cov = zeros(Dim_U, Dim_U, N); % Posterior smoother covariance matrix of the Fourier modes vector

% Smoothing runs backwards; "Intial" values for smoothing (at the last time 
% instant) are exactly the filter estimates.
muT = u_filter_mean(:, N); % "Initial" value of the posterior smoother mean
% Uncomment the following if you want to keep the whole filter covariance 
% matrix instead of just the diagonal (use Ctrl+T (for Windows)).
% RT = u_filter_cov_full(:, :, N); % "Initial" value of the posterior smoother covariance matrix (full filter covariance matrix at the endpoint)
RT = diag(u_filter_cov_diag(:, N)); % "Initial" value of the posterior smoother covariance matrix (diagonal of the full filter covariance matrix at the endpoint)
u_smoother_mean(:, N) = muT;
u_smoother_cov(:, :, N) = RT;
C_jj_matrices = zeros(Dim_U, Dim_U, N); % Auxiliary matrix used in the smoother formula
% C_jj_matrices(:, :, N) = eye(Dim_U) - (-Gamma + S_u_hatoS_u_hat / RT) * dt; % Up to O(Δt) 
C_jj_matrices(:, :, N) = RT * (eye(Dim_U) - Gamma * dt)' / (S_u_hatoS_u_hat * dt + (eye(Dim_U) - Gamma * dt) * RT * (eye(Dim_U) - Gamma * dt)');

rng(20); % Setting the seed number for consistent and reproducible results (needed for the sampling formula)
s_n = 20; % Number of backward (smoother-based) samples to be generated
rd_Y = randn(Dim_U, N-1, s_n); % Pre-generated Wiener noise values for sampling of the unobservable
Y_Sampling_Save = zeros(Dim_U, N, s_n); % Storing the samples of the unobservable process which are the Fourier modes

for j = N-1:-1:1

    % Update the posterior smoother mean vector and posterior smoother 
    % covariance tensor using the discrete update formulas.
    % Uncomment the following if you want to keep the whole filter covariance 
    % matrix instead of just the diagonal (use Ctrl+T (for Windows)).
    % % C_jj = eye(Dim_U) - (-Gamma + S_u_hatoS_u_hat / u_filter_cov_full(:, :, j)) * dt; % Up to O(Δt)
    % C_jj = u_filter_cov_full(:, :, j) * (eye(Dim_U) - Gamma * dt)' / (S_u_hatoS_u_hat * dt + (eye(Dim_U) - Gamma * dt) * u_filter_cov_full(:, :, j) * (eye(Dim_U) - Gamma * dt)');
    % C_jj = eye(Dim_U) - (-Gamma + S_u_hatoS_u_hat / diag(u_filter_cov_diag(:, j))) * dt; % Up to O(Δt) 
    C_jj = diag(u_filter_cov_diag(:, j)) * (eye(Dim_U) - Gamma * dt)' / (S_u_hatoS_u_hat * dt + (eye(Dim_U) - Gamma * dt) * diag(u_filter_cov_diag(:, j)) * (eye(Dim_U) - Gamma * dt)');
    C_jj_matrices(:, :, j) = C_jj;
    
    mu = u_filter_mean(:, j) + C_jj * (muT - f(:, j) * dt - (eye(Dim_U) - Gamma * dt) * u_filter_mean(:, j));
    % Uncomment the following if you want to keep the whole filter covariance 
    % matrix instead of just the diagonal (use Ctrl+T (for Windows)).
    % R = u_filter_cov_full(:, :, j) + C_jj * (RT - (eye(Dim_U) - Gamma * dt) * u_filter_cov_full(:, :, j) * (eye(Dim_U) - Gamma * dt)' - S_u_hatoS_u_hat * dt) * C_jj';
    R = diag(u_filter_cov_diag(:, j)) + C_jj * (RT - (eye(Dim_U) - Gamma * dt) * diag(u_filter_cov_diag(:, j)) * (eye(Dim_U) - Gamma * dt)' - S_u_hatoS_u_hat * dt) * C_jj';
    u_smoother_mean(:, j) = mu;
    u_smoother_cov(:, :, j) = R;
    muT = mu;
    RT = R;

    % Backward sampling of the Fourier modes. The sampled trajectory has random 
    % noise and as such we are essentially solving an SDE using the 
    % Euler-Maruyama method.
    for i = 1:s_n
        % Uncomment the following if you want to keep the whole filter 
        % covariance matrix instead of just the diagonal (use Ctrl+T (for 
        % Windows)).
        % Y_Sampling_Save(:, j, i) = Y_Sampling_Save(:, j+1, i) + (- f(:, j) + Gamma * Y_Sampling_Save(:, j+1, i)) * dt ...
        %                            + S_u_hatoS_u_hat / u_filter_cov_full(:, :, j) * (u_filter_mean(:, j) - Y_Sampling_Save(:, j+1, i)) * dt + Sigma_u_hat * sqrt(dt) * rd_Y(:, j, i); 
        Y_Sampling_Save(:, j, i) = Y_Sampling_Save(:, j+1, i) + (- f(:, j) + Gamma * Y_Sampling_Save(:, j+1, i)) * dt ...
                                   + S_u_hatoS_u_hat / diag(u_filter_cov_diag(:, j)) * (u_filter_mean(:, j) - Y_Sampling_Save(:, j+1, i)) * dt + Sigma_u_hat * sqrt(dt) * rd_Y(:, j, i); 
    end

end 