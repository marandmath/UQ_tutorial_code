% Code corresponding to Section 4.2 - "Lagrangian DA" in the Supplementary 
% Document of the paper "Taming Uncertainty in a Complex World: The Rise of 
% Uncertainty Quantification — A Tutorial for Beginners".
%
% Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
%
% Code: Calculating the posterior filter mean and covariance for the Fourier 
% modes of the velocity field. Only the diagonal part of the filter covariance 
% matrix is being kept for the calculation, which is made possible by the 
% incompressibility of the fluid, with this approximation being more accurate as 
% L grows larger (MATLAB code).
%
% Info: This m-file is called via LDA_Function_of_L.m. It calculates the tracer
% locations through the recovered velocity field via the inverse Fourier 
% transform linear operator, using a noisy version of the acceleration relation 
% while incorporating periodic boundary conditions in [-π,π]. It then uses the 
% tracer locations to calculate the posterior mean and variance in an optimal 
% manner through the filtering solution (this code uses filtering, though a 
% better way is to use smoothing), and then calculates the relative entropy 
% between the Gaussian filter solution and the Gaussian statistical attractor or
% equilibrium distribution of the Fourier modes, known as the information gain. 
% We mention also that we only keep the diagonal of the recovered covariance 
% matrix of the optimal filter posterior distribution since it is known that 
% for L (i.e. number of observations) large then this matrix will converge to a 
% diagonal one for incompressible fluids due to the small Rossby numbers, which 
% is the case here for this experiment since we only consider the 
% geostrophically balanced Fourier modes in the spectral decomposition of the 
% velocity field via the local Fourier basis of planar exponential waves. A code
% snippet is included (but commented out), which can be used to calculate the 
% posterior smoother solution as well for better recovery of the signal.
%
% MATLAB Toolbox and M-file Requirements:
% 
% Code used to obtain the required m-file scripts and MATLAB toolboxes:
% [fList, pList] = matlab.codetools.requiredFilesAndProducts('LDA_Main_Filter.m');
%
% M-file Scripts:
%   * LDA_Main_Filter.m
%
% Toolboxes:
%   * Statistics and Machine Learning Toolbox

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(37); % Setting the seed number for consistent and reproducible results
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

% Provided there is no forcing, the equilibrium/attractor mean is no longer 
% periodic with same period just like the forcing, and instead collapses down to 
% the zero vector, while the covariance tensor has no dependence on the 
% deterministic force regardless of its form.
eq_mean = horzcat(zeros(Dim_U, 1), f) ./  (diag(Gamma) + reshape([
          1i * 2*pi/f_period * ones(1, round(Dim_U/2));
         -1i * 2*pi/f_period * ones(1, round(Dim_U/2))], [], 1)); % This is for the case where there is a periodic forcing in the flow model (if f = 0 then the equilibrium mean is zero)
eq_cov_full = (2 * Gamma) \ S_u_hatoS_u_hat;
eq_cov_diag = diag(diag(eq_cov_full));

% Quantify the uncertainty reduction using relative entropy. This measures the 
% information gain of the filter Gaussian statistics beyond the equilibrium 
% statistics of the Gaussian attractor.
Relative_Entropy_Signal = zeros(1, N);
Relative_Entropy_Dispersion = zeros(1, N);

% Posterior filter solution (smoothing can also be used after the posterior
% filter solution has been recovered, where the code for the smoothing part can 
% be found below but is left commented out).
u_post_mean = zeros(Dim_U, N); % Posterior filter mean of the Fourier modes vector
u_post_cov_full = zeros(Dim_U, Dim_U, N); % Posterior covariance matrix of the Fourier modes vector; Here we keep the whole covariance matrix, not just the diagonal
u_post_cov_diag = zeros(Dim_U, N); % Posterior covariance matrix of the Fourier modes vector; Keeping only the diagonal due to the incompressibility (becomes better as an approximation as L grows)

mu0 = u_hat(:, 1); % Initial value of the posterior filter mean
R0 = 0.0001 * eye(Dim_U); % Initial value of the posterior filter covariance matrix (choosing a positive definite matrix as to preserve positive-definiteness of the posterior filter and smoother covariance matrices)
diag_R0 = diag(R0); % Initial value of the posterior filter covariance matrix's diagonal
u_post_mean(:, 1) = mu0;
u_post_cov_full(:, :, 1) = R0;
u_post_cov_diag(:, 1) = diag_R0;
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
    u_post_mean(:, j) = mu;
    u_post_cov_full(:, :, j) = R;
    u_post_cov_diag(:, j) = real(diag(R));
    mu0 = mu;
    R0 = R;
        
    % Computing the information gain via relative entropy between the posterior 
    % filter solution and the equilibrium or climatological prior distribution
    % of the Fourier modes (both Gaussian for this system, so the 
    % signal-dispersion decomposition formula is utilized).
    Relative_Entropy_Signal(j) = real(1/2 * (mu - eq_mean(:, j-1))' / eq_cov_full * (mu - eq_mean(:, j-1)));
    Relative_Entropy_Dispersion(j) = real(1/2 * (trace( real(R) / eq_cov_full ) - Dim_U - log( det( real(R) / eq_cov_full ) )));
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Use Ctrl+T (for Windows) on the following selection to uncomment %%%%%%%
% % Posterior smoother solution, also Gaussian (see also LDA_Main_Smoother.m,
% % the m-file script which is used for Section 5.2 - "Eddy identification" in 
% % the Supplementary Document).
% u_smoother_mean = zeros(Dim_U, N); % Posterior smoother mean of the Fourier modes vector
% u_smoother_cov = zeros(Dim_U, Dim_U, N); % Posterior smoother covariance matrix of the Fourier modes vector; Here we keep the whole filter covariance matrix, not just the diagonal
% 
% % Quantify the uncertainty reduction using relative entropy. This measures the 
% % information gain of the smoother Gaussian statistics beyond the equilibrium 
% % statistics of the Gaussian attractor.
% Relative_Entropy_Signal_Smoother = zeros(1, N);
% Relative_Entropy_Dispersion_Smoother = zeros(1, N);
% 
% % Smoothing runs backwards; "Intial" values for smoothing (at the final time 
% % instant) are exactly the filter estimates.
% muT = u_post_mean(:, N); % "Initial" value of the posterior smoother mean
% RT = u_post_cov_full(:, :, N); % "Initial" value of the posterior smoother covariance matrix
% u_smoother_mean(:, N) = muT;
% u_smoother_cov(:, :, N) = RT;
% C_jj_matrices = zeros(Dim_U, Dim_U, N); % Auxiliary matrix used in the smoother formula
% % C_jj_matrices(:, :, N) = eye(Dim_U) - (-Gamma + S_u_hatoS_u_hat / RT) * dt; % Up to O(Δt) 
% C_jj_matrices(:, :, N) = RT * (eye(Dim_U) - Gamma * dt)' / (S_u_hatoS_u_hat * dt + (eye(Dim_U) - Gamma * dt) * RT * (eye(Dim_U) - Gamma * dt)');
% 
% for j = N-1:-1:1
% 
%     % Update the posterior smoother mean vector and posterior smoother 
%     % covariance tensor using the discrete update formulas after keeping only 
%     % the O(Δt) order terms for efficiency.
%     % C_jj = eye(Dim_U) - (-Gamma + S_u_hatoS_u_hat / u_post_cov_full(:, :, j)) * dt; % Up to O(Δt) 
%     C_jj = u_post_cov_full(:, :, j) * (eye(Dim_U) - Gamma * dt)' / (S_u_hatoS_u_hat * dt + (eye(Dim_U) - Gamma * dt) * u_post_cov_full(:, :, j) * (eye(Dim_U) - Gamma * dt)');    
%     C_jj_matrices(:, :, j) = C_jj;
%     mu = u_post_mean(:, j) + C_jj * (muT - f(:, j) * dt - (eye(Dim_U) - Gamma * dt) * u_post_mean(:, j)); % Up to O(Δt) 
%     R = u_post_cov_full(:, :, j) + C_jj * (RT * C_jj' - (eye(Dim_U) - Gamma * dt) * u_post_cov_full(:, :, j)); % Up to O(Δt) 
%     u_smoother_mean(:, j) = mu;
%     u_smoother_cov(:, :, j) = R;
%     muT = mu;
%     RT = R;
% 
%     % Computing the information gain via relative entropy between the posterior 
%     % smoother solution and the equilibrium or climatological prior distribution
%     % of the Fourier modes (both Gaussian for this system, so the 
%     % signal-dispersion decomposition formula is utilized).
%     Relative_Entropy_Signal_Smoother(j) = real(1/2 * ((mu - eq_mean(:, j))' / eq_cov_full * (mu - eq_mean(:, j))));
%     Relative_Entropy_Dispersion_Smoother(j) = real(1/2 * (trace( real(R) / eq_cov_full ) - Dim_U - log( det( real(R) / eq_cov_full ) )));
% 
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculating the mean of the signal and dispersion parts of the information 
% gain with respect to time for this value of L, while accounting for some 
% burn-in period initially.
Relative_Entropy_Signal_All = mean(Relative_Entropy_Signal(1000:end)); 
Relative_Entropy_Dispersion_All = mean(Relative_Entropy_Dispersion(1000:end));
% Relative_Entropy_Signal_All = mean(Relative_Entropy_Signal_Smoother(1000:end)); 
% Relative_Entropy_Dispersion_All = mean(Relative_Entropy_Dispersion_Smoother(1000:end));