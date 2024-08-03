% Code corresponding to Section 5.2 - "Eddy identification" in the Supplementary 
% Document of the paper "Taming Uncertainty in a Complex World: The Rise of 
% Uncertainty Quantification — A Tutorial for Beginners".
%
% Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
%
% Code: Uses the OW parameter as the criterion for identifying eddies in a 2D 
% random amplitude incompressible velocity field constructed by a linear 
% stochastic model (OU process) for the Fourier modes, where the flow field is 
% recovered by passive tracers using Lagrangian data assimilation via the 
% posterior smoother Gaussian distribution (MATLAB code).
%
% Info: This code should be run together with LDA_Main_Smoother.m and 
% Flow_Model.m. Both Flow_Model.m and LDA_Main_Smoother.m will be 
% automatically called through this script so no further input from the user 
% is required. Flow_Model.m is being used to generate an underlying 2D
% incompressible velocity field via a spectral decomposition which uses local 
% Fourier basis functions for the geostrophically balanced modes by further 
% assigning to them an OU process. On the other hand, LDA_Main_Smoother.m is 
% being used to calculate the optimal posterior distribution (smoother 
% solution), as to showcase the eddy diagnostic at a specific diagnostic time. 
% The adoption of the OW parameter is used as the criterion for identifying 
% eddies; When the OW parameter is negative, the relative vorticity is larger 
% than the strain components, indicating vortical flow. Backward sampled 
% realizations or smoother-based samples of the unobservable Fourier modes are 
% also generated, which while not the same as the truth, are still crucial as to 
% be collected and construct a PDF describing the statistical behavior of the 
% OW parameter. In the presence of uncertainty, such uncertainty quantification 
% in the diagnostics is important since the deterministic solution arising from 
% averaging may lose a crucial amount of information.
%
% MATLAB Toolbox and M-file Requirements:
% 
% Code used to obtain the required m-file scripts and MATLAB toolboxes:
% [fList, pList] = matlab.codetools.requiredFilesAndProducts('Eddy_Identification.m');
%
% M-file Scripts:
%   * Eddy_Identification.m
%   * Flow_Model.m
%   * LDA_Main_Smoother.m
%
% Toolboxes:
%   * Statistics and Machine Learning Toolbox

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

L = 5; % Number of Lagrangian tracers being deployed in the ocean flow
Flow_Model; % Generating the background 2D incompressible flow field
LDA_Main_Smoother; % Calculating the posterior filter and smoother Gaussian statistics as well as the backward (smoother-based) samples of the unobservable
u_post_mean = u_smoother_mean; % Posterior smoother mean
u_post_cov = u_smoother_cov; % Posterior smoother covariance (keeping only the diagonal of the filter covariance during its calculation)

t_test = 2; % Diagnostic time used to check the flow field recovery and eddy identification skill
t_index = round(t_test/dt);

% Spatiotemporal reconstruction of the true incompressible velocity field.
u = real(exp(1i * x_vec * kk) * (u_hat(:, t_index) .* transpose(rk(1,:))));
v = real(exp(1i * x_vec * kk) * (u_hat(:, t_index) .* transpose(rk(2,:))));
u = reshape(u, Dim_Grid, Dim_Grid);
v = reshape(v, Dim_Grid, Dim_Grid);
u_truth = u;
v_truth = v;

fig = figure();
% OW parameter calculation using the truth.
subplot(2, 3, 1)
% Note that in MATLAB, when approximating derivatives via finite differences 
% with the diff() function the rows and columns correspond to the y and x 
% directions, respectively. As such, the derivative of y (x) should be along the 
% row (column) direction!
uy = diff(u); uy = ([uy(end, :); uy] + [uy; uy(1, :)]) / 2; % y-derivative of the zonal or latitudinal (x-coordinate) velocity
vy = diff(v); vy = ([vy(end, :); vy] + [vy; vy(1, :)]) / 2; % y-derivative of the meridional or longitudinal (y-coordinate) velocity
ux = diff(u'); ux = ux'; ux = ([ux(:, end), ux] + [ux, ux(:, 1)]) / 2; % x-derivative of the zonal or latitudinal (x-coordinate) velocity
vx = diff(v'); vx = vx'; vx = ([vx(:, end), vx] + [vx, vx(:, 1)]) / 2; % x-derivative of the meridional or longitudinal (y-coordinate) velocity
% Computing the OW parameter.
sn = ux - vy; % Normal strain
ss = vx + uy; % Shear strain
w = vx - uy; % Relative vorticity (vorticity's z-coordinate/signed magnitude)
OW_truth = sn.^2 + ss.^2 - w.^2; 
OW_std = std(reshape(OW_truth, 1, [])); 
OW_truth = OW_truth/OW_std; % Normalizing the OW parameter as to render the parameter scale invariant
% Plotting the true OW parameter over the spatial domain.
contourf(xx, yy, OW_truth, 30, 'LineStyle', 'None')
shading interp
colormap jet
max_OW_truth = max(OW_truth, [], 'all'); 
min_OW_truth = min(OW_truth, [], 'all');
colorbar
clim([min_OW_truth, max_OW_truth])
hold on
quiver(xx, yy, u_truth, v_truth, 'LineWidth', 1, 'Color', 'b') % Plotting the true velocity field on top of the OW parameter for reference
box on
set(gca, 'FontSize', 12)
title('(a) $\mbox{OW}[\mathbf{u}^{\mbox{truth}}]$', 'Interpreter', 'LaTeX', 'FontSize', 18)
xlabel('x')
ylabel('y')
axis equal
xlim([-pi, pi])
ylim([-pi, pi])

% OW parameter estimation using the posterior smoother mean.
subplot(2, 3, 2)
% Spatiotemporal reconstruction of the incompressible velocity field using the 
% posterior smoother mean.
u = real(exp(1i * x_vec * kk) * (u_post_mean(:, t_index) .* transpose(rk(1, :))));
v = real(exp(1i * x_vec * kk) * (u_post_mean(:, t_index) .* transpose(rk(2, :))));
u = reshape(u, Dim_Grid, Dim_Grid);
v = reshape(v, Dim_Grid, Dim_Grid);
% Note that in MATLAB, when approximating derivatives via finite differences 
% with the diff() function the rows and columns correspond to the y and x 
% directions, respectively. As such, the derivative of y (x) should be along the 
% row (column) direction!
uy = diff(u); uy = ([uy(end, :); uy] + [uy; uy(1, :)]) / 2; % y-derivative of the zonal or latitudinal (x-coordinate) velocity
vy = diff(v); vy = ([vy(end, :); vy] + [vy; vy(1, :)]) / 2; % y-derivative of the meridional or longitudinal (y-coordinate) velocity
ux = diff(u'); ux = ux'; ux = ([ux(:, end), ux] + [ux, ux(:, 1)]) / 2; % x-derivative of the zonal or latitudinal (x-coordinate) velocity
vx = diff(v'); vx = vx'; vx = ([vx(:, end), vx] + [vx, vx(:, 1)]) / 2; % x-derivative of the meridional or longitudinal (y-coordinate) velocity
% Computing the OW parameter.
sn = ux - vy; % Normal strain
ss = vx + uy; % Shear strain
w = vx - uy; % Relative vorticity (vorticity's z-coordinate/signed magnitude)
OW_post_mean = (sn.^2 + ss.^2 - w.^2) / OW_std; % Normalising with respect to the true standard deviation for uniformity
% Plotting the recovered OW parameter (via the posterior smoother mean) over the 
% spatial domain.
contourf(xx, yy, OW_post_mean, 30, 'LineStyle', 'None')
shading interp
colormap jet
colorbar
clim([min_OW_truth, max_OW_truth])
box on
set(gca, 'FontSize', 12)
hold on
quiver(xx, yy, u_truth, v_truth, 'LineWidth', 1, 'Color', 'b') % Plotting the true velocity field on top of the OW parameter for reference
title('(b) $\mbox{OW}(\bar{\mathbf{u}})$', 'Interpreter', 'LaTeX', 'FontSize', 18)
xlabel('x')
ylabel('y')
axis equal
xlim([-pi, pi])
ylim([-pi, pi])

% OW parameter estimation using the expectation from the backward 
% (smoother-based) generated samples of the Fourier modes.
subplot(2, 3, 3)
OW_sampling = zeros(size(OW_post_mean));
for s = 1:s_n
    %  Spatiotemporal reconstruction of the incompressible velocity field using
    % this specific backward sample.
    u_sampling = squeeze(Y_Sampling_Save(:, :, s));
    u = real(exp(1i * x_vec * kk) * (u_sampling(:, t_index) .* transpose(rk(1,:))));
    v = real(exp(1i * x_vec * kk) * (u_sampling(:, t_index) .* transpose(rk(2,:))));
    u = reshape(u, Dim_Grid, Dim_Grid);
    v = reshape(v, Dim_Grid, Dim_Grid);
    % Note that in MATLAB, when approximating derivatives via finite differences 
    % with the diff() function the rows and columns correspond to the y and x 
    % directions, respectively. As such, the derivative of y (x) should be along the 
    % row (column) direction!
    uy = diff(u); uy = ([uy(end, :); uy] + [uy; uy(1, :)]) / 2; % y-derivative of the zonal or latitudinal (x-coordinate) velocity
    vy = diff(v); vy = ([vy(end, :); vy] + [vy; vy(1, :)]) / 2; % y-derivative of the meridional or longitudinal (y-coordinate) velocity
    ux = diff(u'); ux = ux'; ux = ([ux(:, end), ux] + [ux, ux(:, 1)]) / 2; % x-derivative of the zonal or latitudinal (x-coordinate) velocity
    vx = diff(v'); vx = vx'; vx = ([vx(:, end), vx] + [vx, vx(:, 1)]) / 2; % x-derivative of the meridional or longitudinal (y-coordinate) velocity
    % Computing the OW parameter.
    sn = ux - vy; % Normal strain
    ss = vx + uy; % Shear strain
    w = vx - uy; % Relative vorticity (vorticity's z-coordinate/signed magnitude)
    OW_sampling = OW_sampling + (sn.^2 + ss.^2 - w.^2); % Addidng to the previous result as to take the average at the end
end
OW_sampling = OW_sampling / s_n; % Sample mean which approximates the expectation
OW_sampling = OW_sampling / OW_std; % Normalising with respect to the true standard deviation for uniformity
% Plotting the recovered OW parameter (via averaging over the backward 
% (smoother-based) samples of the Fourier modes) over the spatial domain.
contourf(xx, yy, OW_sampling, 30, 'LineStyle', 'None')
shading interp
colormap jet
colorbar
clim([min_OW_truth, max_OW_truth])
box on
set(gca, 'FontSize', 12)
hold on
quiver(xx, yy, u_truth, v_truth, 'LineWidth', 1, 'Color', 'b') % Plotting the true velocity field on top of the OW parameter for reference
title('(c) $E[\mbox{OW}(\mathbf{u})]$', 'Interpreter', 'LaTeX', 'FontSize', 18)
xlabel('x')
ylabel('y')
axis equal
xlim([-pi, pi])
ylim([-pi, pi])

% Showing the time series of the desired GB Fourier mode's desired part (both 
% the truth and recovered posterior smoother solution).
subplot(2, 3, 4:6)
mode_focus = 2; % Any number between 1 and Dim_U = Dimension of system = Number of Fourier modes - Origin = (2*K_max+1)^2-1 (which mode we want to plot)
real_flag = 1; % Whether to showcase the real or imaginary part of the desired Fourier mode
switch real_flag
    case 1
        part_str = 'Real';
    otherwise
        part_str = 'Imaginary';
end
true_signal = real_flag * real(u_hat(mode_focus, :)) + (1 - real_flag) * imag(u_hat(mode_focus, :)); %  Time series of the truth for the desired part and Fourier mode
smoother_signal = real_flag * real(u_post_mean(mode_focus, :)) + (1 - real_flag) * imag(u_post_mean(mode_focus, :)); %  Time series of the posterior smoother solution for the desired part and Fourier mode
smoother_std = squeeze(sqrt(real(u_post_cov(mode_focus, mode_focus, :)))).'; % Standard deviation of the posterior smoother solution for the desired part and Fourier mode
hold on
h1 = plot(dt:dt:N*dt, true_signal, 'b', 'LineWidth', 2);
h2 = plot(dt:dt:N*dt, smoother_signal, 'r', 'LineWidth', 2);
h3 = patch([dt:dt:N*dt, N*dt:-dt:dt], [smoother_signal + 2 * smoother_std, smoother_signal(end:-1:1) - 2 * smoother_std(end:-1:1)], 'r', 'FaceAlpha', 0.2, 'LineStyle', 'None');
h4 = plot(t_test, true_signal(round(t_test/dt)), 'go', 'LineWidth', 6);
box on
set(gca, 'FontSize', 12)
legend([h1, h2, h3, h4], 'Truth','Posterior Mean (Smoother)','2 Posterior Std (Smoother)','Diagnostic Time (for OW)')
title(sprintf('L = %d Observations: %s Part of (%d,%d) Fourier Mode', L, part_str, kk(1, mode_focus), kk(2, mode_focus)), 'FontSize', 18)
xlabel('t')
ylabel('Amplitude')

fig.WindowState = 'Maximized'; % Maximize the figure window

% Plotting the recovered OW parameter by using 15 of the backward 
% (smoother-based) samples generated for the Fourier modes.
fig = figure();
for s = 1:15
    subplot(3, 5, s)
    % Spatiotemporal reconstruction of the incompressible velocity field using
    % this specific backward sample.
    u_sampling = squeeze(Y_Sampling_Save(:, :, s));
    u = real(exp(1i * x_vec * kk) * (u_sampling(:, t_index) .* transpose(rk(1,:))));
    v = real(exp(1i * x_vec * kk) * (u_sampling(:, t_index) .* transpose(rk(2,:))));
    u = reshape(u, Dim_Grid, Dim_Grid);
    v = reshape(v, Dim_Grid, Dim_Grid);
    % Note that in MATLAB, when approximating derivatives via finite differences 
    % with the diff() function the rows and columns correspond to the y and x 
    % directions, respectively. As such, the derivative of y (x) should be along the 
    % row (column) direction!
    uy = diff(u); uy = ([uy(end, :); uy] +  [uy; uy(1, :)]) / 2; % y-derivative of the zonal or latitudinal (x-coordinate) velocity
    vy = diff(v); vy = ([vy(end, :); vy] +  [vy; vy(1, :)]) / 2; % y-derivative of the meridional or longitudinal (y-coordinate) velocity
    ux = diff(u'); ux = ux'; ux = ([ux(:, end), ux] +  [ux, ux(:, 1)]) / 2; % x-derivative of the zonal or latitudinal (x-coordinate) velocity
    vx = diff(v'); vx = vx'; vx = ([vx(:, end), vx] +  [vx, vx(:, 1)]) / 2; % x-derivative of the meridional or longitudinal (y-coordinate) velocity
    % Computing the OW parameter.
    sn = ux - vy; % Normal strain
    ss = vx + uy; % Shear strain
    w = vx - uy; % Relative vorticity (vorticity's z-coordinate/signed magnitude)
    OW_sampling = (sn.^2 + ss.^2 - w.^2) / OW_std; % Normalising with respect to the true standard deviation for uniformity
    % Plotting the recovered OW parameter for this specific sample over the 
    % spatial domain.
    contourf(xx, yy, OW_sampling, 30, 'LineStyle', 'None')
    shading interp
    colormap jet
    colorbar
    clim([min_OW_truth, max_OW_truth])
    box on
    set(gca, 'FontSize', 12)
    hold on
    quiver(xx, yy, u_truth, v_truth, 'LineWidth', 1, 'Color','b') % Plotting the true velocity field on top of the OW parameter for reference
    title(sprintf('Sampled realization: %d', s), 'FontSize', 14)
    xlabel('x')
    ylabel('y')
    axis equal
    xlim([-pi, pi])
    ylim([-pi, pi])
end

fig.WindowState = 'Maximized'; % Maximize the figure window