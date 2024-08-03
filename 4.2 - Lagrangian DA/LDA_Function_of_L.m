% Code corresponding to Section 4.2 - "Lagrangian DA" in the Supplementary 
% Document of the paper "Taming Uncertainty in a Complex World: The Rise of 
% Uncertainty Quantification — A Tutorial for Beginners".
%
% Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
%
% Code: Lagrangian data assimilation for calculating and displaying the 
% uncertainty reduction (relative entropy) as a function of the number of 
% tracers, L, between the Gaussian posterior filter solution and the Gaussian 
% equilibrium or statistical attractor, also known as the climatological 
% prior distribution (MATLAB code).
%
% Info: This code should be run together with LDA_Main_Filter.m, after running 
% Flow_Model.m (Flow_Model.m should be run prior). LDA_Main_Filter.m will be 
% automatically called through this script, so no further input from the user is 
% required. LDA_Main_Filter.m is being used to calculate the optimal posterior 
% distribution (filter solution), as to calculate the relative entropy between 
% it and the equilibrium distribution or statistical attractor of the system 
% (also known as the climatological prior distribution). This showcases the
% logarithmic growth of the dispersion part of the relative entropy as a 
% function of the number of tracers, L. The time series of a desired GB
% Fourier mode's part (real or imaginary), with its two standard deviations,
% along with the true signal, as well as the true and recovered velocity
% field's magnitude (at some desired time point), are also plotted.
%
% MATLAB Toolbox and M-file Requirements:
% 
% Code used to obtain the required m-file scripts and MATLAB toolboxes:
% [fList, pList] = matlab.codetools.requiredFilesAndProducts('LDA_Function_of_L.m');
%
% M-file Scripts:
%   * Flow_Model.m (Implicitly (manually run by the user prior to this script), 
%     since it uses variables defined by it)
%   * LDA_Function_of_L.m
%   * LDA_Main_Filter.m
%
% Toolboxes:
%   * Statistics and Machine Learning Toolbox

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

RE_s = zeros(1, 8); % Storing the signal part of the relative entropy for each number of tracers (L)
RE_d = zeros(1, 8); % Storing the dispersion part of the relative entropy for each number of tracers (L)
LL = [2, 5, 10, 20, 30, 50, 100, 200]; % Number of tracers (L) in each test

% Spatiotemporal reconstruction of the true incompressible velocity field.
time_focus = 10000*dt; % Time for which to showcase the recovered and true velocity fields
u_truth = real(exp(1i * x_vec * kk) * (u_hat(:, round(time_focus/dt)) .* transpose(rk(1, :))));
v_truth = real(exp(1i * x_vec * kk) * (u_hat(:, round(time_focus/dt)) .* transpose(rk(2, :))));
u_truth = reshape(u_truth, Dim_Grid, Dim_Grid);
v_truth = reshape(v_truth, Dim_Grid, Dim_Grid);
magnitude_truth = sqrt(u_truth.^2 + v_truth.^2);
max_magnitude_truth = max(magnitude_truth, [], 'all'); 
min_magnitude_truth = min(magnitude_truth, [], 'all');

fig = figure();
mode_focus = 6; % Any number between 1 and Dim_U = Dimension of system = Number of Fourier modes - Origin = (2*K_max+1)^2-1 (which mode we want to plot)
real_flag = 1; % Whether to showcase the real or imaginary part of the desired Fourier mode
switch real_flag
    case 1
        part_str = 'Real';
    otherwise
        part_str = 'Imaginary';
end
for p = 1:length(LL)

    L = LL(p); % Number of Lagrangian tracers being deployed in the ocean flow
    
    LDA_Main_Filter % Calling the code which runs Lagrangian data assimilation with a given L and calculates the posterior filter mean, covariance, and information gain beyond the equilibrium distribution
    RE_s(p) = Relative_Entropy_Signal_All; % Relative entropy in the signal part
    RE_d(p) = Relative_Entropy_Dispersion_All; % Relative entropy in the dispersion part
   
    if mod(p, 2) == 1  % Only showing the results from every other test

        true_signal = real_flag * real(u_hat(mode_focus, :)) + (1 - real_flag) * imag(u_hat(mode_focus, :)); % Time series of the truth for the desired part and Fourier mode
        filter_signal = real_flag * real(u_post_mean(mode_focus, :)) + (1 - real_flag) * imag(u_post_mean(mode_focus, :)); % Time series of the posterior filter solution for the desired part and Fourier mode
        filter_std = sqrt(u_post_cov_diag(mode_focus, :)); % Standard deviation of the posterior filter solution for the desired part and Fourier mode

        subplot(2, 4, (p+1)/2) % Showing the time series of the desired GB Fourier mode's desired part (both the truth and recovered posterior filter solution)
        hold on
        plot(dt:dt:N*dt, true_signal, 'b', 'LineWidth', 2)
        plot(dt:dt:N*dt, filter_signal, 'r', 'LineWidth', 2)
        patch([dt:dt:N*dt, N*dt:-dt:dt], [filter_signal + 2 * filter_std, filter_signal(end:-1:1) - 2 * filter_std(end:-1:1)], 'r', 'FaceAlpha', 0.2, 'LineStyle', 'None')
        box on
        set(gca, 'FontSize', 12)
        if p == 7
            legend('True Signal', 'Posterior Mean (Filter)', '2 Posterior Std (Filter)', 'FontSize', 14)
        end
        title(sprintf('L = %d Observations\n%s Part of (%d,%d) Fourier Mode', L, part_str, kk(1, mode_focus), kk(2, mode_focus)), 'FontSize', 16)
        xlabel('t')
        ylabel('Amplitude')

        if p <= 5 
            subplot(2, 4, 4+(p+1)/2) % Plotting the spatiotemporal reconstruction of the incompressible velocity field at a fixed time instant for the recovered posterior filter solution for various L values
            u = real(exp(1i * x_vec * kk) * (u_post_mean(:, round(time_focus/dt)) .* transpose(rk(1, :))));
            v = real(exp(1i * x_vec * kk) * (u_post_mean(:, round(time_focus/dt)) .* transpose(rk(2, :))));
            u = reshape(u, Dim_Grid, Dim_Grid);
            v = reshape(v, Dim_Grid, Dim_Grid);
            hold on
            contourf(xx, yy, sqrt(u.^2 + v.^2), 20);
            shading interp
            colormap jet
            colorbar
            clim([min_magnitude_truth, max_magnitude_truth])
            quiver(xx, yy, u, v, 'LineWidth', 1, 'Color', 'r')
            plot(x(:, round(time_focus/dt)), y(:, round(time_focus/dt)), 'ko', 'MarkerSize', 2, 'LineWidth', 6)   
            box on    
            set(gca, 'FontSize', 12)
            title(sprintf('Recovered (Velocity Magnitude)\nL = %d Observations: t = %0.2f', L, time_focus), 'FontSize', 16)
            xlabel('x')
            ylabel('y')
            axis equal
            xlim([-pi, pi])
            ylim([-pi, pi])
        end

        if p == 7
            subplot(2, 4, 4+(p+1)/2) % Plotting the spatiotemporal reconstruction of the incompressible velocity field at a fixed time instant for the true flow field
            hold on
            contourf(xx, yy, magnitude_truth, 20);
            shading interp
            colormap jet
            colorbar
            clim([min_magnitude_truth, max_magnitude_truth])
            quiver(xx, yy, u_truth, v_truth, 'LineWidth', 1, 'Color', 'b')
            box on    
            set(gca, 'FontSize', 12)
            title(sprintf('Truth (Velocity Magnitude): t = %0.2f', time_focus), 'FontSize', 16)
            xlabel('x')
            ylabel('y')
            axis equal
            xlim([-pi, pi])
            ylim([-pi, pi])
        end
        
    end
    
end

fig.WindowState = 'Maximized'; % Maximize the figure window

% Plotting the information gain (uncertainty reduction) through its signal and
% dispersion parts when expressed as a function of L.
fig = figure();
yyaxis left
plot(LL, RE_s, '-ob', 'LineWidth', 2)
ylabel('Bits')
yyaxis right
hold on
plot(LL, RE_d, '-om', 'LineWidth', 2)
plot(LL, Dim_U*log(LL)/3, '--k', 'LineWidth', 2)
ax = gca; 
ax.YAxis(1).Color = [0 0 1];
ax.YAxis(2).Color = [1 0 1];
legend('Signal', 'Dispersion', 'O(ln(L))', 'FontSize', 14, 'Location', 'Southeast')
set(gca, 'FontSize', 12)
box on
title('Uncertainty Reduction (Information Gain) as a Function of L (Number of Observations) Beyond the Equilibrium Distribution', 'FontSize', 18)
xlabel('L (Number of Tracers)')
ylabel('Bits')

fig.WindowState = 'Maximized'; % Maximize the figure window