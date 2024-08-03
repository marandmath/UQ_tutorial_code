% Code corresponding to Section 5.1 - "Parameter Estimation with Uncertainties 
% in Data" in the Supplementary Document of the paper "Taming Uncertainty in a 
% Complex World: The Rise of Uncertainty Quantification — A Tutorial for 
% Beginners".
%
% Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
%
% Code: Estimating the slope parameter 'a' in the following linear system:
%           xdot = dx/dt = ay,
%           ydot = dy/dt = bx.
% We can regard xdot, ydot, y and x as the four variables which we can use to 
% estimate said parameter (MATLAB code).  
% 
% Info: Only two data points will be considered, so the least-squares solution
% through standard linear regression will be needed; If there was only one data 
% point, then we can simply set a = xdot/y if y is non-zero. We will consider 
% two cases; One case where the data for both xdot and y are available, and one 
% where y is not directly observed and instead it is assigned a Gaussian 
% distribution at each realization which leads to nonlinear uncertainties in the 
% least-squares solution for the parameter.
%
% MATLAB Toolbox and M-file Requirements:
%
% [fList, pList] = matlab.codetools.requiredFilesAndProducts('Parameter_Estimation.m');
%
% M-files/Scripts:
%   * Parameter_Estimation.m
%
% Toolboxes:
%   * Statistics and Machine Learning Toolbox

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% No Uncertainty in y %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xdot = [1; 2]; % Data points of dx/dt
y = [1; 3]; % Data points of y
a_est_deterministic = sum(y.^2) \ sum(y .* xdot); % Least-squares solution through standard linear regression

% Plotting the data points against the least-squares solution from linear 
% regression, where the slope of the fitted line is exactly the estimated 
% parameter.
fig = figure();
subplot(2, 4, [1, 2])
hold on
plot(y, xdot, 'bo', 'LineWidth', 2);
plot([min(y)-1, max(y)+1], a_est_deterministic * [min(y)-1, max(y)+1], 'b', 'LineWidth', 2);
box on
set(gca, 'FontSize', 12)
legend('Data Points', 'Least-Squares Solution', 'FontSize', 14, 'Location', 'Northwest')
title('(a) Least-Squares Solution With No Uncertainty in y', 'FontSize', 18)
xlabel('y')
ylabel('dx/dt')
xlim([min(y)-1, max(y)+1])

fprintf('Estimated value of the slope parameter a by using least-squares, when y has no uncertainty: %0.4f\n', a_est_deterministic)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Gaussian Uncertainty in Unobservable y %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(42) % Setting the seed number for consistent and reproducible results

r = 10; % Uncertainty level (variance) of the unobservable y (assuming the same variance for each observation), which is <y'^2>
a_est_UQ = sum(y.^2 + r) \ sum(y .* xdot); % Least-squares solution through standard linear regression in the presence of uncertainty
fprintf('Estimated value of the slope parameter a by using least-squares, when y contains uncertainty (Gaussian with variance r = %0.2f): %0.4f\n', r, a_est_UQ)

% Generating samples from the distribution of the unobservable y.
sample_num = 5000; % Number of samples to be obtained from the distribution of y
y_samples = y + sqrt(r) * randn(2, sample_num); % Assuming a Gaussian distribution centered around the true value but with large variance

% Plotting the data points against the least-squares solution from linear 
% regression for both cases (with and without uncertainty), along with the
% sampled least-squares solutions. These samples are also plotted and are
% connected via a cyan line denoting their respective pair.
subplot(2, 4, [3, 4])
hold on
h1 = plot(y, xdot, 'bo', 'LineWidth', 2);
box on
set(gca, 'FontSize', 12)
title('(b) Least-Squares Solution With Uncertainty in y', 'FontSize', 18)
xlabel('y')
ylabel('dx/dt')
a_est_each_sample = zeros(1, sample_num);
xdot_plot_point = 3; % Up to which point in the y-axis (dx/dt = xdot) the fitted lines should be plotted
for i = 1:sample_num
    a_est_each_sample(i) = sum(y_samples(:, i).^2) \ sum(y_samples(:, i) .* xdot);  % Least-squares solution through standard linear regression in the presence of uncertainty for each sample of y values
    if i <= 100 % Plotting 100 of the sampled least-squares solutions (via the respective fitted line)
        plot([0, xdot_plot_point/a_est_each_sample(i)],  [0, xdot_plot_point], 'Color', [0.7 0.7 0.7]);
        plot(y_samples(:, i), xdot, 'k.')
        plot(y_samples(:, i), xdot, 'c')
    end
end
h2 = plot([0, xdot_plot_point/a_est_deterministic], [0, xdot_plot_point], 'b', 'LineWidth', 2);
h3 = plot([0, xdot_plot_point/a_est_UQ], [0, xdot_plot_point], 'g', 'LineWidth', 2);
h4 = plot([0, xdot_plot_point/a_est_each_sample(i)],  [0, xdot_plot_point], 'Color', [0.7 0.7 0.7]);
legend([h1, h2, h3, h4], 'Data Points', 'Least-Squares Solution (Deterministic)', 'Least-Squares Solution (With Uncertainty)', 'Solution of Individual Sampled Points', 'FontSize', 14, 'Location', 'Northeast')
xlim([min(y_samples, [], 'all'), max(y_samples, [], 'all')])
ylim([0, 2*xdot_plot_point])

% Plotting the distribution of the slope parameter, a, using the different 
% samples from y when assuming uncertainty in its observation. Its average 
% value, which is exactly the least-squares solution under uncertainty, is also 
% plotted, along with the least-squares solution of the deterministic case.
subplot(2, 4, [6, 7])
[slope_pdf, slope_value] = ksdensity(a_est_each_sample, linspace(-3, 3, 200));
hold on
h1 = plot(slope_value, slope_pdf, 'Color', [0.7 0.7 0.7], 'LineWidth', 2);
box on
set(gca, 'FontSize', 12)
h2 = xline(a_est_deterministic, 'b', 'LineWidth', 2);
h3 = xline(a_est_UQ, 'g', 'LineWidth', 2);
legend([h1, h2, h3], 'Distribution of a in the Case With Uncertainty', 'Least-Squares Solution in Deterministic Case', 'Mean Least-Squares Solution Under Uncertainty', 'FontSize', 14, 'Location', 'Northeast')
title('(c) Estimated Slope Parameter a', 'FontSize', 18)
xlabel('a')
ylabel('Probability Density')

fig.WindowState = 'Maximized'; % Maximize the figure window