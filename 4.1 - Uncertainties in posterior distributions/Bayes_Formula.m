% Code corresponding to Section 4.1 - "Uncertainties in posterior distributions" 
% in the Supplementary Document of the paper "Taming Uncertainty in a Complex 
% World: The Rise of Uncertainty Quantification — A Tutorial for Beginners".
%
% Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
%
% Code: Illustration of Bayes' formula, as a basis for data assimilation, in the
% case of a one dimensional state variable and a single or multiple 
% observation(s)/measurement(s) (MATLAB code).
%
% Info: The Bayesian update as a basis for data assimilation is showcased, where 
% observations are utilized to update our prior beliefs and obtain the optimal 
% posterior distribution. Here we assume the simplified case of both a Gaussian 
% prior as well as likelihood. We also showcase the asymptotic behavior of the 
% posterior statistics, as well as the relative entropy between the prior and 
% posterior Gaussian distributions (posterior distribution is also Gaussian due 
% to the conjugacy), when they are treated as functions of the number of 
% observations. Specifically, the logarithmic growth of the dispersion part of 
% the relative entropy is displayed, which signifies the diminishing returns 
% that stem from unboundedly increasing the number of observations.
%
% MATLAB Toolbox and M-file Requirements:
% 
% Code used to obtain the required m-file scripts and MATLAB toolboxes:
% [fList, pList] = matlab.codetools.requiredFilesAndProducts('Bayes_Formula.m');
%
% M-file Scripts:
%   * Bayes_Formula.m
%
% Toolboxes:
%   * Statistics and Machine Learning Toolbox

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% A Single Observation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u = linspace(-4, 4, 500); % Domain of the random state variable
rng(42) % Setting the seed number for consistent and reproducible results

truth = 1; % Truth (true state/value of u)

% Observational noise and noisy observation/measurement.
ro = 1; % Observational noise, namely the variance
g = 1; % Observational operator
v = g * truth + sqrt(ro) * randn; % Observation/measurement

% Prior.
mu_f = -1; % Prior mean
R_f = 1; % Prior variance
p_prior = normpdf(u, mu_f, sqrt(R_f)); % Prior normal distribution

% Likelihood.
p_likelihood = normpdf(u, v, sqrt(ro)); % Normal likelihood

% Kalman gain.
K = R_f * g' * (g * R_f * g' + ro)^(-1);

% Posterior.
mu_a = mu_f + K * (v - g * mu_f); % Posterior mean
R_a = (1 - K * g) * R_f; % Posterior variance
p_posterior = normpdf(u, mu_a, sqrt(R_a)); % Posterior normal distribution

% Plotting the distributions for a single observation.
fig = figure();
subplot(2, 2, 1)
hold on
plot(u, p_prior, 'g', 'LineWidth', 2)
plot(u, p_likelihood, 'c', 'LineWidth', 2)
plot(u, p_posterior, 'r', 'LineWidth', 2)
xline(truth, 'k--', 'LineWidth', 2)
box on
set(gca, 'FontSize', 12);
legend('Prior Distribution', 'Likelihood', 'Posterior Distribution', 'True Value', 'FontSize', 14, 'Location', 'Northwest')
title('(a) Distributions; Single Observation', 'FontSize', 18)
xlabel('u (State Variable)')
ylabel('Probability Density')
xlim([min(u), max(u)])
ylim([0, 2*max([max(p_prior), max(p_likelihood), max(p_posterior)])])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Multiple Observations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Observational noise and noisy observations/measurements.
L = 10; % Number of observations
ro = 1 * eye(L); % Observational noise, namely the covariance matrix
g = 1 * ones(L, 1); % Observational operator
v = g * truth + sqrtm(ro) * randn(L, 1); % Observations/measurements

% Prior.
mu_f = -1; % Prior mean
R_f = 1; % Prior variance
p_prior = normpdf(u, mu_f, sqrt(R_f)); % Prior normal distribution

% Kalman gain.
K = R_f * g' * (g * R_f * g' + ro)^(-1);

% Posterior.
mu_a = mu_f + K * (v - g * mu_f); % Posterior mean
R_a = (1 - K * g) * R_f; % Posterior variance
p_posterior = normpdf(u, mu_a, sqrt(R_a)); % Posterior normal distribution

% Plotting the distributions for multiple observations.
subplot(2, 2, 2)
hold on
plot(u, p_prior, 'g', 'LineWidth', 2)
plot(u, p_posterior, 'r', 'LineWidth', 2)
xline(truth, 'k--', 'LineWidth', 2)
box on
set(gca, 'FontSize', 12);
legend('Prior Distribution', 'Posterior Distribution', 'True Value', 'FontSize', 14, 'Location', 'Northwest')
title(sprintf('(b) Distributions; L = %d Observations', L), 'FontSize', 18)
xlabel('u (State Variable)')
ylabel('Probability Density')
xlim([min(u), max(u)])
ylim([0, 2*max([max(p_prior), max(p_posterior)])])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Asymptotic Behavior %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For each L (i.e. number of observations/measurements), we repeat the 
% experiment Test_num times to obtain the mean behavior and the confidence 
% interval, since the observational noise differs in each test.
Test_num = 100; % Number of times that the experiment will be repeated
L_all = [1, 2, 5, 10, 20, 30, 100, 200, 500]; % Testing different L (number of observations)
mu_all = zeros(length(L_all), Test_num); % Storing the posterior mean for each number of observations (L) and experiment
R_all = zeros(length(L_all), Test_num); % Storing the posterior variance for each number of observations (L) and experiment

for i = 1:length(L_all)
    for j = 1:Test_num
        
        % Observational noise and noisy observations/measurements.
        L = L_all(i); % Number of observations
        ro = 1 * eye(L); % Observational noise, namely the covariance matrix
        g = 1 * ones(L, 1); % Observational operator
        v = g * truth + sqrtm(ro) * randn(L, 1); % Observations/measurements
 
        % Prior.
        mu_f = -1; % Prior mean
        R_f = 1; % Prior variance

        % Kalman gain.
        K = R_f * g' * (g * R_f * g' + ro)^(-1);
        
        % Posterior.
        mu_a = mu_f + K * (v - g * mu_f); % Posterior mean
        R_a = (1 - K * g) * R_f; % Posterior variance

        mu_all(i, j) = mu_a;
        % In fact, R does not change for different values of the random noise, 
        % as the noise only affects the mean while the posterior variance 
        % depends on the observational operator only (in this case the number of 
        % observations, L), but for completeness we store this matrix here.
        R_all(i, j) = R_a; 

    end
end

% Plotting the results for the asymptotic behavior of the posterior mean and
% variance as functions of the number of observations, L, when averaging over 
% the repeated experiments, as well as the information gain (uncertainty 
% reduction) through its signal and dispersion parts when expressed as a 
% function of L.
subplot(2, 2, 3)
hold on
yyaxis left
h1 = plot(L_all, mean(mu_all, 2), 'b', 'LineWidth', 2);
post_upper_mu = mean(mu_all, 2) + 2 * std(mu_all, 0, 2);
post_lower_mu = mean(mu_all, 2) - 2 * std(mu_all, 0, 2);
p = patch([L_all, L_all(end:-1:1)], [post_lower_mu.', post_upper_mu(end:-1:1).'], 'b', 'FaceAlpha', 0.2, 'LineStyle', 'None');
% Posterior mean asymptotes at the limit of (Σ_{l=1,...,L} v_l)/L as L tends to 
% infinity, which in this case ends up being 1 (the true value), since g_l = 1, 
% u = 1 (true value), and because the sum of the Gaussian error terms satisfy 
% E(Σ_{l=1,...,L} ε_l) = 0 and Var(Σ_{l=1,...,L} ε_l) = L/(L+1)^2 -> 0, 
% as L grows to infinity.
h2 = yline(1, 'k--', 'LineWidth', 2); 
ylabel('\mu_a (Posterior Mean)')
yyaxis right
h3 = plot(L_all, mean(R_all, 2), 'Color', [1 0 1], 'LineWidth', 2);
ax = gca; 
ax.YAxis(1).Color = [0 0 1];
ax.YAxis(2).Color = [1 0 1];
box on
set(gca, 'FontSize', 12);
legend([h1, p, h2, h3], 'Posterior Mean', '2 Std Posterior Mean', 'True Value', 'Posterior Variance', 'FontSize', 14, 'Location', 'Southeast')
title('(c) Asymptotic Behavior of Posterior Statistics', 'FontSize', 18)
xlabel('L (Number of Measurements)')
ylabel('R_a (Posterior Variance)')
subplot(2, 2, 4)
signal = 1/2 * (mean(mu_all, 2) - mu_f).^2 / R_f;
dispersion = 1/2 * (mean(R_all, 2) / R_f - 1 - log( mean(R_all, 2) / R_f ));
hold on
yyaxis left
h4 = plot(L_all, signal, 'b', 'LineWidth', 2);
box on
set(gca, 'FontSize', 12);
xlabel('L (Number of Measurements)')
ylabel('Bits')
yyaxis right
hold on
h5 = plot(L_all, dispersion, '-m', 'LineWidth', 2);
h6 = plot(L_all, log( L_all + 1 ) / 2 - L_all / (L_all+1) / 2, '--k', 'LineWidth', 2);
box on
set(gca, 'FontSize', 12);
ax = gca; 
ax.YAxis(1).Color = [0 0 1];
ax.YAxis(2).Color = [1 0 1];
box on
set(gca, 'FontSize', 12);
legend([h4, h5, h6], 'Signal', 'Dispersion', '(ln(L+1)-L/(L+1))/2', 'FontSize', 14, 'Location', 'Southeast')
title('(d) Uncertainty Reduction (Relative Entropy)', 'FontSize', 18)
ylabel('Bits')

fig.WindowState = 'Maximized'; % Maximize the figure window