% Code corresponding to Section 2.2 - "Relative entropy" in the Supplementary 
% Document of the paper "Taming Uncertainty in a Complex World: The Rise of 
% Uncertainty Quantification — A Tutorial for Beginners".
%
% Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
% 
% Code: Computing the relative entropy for Gaussian distributions (MATLAB code).
%
% Info: Results from the analytic formula and numerical integration will be 
% compared for the relative entropy between two Gaussian distributions; One will
% be considered the ground truth and the other one will be the model 
% distribution. Two cases will be considered; One where both distributions are 
% known, and one where they are both unknown and instead constructed via 
% samples. Special caution is required when dealing with the tail values of the 
% distribution appearing in the denominator of the relative entropy's definition 
% (as well as the reference/true distribution for consistency).
%
% MATLAB Toolbox and M-file Requirements:
% 
% Code used to obtain the required m-file scripts and MATLAB toolboxes:
% [fList, pList] = matlab.codetools.requiredFilesAndProducts('Computing_Relative_Entropy.m');
%
% M-file Scripts:
%   * Computing_Relative_Entropy.m
%
% Toolboxes:
%   * Statistics and Machine Learning Toolbox

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Gaussian Distribution Given by Analytic Formulae %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Defining the reference/ground truth and model Gaussian distributions.
m = 1; % Dimension of the random variable
x = linspace(-10, 10, 500); % Domain of the random variable
mu = 0; R = 1; % Mean and variance of reference/ground truth distribution (R is the covariance matrix for m > 1)
fprintf('Mean of p = %0.4f | Variance of p = %0.4f\n', mu, R) % Displaying the leading two moments
p = 1/sqrt( 2 * pi * R ) * exp( -(x - mu).^2 / 2 / R ); % Computing the PDF of p over the domain (p = normpdf(x, mu, sqrt(R)), or mvnpdf(x, mu, R) for m > 1 with R being the covariance matrix)
mu_M = 3; R_M = 1; % Mean and variance of the model distribution (R_M is the covariance matrix for m > 1)
fprintf('Mean of p^M = %0.4f | Variance of p^M = %0.4f\n', mu_M, R_M)  % Displaying the leading two moments
p_M = 1/sqrt(2 * pi * R_M) * exp( -(x - mu_M).^2 / 2 / R_M ); % Computing the PDF of pᴹ over the domain (p_M = normpdf(x, mu_M, sqrt(R_M)), or mvnpdf(x, mu_M, R_M) for m > 1 with R_M being the covariance matrix)

% Plotting the PDFs defined by the analytic formulae.
fig = figure(); 
subplot(2, 2, 1)
hold on
plot(x, p, 'b', 'LineWidth', 2);
plot(x, p_M, 'r', 'LineWidth', 2);
box on
set(gca, 'FontSize', 12)
legend('p(x)', 'p^M(x)', 'FontSize', 14, 'Location', 'Northwest')
title('(a) p(x) Computed Analytically', 'FontSize', 18)
xlabel('x')
ylabel('Probability Density')
ylim([0, 1.1*max(max(p), max(p_M))])
subplot(2, 2, 3) % Plotting the PDFs on a logarithmic scale to better see the tail behavior
hold on
plot(x, p, 'b', 'LineWidth', 2);
plot(x, p_M, 'r', 'LineWidth', 2);
box on
set(gca, 'FontSize', 12)
set(gca, 'YScale', 'Log')
title('Y-axis in Logarithmic Scale', 'FontSize', 18)
xlabel('x')
ylabel('Probability Density')
xlim([min(x), max(x)])
ylim([1e-20, 1])

% Computing the relative entropy using the signal-dispersion decomposition 
% formula since both distributions are Gaussian.
R_M_inv = R_M^(-1);
Signal = 1/2 * (mu - mu_M)' * R_M_inv * (mu - mu_M);
Dispersion = 1/2 * (trace( R * R_M_inv ) - m - log( det( R * R_M_inv ) ));
RE_theoretic = Signal + Dispersion;

% Computing the relative entropy using the definition and numerical integration, 
% where no consideration is made about the tails of the model distribution in
% the denominator (or the one corresponding to the truth).
RE_numerical_no_normalization = trapz(x, p .* log( p ./ p_M ));

% Computing the relative entropy using the definition and numerical integration, 
% where the tail probability is set to be a small but nonzero value to account
% for the PDF ratio, and then a corrective normalization is applied to the PDFs. 
p_no_remedy = p;
p_M_no_remedy = p_M;
p( p <= 1e-5 ) = 1e-5;
p_M( p_M <= 1e-5 ) = 1e-5;
p = p / trapz(x, p);
p_M = p_M / trapz(x, p_M);
RE_numerical_with_normalization = trapz(x, p .* log( p ./ p_M ));

fprintf('\nGaussian distributions given by analytic formulae\n')
fprintf('Relative entropy:\nTheoretical value: %0.4f\nNumerical value without normalization: %0.4f\nNumerical value with normalization: %0.4f\n', ...
    RE_theoretic, RE_numerical_no_normalization, RE_numerical_with_normalization ...
)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Gaussian Distribution Constructed From Samples %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(42) % Setting the seed number for consistent and reproducible results
sample_number = 10000; % Number of samples to be used to recover the PDFs

% Generating samples to reconstruct the PDFs numerically.
Gaussian_rd = normrnd(mu, sqrt(R), [1, sample_number]); % Use mvnrnd(mu, R, sample_number) when m > 1 for multivariate normal random samples (R is the covariance matrix)
Gaussian_rd_M = normrnd(mu_M, sqrt(R_M), [1, sample_number]); % Use mvnrnd(mu_M, R_M, sample_number) when m > 1 for multivariate normal random samples (R_M is the covariance matrix)

% Reconstructed PDFs from the generated samples using KDE with normal kernels.
p_sampled = ksdensity(Gaussian_rd, x); % mvksdensity(Gaussian_rd, x) for m > 1
p_M_sampled = ksdensity(Gaussian_rd_M, x); % mvksdensity(Gaussian_rd_M, x) for m > 1

% Numerically computing the mean and variance.
mu_sampled = mean(Gaussian_rd);
R_sampled = cov(Gaussian_rd);
mu_M_sampled = mean(Gaussian_rd_M);
R_M_sampled = cov(Gaussian_rd_M);

% Plotting the PDFs obtained from the samples.
subplot(2, 2, 2)
hold on
plot(x, p_sampled, 'b', 'LineWidth', 2);
plot(x, p_M_sampled, 'r', 'LineWidth', 2);
box on
set(gca, 'FontSize', 12)
title('(b) p(x) Computed Based on Samples', 'FontSize', 18)
xlabel('x')
ylabel('Probability Density')
ylim([0, 1.1*max(max(p_sampled), max(p_M_sampled))])
subplot(2, 2, 4) % Plotting the PDFs on a logarithmic scale to better see the tail behavior
hold on
plot(x, p_sampled, 'b', 'LineWidth', 2);
plot(x, p_M_sampled, 'r', 'LineWidth', 2);
plot(x, p_no_remedy, 'Color', [0, 0, 1, 0.2], 'LineStyle', ':', 'LineWidth', 2);
plot(x, p_M_no_remedy, 'Color', [1, 0, 0, 0.2], 'LineStyle', ':', 'LineWidth', 2);
box on
set(gca, 'FontSize', 12)
set(gca, 'YScale', 'Log')
title('Y-axis in Logarithmic Scale', 'FontSize', 18)
xlabel('x')
ylabel('Probability Density')
xlim([min(x), max(x)])
ylim([1e-20, 1])

fig.WindowState = 'Maximized'; % Maximize the figure window

% Computing the relative entropy using the signal-dispersion decomposition 
% formula since both distributions are Gaussian.
R_M_sampled_inv = R_M_sampled^(-1);
Signal_sampled = 1/2 * (mu_sampled - mu_M_sampled)' * R_M_sampled_inv * (mu_sampled - mu_M_sampled);
Dispersion_sampled = 1/2 * (trace( R_sampled * R_M_sampled_inv ) - m - log( det( R_sampled * R_M_sampled_inv ) ));
RE_theoretic_sampled = Signal_sampled + Dispersion_sampled;

% Directly computing the relative entropy based on the PDFs using the definition
% and numerical integration, which suffers from the undersampling of the tail 
% events, with no consideration about the tails of the model distribution
% in the denominator (or the one corresponding to the truth).
RE_numerical_no_normalization_sampled = trapz(x, p_sampled .* log( p_sampled ./ p_M_sampled ));

% Computing the relative entropy using the definition and numerical integration, 
% where the tail probability is set to be a small but nonzero value to account
% for the PDF ratio, and then a corrective normalization is applied to the PDFs.  
p_sampled( p_sampled <= 1e-5 ) = 1e-5;
p_M_sampled( p_M_sampled <= 1e-5 ) = 1e-5;
p_sampled = p_sampled / trapz(x, p_sampled);
p_M_sampled = p_M_sampled / trapz(x, p_M_sampled);
RE_numerical_with_normalization_sampled = trapz(x, p_sampled .* log( p_sampled ./ p_M_sampled ));

fprintf('\nGaussian distribution constructed from samples\n')
fprintf('Relative entropy:\nTheoretical value: %0.4f\nNumerical value without normalization: %0.4f\nNumerical value with normalization: %0.4f\n', ...
    RE_theoretic_sampled, RE_numerical_no_normalization_sampled, RE_numerical_with_normalization_sampled ...
)