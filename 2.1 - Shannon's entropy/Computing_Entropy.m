% Code corresponding to Section 2.1 - "Shannon’s entropy" in the Supplementary 
% Document of the paper "Taming Uncertainty in a Complex World: The Rise of 
% Uncertainty Quantification — A Tutorial for Beginners".
%
% Authors: Nan Chen, Stephen Wiggins, Marios Andreou.
% 
% Code: Computing Shannon's entropy (MATLAB code).
%
% Info: Shannon's differential entropy is calculated numerically and via
% analytic formulae for a Gaussian and a Gamma random variable.
%
% MATLAB Toolbox and M-file Requirements:
% 
% Code used to obtain the required m-file scripts and MATLAB toolboxes:
% [fList, pList] = matlab.codetools.requiredFilesAndProducts('Computing_Entropy.m');
%
% M-file Scripts:
%   * Computing_Entropy.m
%
% Toolboxes:
%   * Statistics and Machine Learning Toolbox

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian Distribution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculating Shannon's entropy using numerical integration and the analytic 
% formula.
x1 = linspace(-10, 10, 500); % Domain of the Gaussian random variable
x = x1;
mu = 0; R = 1; % Mean and variance of the Gaussian distribution
fprintf('Mean of Gaussian = %0.4f | Variance of Gaussian = %0.4f\n', mu, R) % Displaying the leading two moments
p_Gauss = 1/sqrt( 2 * pi * R ) * exp( -(x - mu).^2 / 2 / R ); % Computing the Gaussian PDF over the domain (p_Gauss = normpdf(x, mu, R))
p_Gauss = p_Gauss / trapz(x, p_Gauss); % Normalization to eliminate small numerical errors due to the finite measure of the domain
entropy_numerical = - trapz(x, p_Gauss .* log( p_Gauss )); % Numerically computing Shannon's entropy using the definition
entropy_theoretic = 1/2 * log( 2 * pi ) + 1/2 * log( R ) + 1/2; % Analytic formula of computing Shannon's entropy for a Gaussian distribution
fprintf('Shannon entropy for the Gaussian distribution N(%0.3f, %0.3f):\nNumerical value: %0.4f\nTheoretical value: %0.4f\n', mu, R, entropy_numerical, entropy_theoretic)

% Plotting the PDF and the theoretical Shannon entropy for the Gaussian 
% random variable.
fig = figure(); 
plot(x, p_Gauss, 'b', 'LineWidth', 2)
box on
set(gca, 'FontSize', 12)
title(sprintf('PDF of Gaussian Distribution With Mean \\mu = %0.4f and Variance R = %0.4f', mu, R), 'FontSize', 18)
xlabel('x')
ylabel('p(x) (Probability Density)')
xlim([min(x), max(x)])
ylim([min(p_Gauss), max(p_Gauss)])
text(0.95*min(x), 0.95*max(p_Gauss), sprintf('Entropy = %0.4f', entropy_theoretic), 'FontSize', 16)

fig.WindowState = 'Maximized'; % Maximize the figure window

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gamma Distribution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculating Shannon's entropy using numerical integration and the analytic 
% formula.
x2 = linspace(0.01, 20.01, 500); % Domain of the Gamma random variable (x must be positive since Gamma distributions are supported on the positive reals)
x = x2;
k = 2; theta = sqrt(2); % Shape and scale parameters of the Gamma distribution
fprintf('\nMean of Gamma(%0.3f, %0.3f) = %0.4f | Variance of Gamma(%0.3f, %0.3f) = %0.4f | Skewness of Gamma(%0.3f, %0.3f) = %0.4f | Kurtosis of Gamma(%0.3f, %0.3f) = %0.4f\n', ...
    k, theta, k * theta, ...
    k, theta, k * theta^2, ...
    k, theta, 2 / sqrt( k ), ...
    k, theta, 6 / k ...
) % Displaying the leading four moments
p_Gamma = gampdf(x, k, theta); % Computing the Gamma PDF over the domain
peak_value = max(p_Gamma); % Computing the peak value
p_Gamma = p_Gamma / trapz(x, p_Gamma); % Normalization to eliminate small numerical errors due to the finite measure of the domain
entropy_numerical = - trapz(x, p_Gamma .* log( p_Gamma )); % Numerically computing Shannon's entropy using the definition
entropy_theoretic = k + log( theta ) + log( gamma( k ) ) + (1 - k) * psi( k ); % Analytic formula of computing Shannon's entropy for a Gamma distribution
fprintf('Shannon entropy for the Gamma distribution Gamma(%0.3f, %0.3f):\nNumerical value: %0.4f\nTheoretical value: %0.4f\n', k, theta, entropy_numerical, entropy_theoretic)

% Plotting the PDF and the theoretical Shannon entropy for the Gamma random
% variable.
fig = figure(); 
plot(x, p_Gamma, 'b', 'LineWidth', 2)
box on
set(gca, 'FontSize', 12)
title(sprintf('PDF of Gamma Distribution With Shape k = %0.4f and Scale \\theta = %0.4f', k, theta), 'FontSize', 18)
xlabel('x')
ylabel('p(x) (Probability Density)')
xlim([min(x), max(x)])
ylim([min(p_Gamma), max(p_Gamma)])
text(0.85*max(x), 0.95*peak_value, sprintf('Entropy = %0.4f', entropy_theoretic), 'FontSize', 16)

fig.WindowState = 'Maximized'; % Maximize the figure window