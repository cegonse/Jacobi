% Jacobi method test
%
% Cesar Gonzalez Segura
% CMCP (Conceptos y Métodos de la Computación Paralela)
%
% This script evaluates the performance of the prototype
% Jacobi iteration method, obtaining the solution for a
% linear system using different convergence thresholds.
%

close all;
clear all;

% Convergence thresholds
e = zeros(1,15);

for i = 1:length(e)
    e(i) = 10^-i;
end

% Number of equations and of variables
n = 6;

% Coefficient matrix, independent terms
% and initial solution
A = rand(n) + (eye(n).*10);
b = rand(n); b = b(:,1);
x0 = ones(1, n)';

% Obtain time needed to solve the system
% using the built-in method
t0 = tic();
x_builtin = inv(A)*b;
t_builtin = toc(t0);
tv_builtin = ones(1,length(e)) .* t_builtin;

tm = zeros(1,length(e));
km = zeros(1,length(e));
em = zeros(1,length(2));

for i = 1:length(e)
    % Obtain solution for current convergence threshold
    t0 = tic();
    [x, k] = jacobi(A, b, x0, e(i));
    tm(i) = toc(t0);
    km(i) = k;
    em(i) = mean(x_builtin - x)^2;
end

% Plot time needed to solve
figure(1);
subplot(2,1,1);
loglog(e,tm);
title('Time needed to solve for each threshold');
xlabel('Threshold');
ylabel('Time (seconds)');

% Plot iterations needed to solve
subplot(2,1,2);
semilogx(e, km);
title('Iterations needed to solve for each threshold');
xlabel('Threshold');
ylabel('Iterations');

% Plot error between built-in solution
% and Jacobi solution
figure(2);
loglog(e, em);
title('MSE of the Jacobi solution compared to the built-in solution');
xlabel('Threshold');
ylabel('Mean Square Error');