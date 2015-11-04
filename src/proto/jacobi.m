%JACOBI Prototype for the Jacobi iteration method
% Cesar Gonzalez Segura
% CMCP (Conceptos y Métodos de la Computación Paralela)
%
% This method obtains the solution for a linear
% equation system using the Jacobi iteration.
%
% Input arguments:
% - A: coefficient matrix of the linear system
% - b: independent term matrix of the linear system
% - x0: initial approximation to the solution
% - conv: convergence limit
%
% Output arguments:
% - xs: solution for the linear system
% - k: iterations needed to converge
%

function [ xs, k ] = jacobi( A, b, x0, conv )
    % Assume square matrix
    n = length(A);
    
    % Convergence is only ensured for strict
    % dominant diagonal matrices
    if ~isdominant(A)
        error('The input matrix must be diagonally strictly dominant');
    end
    
    % Divide A into L+U+D
    D = zeros(n);
    Dinv = zeros(n);
    L = zeros(n);
    U = zeros(n);
    
    for i = 1:n
        for j = 1:n
            if i == j
                % Diagonal matrix and its inverse
                if A(i,i) == 0
                    error('The input matrix diagonal elements must be non-zero');
                end
                
                D(i,i) = A(i,i);
                Dinv(i,i) = 1 / A(i,i);
            else
                % Upper triangular
                if i > j
                    L(i,j) = A(i,j);
                else
                    U(i,j) = A(i,j);
                end
            end
        end
    end
    
    % T and C matrices
    T = -Dinv*(L+U);
    C = Dinv*b;
    
    e = 100;
    xprev = x0;
    x = x0;
    k = 0;
    
    while e > conv
        x = T*xprev + C;
        xprev = x;
        
        e = norm(A*x - b);
        k = k + 1;
        
        % Halt if it takes way too much time
        if k > 100
            break;
        end
    end
    
    xs = x;
    pause;
end

