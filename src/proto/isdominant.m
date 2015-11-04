%JACOBI Prototype for the diagonal dominant matrix check
%       method
%
% Cesar Gonzalez Segura
% CMCP (Conceptos y Métodos de la Computación Paralela)
%
% This method checks if the input matrix is row diagonal
% dominant.
%
% Input arguments:
% - A: input matrix
%
% Output arguments:
% - dom: true if it is dominant, false if it is not
%

function [ dom ] = isdominant( A )
    n = length(A);
    dom = true;
    
    for i = 1:n
        diag = A(i,i);
        rownm = 0;
        
        for j = 1:n
            if j ~= i
                rownm = A(i,j)^2;
            end
        end
        
        if diag < sqrt(rownm)
            dom = false;
        end
    end
end

