#include "jacobi.h"

int jacobi(double *A, double *b, double *x0, double conv, int n)
{
	double *Dinv = NULL, *T = NULL, *C = NULL, *x1 = NULL, *LPU = NULL, e = conv + 1.0;
	int i, j, k = 0;

	// Ejecutar sólo si la matriz de términos independientes
	// es estrictamente dominante.
	if (!isDominant(A, n)) return ENONDOM;
	
	// Reserva de  memoria para la inversa de la diagonal
	// de la matriz A
	Dinv = (double*) malloc(sizeof(double)*n*n);
	if (Dinv == NULL) return errno;
	
	// Obtenemos la inversa de la diagonal de A
	Dinv = diaginv(A, n, Dinv);
	
	// Reserva de memoria para las matrices T, LPU y C
	// y para el vector x1
	T = (double*) malloc(sizeof(double)*n*n);
	LPU = (double*) malloc(sizeof(double)*n*n);
	C = (double*) malloc(sizeof(double)*n);
	x1 = (double*) malloc(sizeof(double)*n);
	
	if (T == NULL || C == NULL || x1 == NULL || LPU == NULL)
	{
		free(Dinv);
		return errno;
	}
	
	// Separamos la matriz A en L+U+D
	// Sólo almacenamos el resultado de L+U
	for (i = 0; i < n; i++)
	{
        for (j = 0; j < n; j++)
		{
			if (i != j) LPU[i + j*n] = A[i + j*n];
        }
    }
	
	// Obtenemos las matrices T y C
	// Utilizando estas matrices, transformamos
	// la operación a la forma x1 = T*x0 + C
	//
	// T = -Dinv * LPU
	// C = Dinv * b
	//
	char trans = 'N';
	double alpha = -1.0, beta = 1.0;
	
	// 1: T = -Dinv * LPU
	//dgemm_(&trans, &trans, &n, &n, &n, &alpha, Dinv, &n, LPU, &beta, T, &n); Versión de Fortran
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, Dinv, n, LPU, n, beta, T, n); // Versión CBLAS
	
	// Iteramos hasta alcanzar la razón de convergencia
	//while (e > conv)
	{
		k++;
	}
	
	// Liberamos la memoria utilizada
	free(Dinv);
	free(T);
	free(C);
	free(x1);
	
	return k;
}


double* diaginv(double *A, int n, double *diag)
{
	int i;
	
	for (i = 0; i < n; i++)
	{
		diag[i] =  1.0 / A[i + n*i];
	}
	
	return diag;
}


int isDominant(double *A, int n)
{
	int dom = 1, i, j;
	double diag = 0.0, rowmn = 0.0, rowmn2 = 0.0;
	
	for (i = 0; i < n; i++)
	{
		diag = A[i + i*n];
	
		for (j = 0; j < n; j++)
		{
			if (j != i)
			{
				rowmn2 = A[i + j*n];
				rowmn = rowmn2 * rowmn2;
			}
		}
		
		if (diag < sqrt(rowmn))
		{
			dom = 0;
		}
	}
	
	return dom;
}