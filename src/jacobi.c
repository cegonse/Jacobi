#include "jacobi.h"

int jacobi(double *A, double *b, double *x0, double conv, int n)
{
	double *Dinv = NULL, *T = NULL, *C = NULL, *x1 = NULL, e = conv + 1.0;
	int k = 0;

	// Ejecutar sólo si la matriz de términos independientes
	// es estrictamente dominante.
	if (!isDominant(A, n)) return ENONDOM;
	
	// Reserva de  memoria para la inversa de la diagonal
	// de la matriz A
	Dinv = (double*) malloc(sizeof(double)*n*n);
	if (Dinv == NULL) return errno;
	
	// Obtenemos la inversa de la diagonal de A
	Dinv = diaginv(A, n, Dinv);
	
	// Reserva de memoria para las matrices T y C
	// y para el vector x1
	T = (double*) malloc(sizeof(double)*n);
	C = (double*) malloc(sizeof(double)*n);
	x1 = (double*) malloc(sizeof(double)*n);
	
	if (T == NULL || C == NULL || x1 == NULL)
	{
		free(Dinv);
		return errno;
	}
	
	// Obtenemos T y C
	T = matscal(matmat(1, Dinv, A, T, n), -1.0, n);
	C = matvec(1, Dinv, b, C, n);
	
	printf("\n\n");
	
	printf("T:\n");
	printMat(1,T,n);
	printf("\n\n");
	
	printf("C:\n");
	printVec(C,n);
	printf("\n\n");
	
	while (/*e > conv*/0)
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


void printVec(double* a, int n)
{
	int i;
	
	printf("[ ");
	
	for (i = 0; i < n; i++)
	{
		if (i != n - 1)
		{
			printf("%.2f\n", a[i]);
		}
		else
		{
			printf("%.2f ]\n", a[i]);
		}
	}
}


void printMat(int isdiag, double* A, int n)
{
	int i, j;
	
	for (i = 0; i < n; i++)
	{
		printf("[ ");
		
		for (j = 0; j < n; j++)
		{
			if (isdiag)
			{
				if (i == j)
				{
					printf("%.2f ", A[i]);
				}
				else
				{
					printf("0 ");
				}
			}
			else
			{
				printf("%.2f ", A[i + j*n]);
			}
		}
		
		printf("]\n");
	}
}


double* matscal(double* A, double s, int n)
{
	int i;
	
	for (i = 0; i < n*n; i++)
	{
		A[i] *= s;
	}

	return A;
}


double* matvec(int isdiag, double *A, double *b, double *C, int n)
{
	int i, j;
	
	// En función de si es una matriz diagonal o una matriz
	// completa, esperamos sólo los valores de la diagonal como
	// matriz A o la matriz completa
	if (isdiag)
	{
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				if (i == j) C[i] += A[i] * b[i];
			}
		}
	}
	else
	{
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				C[i] += A[i + j*n] * b[i];
			}
		}
	}

	return C;
}


double* matmat(int isdiag, double *A, double *B, double *C, int n)
{
	int i, j, k;

	// En función de si es una matriz diagonal o una matriz
	// completa, esperamos sólo los valores de la diagonal como
	// matriz A o la matriz completa
	if (isdiag)
	{
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				for (k = 0; k < n; k++)
				{
					if (i == j)
					{
						C[i + j*n] += A[i] * B[k + j*n];
					}
					else
					{
						C[i + j*n] = 0;
					}
				}
			}
		}
	}
	else
	{
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
				for (k = 0; k < n; k++)
				{
					C[i + j*n] += A[i + j*n] * B[k + j*n];
				}
			}
		}
	}

	return C;
}


double* diaginv(double *A, int n, double *diag)
{
	int i;
	
	for (i = 0; i < n; i++)
	{
		diag[i] =  1 / A[i + n*i];
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