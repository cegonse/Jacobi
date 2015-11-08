#include "mathsub.h"


double dnrm2_seq(int n, double *x)
{
	double nrm;
	int i;
	
	for (i = 0; i < n; i++)
	{
		nrm += x[i]*x[i];
	}
	
	return sqrt(nrm);
}


void dgemm_seq(int n, double alpha, double *A, double *B, double beta, double *C)
{
	int i, k, j;

	for (k = 0; k < n; k++) 
	{
		for (j = 0; j < n; j++)
		{
			for (i = 0; i < n; i++)
			{
				C[i + j*n] += alpha * A[i + k*n] * B[k + j*n] + beta*C[i + j*n];
			}
		}
    }
}


void dgemv_seq(int n, double alpha, double *A, double *b, double beta, double *c)
{
	int i, j;
	
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			c[i] = alpha * A[i + j*n] * x[i] + beta * c[i];
		}
	}
}

