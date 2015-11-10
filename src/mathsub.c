#include "mathsub.h"
#include <math.h>
#include <string.h>

double dnrm2_seq(int n, double *x)
{
	double nrm;
	int i;
	
	for (i = 0; i < n; i ++)
	{
		nrm += x[i]*x[i];
	}
	
	return sqrt(nrm);
}


void dgemm_seq(int n, double alpha, double *A, double *B, double beta, double *C)
{
	int i, k, j;
	
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			C[i + j*n] = 0.0;
			
			for (k = 0; k < n; k++) 
			{
				C[i + j*n] += alpha * A[i + k*n] * B[k + j*n];// + beta * C[i + j*n];
			}
		}
    }
}


void dgemv_seq(int n, double alpha, double *A, double *b, double beta, double *c)
{
	int i, j;
	
	for (i = 0; i < n; i++)
	{
		c[i] = 0.0;
		
		for (j = 0; j < n; j++)
		{
			c[i] += alpha * A[i + j*n] * b[j] + beta * c[i];
		}
	}
}


void dcopy_seq(int n, double *x, double *y)
{
	memcpy(y, x, sizeof(double) * n);
}

void daxpy_seq(int n, double alpha, double *x, double *y)
{
	int i;
	
	for (i = 0; i < n; i++)
	{
		y[i] = alpha * x[i] + y[i];
	}
}
