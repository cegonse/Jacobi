#include "mathsub.h"
#include <math.h>
#include <string.h>

void dnrm2_seq(int n, double *x, double* nrm)
{
	*nrm = 0.0;
	int i, nc;
	
	nc = n - (n % 4);
	
	for (i = 0; i < nc; i += 4)
	{
		*nrm += x[i]*x[i];
		*nrm += x[i+1]*x[i+1];
		*nrm += x[i+2]*x[i+2];
		*nrm += x[i+3]*x[i+3];
	}
	
	if (nc != n) {
		for (i = nc; i < n; i++) *nrm += x[i]*x[i];
	}
	
	*nrm = sqrt(*nrm);
}


void dgemm_seq(int n, int m, double alpha, double *A, double *B, double beta, double *C)
{
	int i, k, j;
	
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			C[i + j*n] = 0.0;
			
			for (k = 0; k < m; k++) 
			{
				C[i + j*n] += alpha * A[i + k*m] * B[k + j*m];
			}
		}
    }
}


void dgemv_seq(int n, int m, double alpha, double *A, double *b, double beta, double *c)
{
	int i, j;
	
	for (i = 0; i < n; i++)
	{
		c[i] = 0.0;
		
		for (j = 0; j < m; j++)
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
	int i, nc;
	
	nc = n - (n % 4);
	
	for (i = 0; i < nc; i += 4)
	{
		y[i] = alpha * x[i] + y[i];
		y[i+1] = alpha * x[i+1] + y[i+1];
		y[i+2] = alpha * x[i+2] + y[i+2];
		y[i+3] = alpha * x[i+3] + y[i+3];
	}
	
	if (nc != n) {
		for (i = nc; i < n; i++) y[i] = alpha * x[i] + y[i];
	}
}
