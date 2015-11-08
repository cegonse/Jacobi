#include "jacobi.h"


void printVec(double* a, int n)
{
	int i;
	
	for (i = 0; i < n; i++)
	{
		printf("[ %.6f ]\n", a[i]);
	}
}


void printMat(double* A, int n)
{
	int i, j;
	
	for (i = 0; i < n; i++)
	{
		printf("[ ");
		
		for (j = 0; j < n; j++)
		{
			printf("%.6f ", A[i + j*n]);
		}
		
		printf("]\n");
	}
}