#include "jacobi.h"


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