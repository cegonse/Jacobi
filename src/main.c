#include "jacobi.h"

int main (int argc, char* argv[])
{
	double *A, *b, *x0, conv = 0.01;
	int n = 8, i, j;
	
	A = (double*) malloc(sizeof(double)*n*n);
	b = (double*) malloc(sizeof(double)*n);
	x0 = (double*) malloc(sizeof(double)*n);
	
	srand(time(NULL));
	
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			A[i + j*n] = 1 + rand() % 10;
			
			if (i == j) A[i + j*n] += 10.0;
			
			b[i] = 1 + rand() % 10;
			x0[i] = 0.0;
		}
	}
	
	printf("Ax = b\n\n");
	
	printf("A:\n");
	printMat(0, A, n);
	printf("\n\n");
	
	printf("b:\n");
	printVec(b, n);
	printf("\n\n");
	
	jacobi(A, b, x0, conv, n);
	
	printf("x:\n");
	printVec(x0, n);
	printf("\n\n");
	
	free(A);
	free(b);
	free(x0);
	
	return 0;
}