#include <sys/time.h>
#include "jacobi.h"

void saveSolution(double* x, int n)
{
	FILE *f = fopen("solution.dlm", "w+");
	int i;
	
	if (f == NULL)
	{
		fprintf(stderr, "Error saving result: %s\n", strerror(errno));
		return;
	}
	
	for (i = 0; i < n; i++)
	{
		fprintf(f, "%.12f\n", x[i]);
	}
	
	fclose(f);
}

int main (int argc, char* argv[])
{
	double conv = 0.01;
	int n, i, j, k;
	struct timeval t0, tf;
	double *A, *b, *x0, ep;
	
	if (argc < 2)
	{
		printf("Usage: jacobi [matrix_size]\n");
		exit(0);
	}
	
	srand(time(NULL));
	n = atoi(argv[1]);
	
	A = (double*) malloc(sizeof(double)*n*n);
	b = (double*) malloc(sizeof(double)*n);
	x0 = (double*) malloc(sizeof(double)*n);
	
	if (A == NULL || b == NULL || x0 == NULL)
	{
		fprintf(stderr, "Error creating input data: %s\n", strerror(errno));
		exit(1);
	}
	
	// Rellenamos A y b con valores aleatorios,
	// y x0 con todo unos
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			// Forzamos que A sea de diagonal dominante
			A[i + j*n] = 1.0 + (double)(rand() % 10);
			if (i == j) A[i + j*n] += (double)n*10.0;
		}
		
		b[i] = 1.0 + (double)(rand() % 10);
		x0[i] = 1.0;
	}
	
	gettimeofday(&t0, NULL);
	k = jacobi(A, b, x0, conv, n);
	gettimeofday(&tf, NULL);
	
	if (k < 0)
	{
		fprintf(stderr, "Error obtaining Jacobi solution: %s\n", strerror(k));
	}
	
	ep = (tf.tv_sec - t0.tv_sec) + (tf.tv_usec - t0.tv_usec)/1000000.0;
	
	printf("%d %d %.6f\n", n, k, ep);
	saveSolution(x0, n);
	
	free(A);
	free(b);
	free(x0);
	
	return 0;
}